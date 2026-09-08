"""Peak GPU memory and step/example latency for train and eval compares."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers.trainer_callback import TrainerCallback

from src.progress import report


def gpu_memory_mb():
    if not torch.cuda.is_available():
        return {"allocated_mb": None, "reserved_mb": None, "peak_allocated_mb": None}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
    }


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] * (1.0 - frac) + ordered[hi] * frac, 4)


def series_stats(values, prefix, unit):
    if not values:
        return {}
    mean = sum(values) / len(values)
    suffix = f"_{unit}" if unit else ""
    return {
        f"mean_{prefix}{suffix}": round(mean, 4),
        f"p50_{prefix}{suffix}": percentile(values, 50),
        f"p90_{prefix}{suffix}": percentile(values, 90),
        f"p95_{prefix}{suffix}": percentile(values, 95),
        f"p99_{prefix}{suffix}": percentile(values, 99),
        f"min_{prefix}{suffix}": round(min(values), 4),
        f"max_{prefix}{suffix}": round(max(values), 4),
    }


def latency_stats(times, prefix="step"):
    return series_stats(times, prefix, "s")


class EfficiencyCallback(TrainerCallback):
    def __init__(self, output_path, method="qlora"):
        self.output_path = Path(output_path)
        self.method = method
        self.step_times = []
        self.allocated_mb_hist = []
        self.reserved_mb_hist = []
        self.last_step_t = None
        self.train_t0 = None
        self.samples_per_step = None

    def _payload(self, state=None, args=None, final=False):
        times = self.step_times[1:] or self.step_times
        elapsed = time.perf_counter() - self.train_t0 if self.train_t0 else None
        samples_per_step = self.samples_per_step
        if samples_per_step is None and args is not None:
            samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
            self.samples_per_step = samples_per_step
        mem_now = gpu_memory_mb()
        payload = {
            "method": self.method,
            "phase": "train",
            "final": final,
            "steps": len(self.step_times),
            "global_step": None if state is None else state.global_step,
            "elapsed_s": round(elapsed, 2) if elapsed else None,
            "samples_per_step": samples_per_step,
            "last_step_s": round(times[-1], 4) if times else None,
            "note": (
                "Timings are per optimizer step (microbatch x grad_accum). "
                "allocated_* is live torch allocated at step end; "
                "reserved_* is the CUDA caching allocator pool; "
                "peak_allocated_mb is CUDA high-water since train start."
            ),
            **latency_stats(times, "step"),
            **series_stats(self.allocated_mb_hist, "allocated", "mb"),
            **series_stats(self.reserved_mb_hist, "reserved", "mb"),
            **mem_now,
        }
        mean_step = payload.get("mean_step_s")
        if mean_step and samples_per_step:
            payload["samples_per_sec"] = round(samples_per_step / mean_step, 3)
            payload["mean_sample_s"] = round(mean_step / samples_per_step, 4)
        return payload

    def _flush(self, state=None, args=None, final=False):
        payload = self._payload(state=state, args=args, final=final)
        write_json(self.output_path, payload)
        return payload

    def on_train_begin(self, args, state, control, **kwargs):
        reset_peak_memory()
        self.step_times = []
        self.allocated_mb_hist = []
        self.reserved_mb_hist = []
        self.last_step_t = time.perf_counter()
        self.train_t0 = self.last_step_t
        self.samples_per_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
        report(phase="profile", message="efficiency tracking started", **gpu_memory_mb())
        self._flush(state=state, args=args)

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        dt = now - self.last_step_t
        self.last_step_t = now
        self.step_times.append(dt)
        mem = gpu_memory_mb()
        if mem["allocated_mb"] is not None:
            self.allocated_mb_hist.append(mem["allocated_mb"])
            self.reserved_mb_hist.append(mem["reserved_mb"])
        payload = self._flush(state=state, args=args)
        if state.global_step % 10 == 0 or state.global_step <= 2:
            report(
                phase="profile",
                message=(
                    f"step {state.global_step} {dt:.2f}s "
                    f"p50 {payload.get('p50_step_s')} "
                    f"p95 {payload.get('p95_step_s')} "
                    f"peak {mem['peak_allocated_mb']} MB"
                ),
                step_latency_s=round(dt, 3),
                **{k: payload.get(k) for k in (
                    "mean_step_s", "p50_step_s", "p90_step_s",
                    "p95_step_s", "p99_step_s",
                    "mean_allocated_mb", "p95_allocated_mb",
                )},
                **mem,
            )

    def on_train_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        payload = self._flush(state=state, args=args, final=True)
        print("Wrote efficiency report:", self.output_path)
        report(phase="profile", message="train efficiency written", **payload)


class InferenceTimer:
    def __init__(self, method="qlora"):
        self.method = method
        self.latencies = []
        reset_peak_memory()

    def time_call(self, fn):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.latencies.append(time.perf_counter() - t0)
        return result

    def summary(self, extra=None):
        times = self.latencies
        payload = {
            "method": self.method,
            "phase": "eval",
            "examples": len(times),
            **latency_stats(times, "latency"),
            "examples_per_sec": (
                round(len(times) / sum(times), 3) if times and sum(times) else None
            ),
            **gpu_memory_mb(),
        }
        if extra:
            payload.update(extra)
        return payload
