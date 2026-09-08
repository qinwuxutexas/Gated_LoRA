"""Colab workplace worker: heartbeat + Drive job queue."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jobs import (
    load_config,
    log_path,
    next_queued_job,
    repo_root,
    update_job,
    write_heartbeat,
)
from src.progress import read_progress, write_progress

POLL_SECONDS = 8
HEARTBEAT_SECONDS = 10


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        return None
    return None


def prepare_local_model(cfg: dict) -> None:
    local_model = Path(cfg["workplace"]["local_model"])
    drive_model = Path(cfg["workplace"]["drive_model"])
    if local_model.exists():
        print("Local model ready:", local_model)
        return
    if not drive_model.exists():
        print("Drive model missing:", drive_model)
        return
    print("Copying model from Drive to local SSD (several minutes, not idle)...")
    print("From:", drive_model)
    print("To:  ", local_model)
    shutil.copytree(drive_model, local_model)
    print("Local model ready:", local_model)


def build_command(job: dict, root: Path) -> list[str]:
    params = job["params"]
    if job["type"] == "train":
        script = root / "scripts" / "train_qlora.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_name", str(params["model_name"]),
            "--train_manifest", str(params["train_manifest"]),
            "--eval_manifest", str(params["eval_manifest"]),
            "--output_dir", str(params["output_dir"]),
            "--data_root", str(params["data_root"]),
            "--lora_rank", str(params.get("lora_rank", 8)),
            "--batch_size", str(params.get("batch_size", 8)),
            "--grad_accum", str(params.get("grad_accum", 4)),
            "--epochs", str(params.get("epochs", 1)),
            "--learning_rate", str(params.get("learning_rate", 2e-4)),
            "--save_steps", str(params.get("save_steps", 200)),
        ]
        return cmd

    if job["type"] == "gated_train":
        script = root / "scripts" / "train_gated_lora.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_name", str(params["model_name"]),
            "--train_manifest", str(params["train_manifest"]),
            "--eval_manifest", str(params["eval_manifest"]),
            "--output_dir", str(params["output_dir"]),
            "--data_root", str(params["data_root"]),
            "--lora_rank", str(params.get("lora_rank", 8)),
            "--batch_size", str(params.get("batch_size", 8)),
            "--grad_accum", str(params.get("grad_accum", 4)),
            "--epochs", str(params.get("epochs", 1)),
            "--learning_rate", str(params.get("learning_rate", 2e-4)),
            "--gate_init", str(params.get("gate_init", 2.0)),
            "--save_steps", str(params.get("save_steps", 200)),
        ]
        if params.get("lora_alpha") is not None:
            cmd.extend(["--lora_alpha", str(params["lora_alpha"])])
        return cmd

    if job["type"] == "eval":
        script = root / "scripts" / "eval_qlora.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_name", str(params["model_name"]),
            "--lora_path", str(params["lora_path"]),
            "--eval_manifest", str(params["eval_manifest"]),
            "--output_file", str(params["output_file"]),
            "--data_root", str(params["data_root"]),
            "--max_new_tokens", str(params.get("max_new_tokens", 32)),
        ]
        if params.get("max_samples") is not None:
            cmd.extend(["--max_samples", str(params["max_samples"])])
        if params.get("adapter_type"):
            cmd.extend(["--adapter_type", str(params["adapter_type"])])
        return cmd

    if job["type"] == "gated_eval":
        script = root / "scripts" / "eval_qlora.py"
        cmd = [
            sys.executable,
            str(script),
            "--model_name", str(params["model_name"]),
            "--lora_path", str(params["lora_path"]),
            "--eval_manifest", str(params["eval_manifest"]),
            "--output_file", str(params["output_file"]),
            "--data_root", str(params["data_root"]),
            "--adapter_type", "gated",
            "--max_new_tokens", str(params.get("max_new_tokens", 32)),
        ]
        if params.get("max_samples") is not None:
            cmd.extend(["--max_samples", str(params["max_samples"])])
        return cmd

    raise ValueError(f"Unknown job type: {job['type']}")


class Heartbeat:
    def __init__(self):
        self.current_job_id = None
        self.gpu = gpu_name()
        self._stop = threading.Event()

    def payload(self) -> dict:
        extra = {
            "gpu": self.gpu,
            "current_job_id": self.current_job_id,
        }
        if self.current_job_id:
            extra["progress"] = read_progress(self.current_job_id)
        return extra

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                write_heartbeat(self.payload())
            except Exception as exc:
                print("heartbeat failed:", exc)
            self._stop.wait(HEARTBEAT_SECONDS)

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        self._stop.set()


def run_job(job: dict, root: Path) -> None:
    job_id = job["id"]
    log_file = log_path(job_id, root)
    cmd = build_command(job, root)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print("Running", job_id, cmd)
    write_progress(job_id, phase="start", message="worker launched command", last_line=" ".join(cmd))
    update_job(job_id, status="running", message="running on Colab")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["GATED_LORA_JOB_ID"] = job_id

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    last_flush = 0.0
    with log_file.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        assert proc.stdout is not None
        for line in proc.stdout:
            handle.write(line)
            handle.flush()
            now = time.time()
            stripped = line.strip()
            if stripped and now - last_flush >= 8:
                write_progress(job_id, last_line=stripped[:300])
                last_flush = now

    returncode = proc.wait()
    if returncode == 0:
        update_job(job_id, status="succeeded", message="finished")
        write_progress(job_id, message="finished", percent=100)
        print("Succeeded", job_id)
    else:
        update_job(job_id, status="failed", error=f"exit {returncode}", message=f"failed (exit {returncode})")
        write_progress(job_id, message=f"failed (exit {returncode})")
        print("Failed", job_id, "exit", returncode)


def main() -> None:
    root = repo_root()
    cfg = load_config(root)
    print("Workplace root:", root)
    print("Requested GPU:", cfg["workplace"]["gpu"])

    gpu = gpu_name()
    if gpu:
        print("Visible GPU:", gpu)
        if "A100" not in gpu:
            print("Warning: runtime is not A100. Switch Colab runtime type to A100.")
    else:
        print("Warning: no CUDA GPU visible.")

    beat = Heartbeat()
    try:
        write_heartbeat(beat.payload())
        print("Heartbeat written.")
    except OSError as exc:
        print("Warning: first heartbeat failed:", exc)
    beat.start()
    print("Colab workplace online.")
    print("If this is a new runtime, the next step copies Qwen3-VL-8B to local SSD.")
    print("Training waits until that copy finishes. Leave this cell running.")
    prepare_local_model(cfg)
    print("Waiting for Drive jobs...")

    try:
        while True:
            job = next_queued_job(root)
            if job is None:
                time.sleep(POLL_SECONDS)
                continue

            beat.current_job_id = job["id"]
            update_job(job["id"], status="running", gpu=gpu)
            write_heartbeat(beat.payload())
            try:
                run_job(job, root)
            except Exception as exc:
                update_job(job["id"], status="failed", error=str(exc))
                print("Job crashed:", job["id"], exc)
            finally:
                beat.current_job_id = None
                write_heartbeat(beat.payload())
    except KeyboardInterrupt:
        print("Workplace stopped.")
    finally:
        beat.stop()


if __name__ == "__main__":
    main()
