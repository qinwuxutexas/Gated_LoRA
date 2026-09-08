"""Submit and watch Colab jobs from the local Cursor terminal."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jobs import (
    get_job,
    list_jobs,
    log_path,
    new_job,
    read_heartbeat,
    update_job,
)
from src.progress import read_progress


def cmd_status(_: argparse.Namespace) -> int:
    wp = read_heartbeat()
    print(f"workplace: {'ONLINE' if wp.get('online') else 'OFFLINE'}")
    print(f"gpu:       {wp.get('gpu') or wp.get('requested_gpu') or 'A100'}")
    print(f"job:       {wp.get('current_job_id') or 'idle'}")
    print(f"heartbeat: {wp.get('updated_at') or 'none'}")
    if not wp.get("online"):
        print()
        print(wp.get("reason") or "Colab worker is not connected.")
        print_colab_help()
    return 0


def print_colab_help() -> None:
    print()
    print("Start the GPU worker in Colab (this laptop cannot train):")
    print("  1. Browser: colab.research.google.com")
    print("  2. File -> Open notebook -> Google Drive")
    print("  3. gated_lora/workplace/colab_worker.ipynb")
    print("  4. Runtime -> Change runtime type -> GPU -> A100")
    print("  5. Run all cells and leave the last cell running")
    print("Then this terminal will start printing step/loss/log lines.")


def cmd_list(_: argparse.Namespace) -> int:
    jobs = list_jobs()
    if not jobs:
        print("No jobs.")
        return 0
    print(f"{'ID':12}  {'TYPE':5}  {'STATUS':10}  MESSAGE")
    for job in jobs:
        progress = read_progress(job["id"]) or {}
        message = progress.get("message") or job.get("message") or ""
        print(f"{job['id']:12}  {job['type']:5}  {job['status']:10}  {message}")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    job_type = args.command.replace("-", "_")
    job = new_job(job_type)
    print(f"queued {args.command} job {job['id']}")
    wp = read_heartbeat()
    if not wp.get("online"):
        print("workplace OFFLINE — job will wait in Drive until Colab starts.")
        print_colab_help()
    if args.no_watch:
        print(f"watch later: python scripts/ctl.py watch {job['id']}")
        return 0
    return watch_job(job["id"])


def cmd_cancel(args: argparse.Namespace) -> int:
    job = get_job(args.job_id)
    if job["status"] != "queued":
        print(f"cannot cancel {args.job_id} (status={job['status']})")
        return 1
    update_job(args.job_id, status="cancelled", message="cancelled locally")
    print(f"cancelled {args.job_id}")
    return 0


def pick_job(job_id: str | None) -> str:
    if job_id:
        return job_id
    jobs = list_jobs()
    for job in jobs:
        if job["status"] == "running":
            return job["id"]
    for job in jobs:
        if job["status"] == "queued":
            return job["id"]
    if jobs:
        return jobs[0]["id"]
    raise FileNotFoundError("No jobs to watch. Run: python scripts/ctl.py train")


def watch_job(job_id: str) -> int:
    log_file = log_path(job_id)
    seen_log = 0
    last_signature = None
    last_help = 0.0
    print(f"watching {job_id}  (Ctrl+C to stop watching; the job stays queued)")
    print()

    while True:
        try:
            job = get_job(job_id)
        except FileNotFoundError:
            print(f"job {job_id} not found")
            return 1

        wp = read_heartbeat()
        progress = read_progress(job_id) or {}
        signature = (
            job.get("status"),
            job.get("updated_at"),
            progress.get("updated_at"),
            progress.get("step"),
            progress.get("loss"),
            progress.get("message"),
            wp.get("online"),
            wp.get("updated_at"),
        )
        if signature != last_signature:
            last_signature = signature
            print_snapshot(job, progress, wp)

        if log_file.exists():
            text = log_file.read_text(encoding="utf-8", errors="replace")
            if len(text) > seen_log:
                chunk = text[seen_log:]
                sys.stdout.write(chunk)
                if not chunk.endswith("\n"):
                    sys.stdout.write("\n")
                sys.stdout.flush()
                seen_log = len(text)

        if job["status"] in {"succeeded", "failed", "cancelled"}:
            print()
            print(f"done: {job['status']}")
            if job.get("error"):
                print("error:", job["error"])
            return 0 if job["status"] == "succeeded" else 1

        if not wp.get("online") and time.time() - last_help > 30:
            print_colab_help()
            last_help = time.time()

        time.sleep(3)


def print_snapshot(job: dict, progress: dict, wp: dict) -> None:
    online = "ONLINE" if wp.get("online") else "OFFLINE"
    step = progress.get("step")
    max_steps = progress.get("max_steps")
    step_s = f"{step}/{max_steps}" if step is not None and max_steps else str(step or "-")
    print(
        f"[{online}] {job['id']} {job['type']} {job['status']}  "
        f"step {step_s}  "
        f"loss {progress.get('loss', '-')}  "
        f"epoch {progress.get('epoch', '-')}  "
        f"{progress.get('percent', '-')}%  "
        f"{progress.get('message') or job.get('message') or ''}"
    )
    if progress.get("last_line"):
        print("  ", progress["last_line"])


def cmd_watch(args: argparse.Namespace) -> int:
    try:
        job_id = pick_job(args.job_id)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    return watch_job(job_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Queue Colab GPU jobs and watch progress in this terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show whether the Colab worker is online")
    sub.add_parser("list", help="List jobs")

    train = sub.add_parser("train", help="Queue a standard QLoRA train job")
    train.add_argument("--no-watch", action="store_true")

    ev = sub.add_parser("eval", help="Queue a standard QLoRA eval job")
    ev.add_argument("--no-watch", action="store_true")

    gated_train = sub.add_parser("gated-train", help="Queue a gated-LoRA train job")
    gated_train.add_argument("--no-watch", action="store_true")

    gated_eval = sub.add_parser("gated-eval", help="Queue a gated-LoRA eval job")
    gated_eval.add_argument("--no-watch", action="store_true")

    watch = sub.add_parser("watch", help="Watch the latest or a given job")
    watch.add_argument("job_id", nargs="?")

    cancel = sub.add_parser("cancel", help="Cancel a queued job")
    cancel.add_argument("job_id")

    sub.add_parser("cpu-test", help="Run local CPU smoke tests (no GPU, no Colab job)")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "status":
        return cmd_status(args)
    if args.command == "list":
        return cmd_list(args)
    if args.command in {"train", "eval", "gated-train", "gated-eval"}:
        return cmd_submit(args)
    if args.command == "watch":
        return cmd_watch(args)
    if args.command == "cancel":
        return cmd_cancel(args)
    if args.command == "cpu-test":
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "test_cpu",
            ROOT / "scripts" / "test_cpu.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.main()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
