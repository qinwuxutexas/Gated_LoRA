"""Drive-backed job queue shared by the local gateway and the Colab worker."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()

COLAB_ROOT = Path("/content/drive/MyDrive/gated_lora")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    if COLAB_ROOT.exists():
        return COLAB_ROOT
    return Path(__file__).resolve().parents[1]


def load_config(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    path = root / "workspace" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def jobs_dir(root: Path | None = None) -> Path:
    root = root or repo_root()
    cfg = load_config(root)
    path = root / cfg["paths"]["jobs_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir(root: Path | None = None) -> Path:
    root = root or repo_root()
    cfg = load_config(root)
    path = root / cfg["paths"]["logs_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def status_path(root: Path | None = None) -> Path:
    root = root or repo_root()
    cfg = load_config(root)
    path = root / cfg["paths"]["status_file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def job_path(job_id: str, root: Path | None = None) -> Path:
    return jobs_dir(root) / f"{job_id}.json"


def log_path(job_id: str, root: Path | None = None) -> Path:
    return logs_dir(root) / f"{job_id}.log"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON in a Drive-FUSE-safe way. os.replace() is unreliable on Drive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    last_err: Exception | None = None
    with _WRITE_LOCK:
        for attempt in range(8):
            try:
                with path.open("w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                return
            except OSError as exc:
                last_err = exc
                time.sleep(0.4 * (attempt + 1))
    raise last_err


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def new_job(job_type: str, params: dict[str, Any] | None = None, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    cfg = load_config(root)
    if job_type not in ("train", "eval", "gated_train", "gated_eval"):
        raise ValueError(f"Unsupported job type: {job_type}")

    merged = dict(cfg["defaults"][job_type])
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})

    now = utc_now()
    job = {
        "id": uuid.uuid4().hex[:12],
        "type": job_type,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        "params": merged,
        "error": None,
        "gpu": None,
        "workplace": cfg["workplace"]["name"],
        "message": "queued — waiting for the Colab A100 worker",
    }
    write_json(job_path(job["id"], root), job)
    return job


def get_job(job_id: str, root: Path | None = None) -> dict[str, Any]:
    path = job_path(job_id, root)
    if not path.exists():
        raise FileNotFoundError(job_id)
    return read_json(path)


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    root = repo_root()
    job = get_job(job_id, root)
    job.update(fields)
    job["updated_at"] = utc_now()
    write_json(job_path(job_id, root), job)
    return job


def list_jobs(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repo_root()
    jobs = []
    for path in jobs_dir(root).glob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        try:
            jobs.append(read_json(path))
        except json.JSONDecodeError:
            continue
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


def next_queued_job(root: Path | None = None) -> dict[str, Any] | None:
    queued = [j for j in list_jobs(root) if j.get("status") == "queued"]
    queued.sort(key=lambda j: j.get("created_at", ""))
    return queued[0] if queued else None


def write_heartbeat(extra: dict[str, Any] | None = None, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    cfg = load_config(root)
    payload = {
        "online": True,
        "workplace": cfg["workplace"]["name"],
        "kind": cfg["workplace"]["kind"],
        "requested_gpu": cfg["workplace"]["gpu"],
        "updated_at": utc_now(),
    }
    if extra:
        payload.update(extra)
    write_json(status_path(root), payload)
    return payload


def read_heartbeat(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    path = status_path(root)
    if not path.exists():
        return {
            "online": False,
            "workplace": load_config(root)["workplace"]["name"],
            "reason": "Workplace has not sent a heartbeat yet. Start the Colab worker.",
        }

    payload = read_json(path)
    stale_after = load_config(root)["workplace"]["heartbeat_stale_seconds"]
    updated = payload.get("updated_at")
    online = False
    if updated:
        try:
            then = datetime.fromisoformat(updated)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - then).total_seconds()
            online = age <= stale_after
            payload["age_seconds"] = round(age, 1)
        except ValueError:
            online = False
    payload["online"] = online
    if not online:
        payload["reason"] = "Colab heartbeat is stale. Re-run the workplace notebook on an A100."
    return payload
