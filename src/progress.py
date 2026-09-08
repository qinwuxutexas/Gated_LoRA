"""Live progress files so the local gateway can show Colab job status."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.jobs import repo_root, utc_now, write_json, read_json


def current_job_id() -> str | None:
    return os.environ.get("GATED_LORA_JOB_ID") or None


def progress_dir(root: Path | None = None) -> Path:
    path = (root or repo_root()) / "workspace" / "progress"
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress_path(job_id: str, root: Path | None = None) -> Path:
    return progress_dir(root) / f"{job_id}.json"


def write_progress(job_id: str, **fields: Any) -> dict[str, Any]:
    root = repo_root()
    path = progress_path(job_id, root)
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            payload = read_json(path)
        except Exception:
            payload = {}
    payload.update({k: v for k, v in fields.items() if v is not None})
    payload["job_id"] = job_id
    payload["updated_at"] = utc_now()
    write_json(path, payload)
    return payload


def report(**fields: Any) -> dict[str, Any] | None:
    job_id = current_job_id()
    if not job_id:
        return None
    return write_progress(job_id, **fields)


def read_progress(job_id: str, root: Path | None = None) -> dict[str, Any] | None:
    path = progress_path(job_id, root)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None
