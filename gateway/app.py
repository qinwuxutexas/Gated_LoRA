from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.jobs import (  # noqa: E402
    get_job,
    list_jobs,
    load_config,
    log_path,
    new_job,
    read_heartbeat,
    repo_root,
    update_job,
)
from src.progress import read_progress  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="gated_lora gateway", version="0.1.0")


class JobCreate(BaseModel):
    type: str = Field(pattern="^(train|eval)$")
    params: dict = Field(default_factory=dict)


@app.get("/api/health")
def health():
    return {"ok": True, "role": "gateway", "root": str(repo_root())}


@app.get("/api/workspace")
def workspace():
    cfg = load_config()
    return {
        "config": cfg,
        "workplace": read_heartbeat(),
        "root": str(repo_root()),
    }


def with_progress(job: dict) -> dict:
    progress = read_progress(job["id"])
    if progress:
        job = dict(job)
        job["progress"] = progress
    return job


@app.get("/api/jobs")
def jobs():
    return {"jobs": [with_progress(job) for job in list_jobs()]}


@app.get("/api/jobs/{job_id}")
def job_detail(job_id: str):
    try:
        return with_progress(get_job(job_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc


@app.get("/api/jobs/{job_id}/progress")
def job_progress(job_id: str):
    return read_progress(job_id) or {}


@app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
def job_log(job_id: str):
    path = log_path(job_id)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > 200:
        text = "\n".join(lines[-200:])
    return text


@app.post("/api/jobs")
def create_job(body: JobCreate):
    try:
        return new_job(body.type, body.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        job = get_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    if job["status"] != "queued":
        raise HTTPException(
            status_code=409,
            detail=f"Only queued jobs can be cancelled (status={job['status']})",
        )
    return update_job(job_id, status="cancelled")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
