"""Admin routes (SPEC 7). Every route needs the X-Admin-Token header (token from the ADMIN_TOKEN env var)."""

from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import shutil
import time

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from app import logbuf
from app.api.errors import api_error
from app.schemas import DemoModeIn, JobActionOut, JobOut, OkOut, QueueOut, ResetIn, ResetOut
from app.sse import SSE_HEADERS, sse_event

KEEP_ON_RESET = ("house-net",)  # the House Net is not a participant run; a nuclear reset must not delete it


def require_admin(request: Request, x_admin_token: str | None = Header(default=None)) -> None:
    token = request.app.state.settings.admin_token
    if not token:
        raise api_error(503, "admin_disabled", "ADMIN_TOKEN is not set on the server")
    if not x_admin_token or not hmac.compare_digest(x_admin_token.encode(), token.encode()):
        raise api_error(401, "unauthorized", "missing or wrong X-Admin-Token")


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])


def _summary(cfg_json: str | None) -> str | None:
    if not cfg_json:
        return None
    c = json.loads(cfg_json)
    return (f"{c['layers']}x{'/'.join(map(str, sorted(set(c['widths']))))} {c['activation']['name']} {c['loss']['name']} "
            f"{c['optimizer']} lr={c['lr']:g} bs={c['batch_size']}")


def _job_out(j: dict, budget: float) -> JobOut:
    prog = min(1.0, j["flops_used"] / budget) if j["kind"] == "competition" and budget else None
    return JobOut(id=j["id"], kind=j["kind"], status=j["status"], nickname=j.get("nickname") or ("demo race" if j["kind"] == "demo_race" else None),
                  model_name=j.get("model_name"), participant_code=j.get("participant_code"), tier=j.get("tier"),
                  param_count=j.get("param_count"), config_summary=_summary(j.get("config_json")), priority=j["priority"],
                  stop_reason=j["stop_reason"], progress=prog, samples_seen=j["samples_seen"], flops_used=j["flops_used"],
                  active_gpu_seconds=j["active_gpu_seconds"], preemptions=j["preemptions"], error_message=j["error_message"],
                  created_at=j["created_at"], started_at=j["started_at"], finished_at=j["finished_at"])


@router.get("/queue", response_model=QueueOut)
def queue(request: Request, include_removed: bool = False) -> QueueOut:
    st = request.app.state
    cur = st.scheduler.current
    return QueueOut(paused=st.store.is_paused(), demo_mode=st.store.is_demo_mode(), current_job_id=cur.job_id if cur else None,
                    budget_flops=st.settings.budget_flops,
                    jobs=[_job_out(j, st.settings.budget_flops) for j in st.store.list_jobs(include_removed)])


def _job_or_404(request: Request, job_id: int) -> dict:
    job = request.app.state.store.get_job(job_id)
    if job is None:
        raise api_error(404, "not_found", f"no job {job_id}")
    return job


@router.post("/jobs/{job_id}/kill", response_model=JobActionOut)
def kill(job_id: int, request: Request) -> JobActionOut:
    st = request.app.state
    job = _job_or_404(request, job_id)
    if job["status"] not in ("queued", "running"):
        raise api_error(409, "conflict", f"job {job_id} is {job['status']}, nothing to kill")
    status = st.scheduler.kill_job(job_id)
    st.store.log_event("job_kill", {"job_id": job_id})
    return JobActionOut(ok=True, status=status)


@router.delete("/jobs/{job_id}", response_model=JobActionOut)
def remove(job_id: int, request: Request) -> JobActionOut:
    st = request.app.state
    job = _job_or_404(request, job_id)
    if job["status"] == "running":
        raise api_error(409, "conflict", "job is running: kill it first")
    st.scheduler.remove_job(job_id)
    st.store.log_event("job_remove", {"job_id": job_id})
    return JobActionOut(ok=True, status="removed")


@router.post("/jobs/{job_id}/redo", response_model=JobActionOut)
def redo(job_id: int, request: Request) -> JobActionOut:
    st = request.app.state
    job = _job_or_404(request, job_id)
    if job["kind"] != "competition":
        raise api_error(400, "bad_request", "only competition jobs can be redone")
    status = st.scheduler.redo_job(job_id)
    st.store.log_event("job_redo", {"job_id": job_id})
    return JobActionOut(ok=True, status=status)


@router.post("/queue/pause", response_model=OkOut)
def pause(request: Request) -> OkOut:
    st = request.app.state
    st.store.log_event("pause")
    st.scheduler.apply_queue_state()
    return OkOut(ok=True)


@router.post("/queue/resume", response_model=OkOut)
def resume(request: Request) -> OkOut:
    st = request.app.state
    st.store.log_event("resume")
    st.scheduler.apply_queue_state()
    return OkOut(ok=True)


@router.post("/demo-mode", response_model=OkOut)
def demo_mode(body: DemoModeIn, request: Request) -> OkOut:
    st = request.app.state
    st.store.log_event("demo_mode", {"enabled": body.enabled})
    st.scheduler.apply_queue_state()
    return OkOut(ok=True)


@router.post("/reset", response_model=ResetOut)
def reset(body: ResetIn, request: Request) -> ResetOut:
    """Nuclear reset: snapshot the database FIRST, then clear queue, results and stored runs."""
    st = request.app.state
    if body.confirm != "RESET":
        raise api_error(400, "bad_request", "type RESET to confirm")
    snap = st.store.snapshot(st.settings.state_dir / "snapshots" / f"demo-{time.strftime('%Y%m%d-%H%M%S')}.db")
    cur = st.scheduler.current
    if cur:
        cur.abort("kill")
        st.scheduler.wait_idle(20.0)
    st.store.clear_runs()
    for d in (st.settings.models_dir, st.settings.checkpoints_dir):
        if d.is_dir():
            for child in d.iterdir():
                if child.name not in KEEP_ON_RESET:
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    st.store.log_event("reset", {"snapshot": str(snap)})
    return ResetOut(ok=True, snapshot=str(snap))


@router.get("/export")
def export(request: Request, format: str = Query("json", pattern="^(json|csv)$"), include_contact: bool = False):
    st = request.app.state
    rows = st.store.export_rows(include_contact=include_contact)
    if format == "json":
        return JSONResponse({"generated_at": time.time(), "include_contact": include_contact, "submissions": rows},
                            headers={"Content-Disposition": 'attachment; filename="export.json"'})
    flat = []
    for r in rows:
        cfg = r.pop("config")
        flat.append({**r, "layers": cfg["layers"], "widths": "/".join(map(str, cfg["widths"])), "activation": cfg["activation"]["name"],
                     "loss": cfg["loss"]["name"], "optimizer": cfg["optimizer"], "lr": cfg["lr"], "batch_size": cfg["batch_size"],
                     "config_json": json.dumps(cfg)})
    buf = io.StringIO()
    if flat:
        w = csv.DictWriter(buf, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    return PlainTextResponse(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="export.csv"'})


@router.get("/logs")
def logs(tail: int = Query(200, ge=1, le=2000)) -> dict:
    return {"lines": logbuf.tail(tail)}


def gpu_stats(device: str) -> dict:
    out: dict = {"device": device, "utilization": None, "memory_used_gb": None, "memory_total_gb": None}
    try:
        import torch

        if device == "cuda":
            free, total = torch.cuda.mem_get_info()
            out["memory_used_gb"], out["memory_total_gb"] = round((total - free) / 1024**3, 2), round(total / 1024**3, 2)
            try:
                out["utilization"] = torch.cuda.utilization()  # needs pynvml; None if unavailable
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return out


@router.get("/stream")
async def stream(request: Request, limit: int | None = Query(None, ge=1, description="stop after N events (tests)")) -> StreamingResponse:
    """Live monitor (SSE): one `status` event per second with the current job's curves, GPU stats and a log tail."""
    st = request.app.state

    async def gen():
        sent = 0
        while limit is None or sent < limit:
            if await request.is_disconnected():
                return
            sent += 1
            cur = st.scheduler.current
            job = st.store.get_job(cur.job_id) if cur else None
            payload = {"t": time.time(), "paused": st.store.is_paused(), "demo_mode": st.store.is_demo_mode(),
                       "current_job_id": cur.job_id if cur else None, "kind": cur.kind if cur else None,
                       "progress": min(1.0, job["flops_used"] / st.settings.budget_flops) if job and job["kind"] == "competition" else None,
                       "curves": st.store.curves(cur.job_id, 200) if cur and cur.kind == "competition" else None,
                       "gpu": gpu_stats(st.device.device), "log_tail": logbuf.tail(15)}
            yield sse_event("status", payload)
            await asyncio.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
