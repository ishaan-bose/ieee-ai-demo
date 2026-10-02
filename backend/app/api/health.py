"""GET /api/health (SPEC 7). Polled every few seconds by the frontend (SPEC 5.1)."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app import __version__, db

router = APIRouter()


class HealthResponse(BaseModel):
    ok: bool
    device: str  # "cuda" or "cpu"
    gpu_name: str | None
    vram_gb: float | None
    queue_length: int  # jobs queued or running
    demo_mode: bool
    paused: bool
    version: str


@router.get("/api/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    state = request.app.state
    conn = db.connect(state.settings.db_path)
    try:
        queue_length = db.queue_length(conn)
        demo_mode = db.is_demo_mode(conn)
        paused = db.is_paused(conn)
    finally:
        conn.close()
    dev = state.device
    return HealthResponse(
        ok=True,
        device=dev.device,
        gpu_name=dev.gpu_name,
        vram_gb=dev.vram_gb,
        queue_length=queue_length,
        demo_mode=demo_mode,
        paused=paused,
        version=__version__,
    )
