"""Demo races (SPEC 7): POST /api/demo/races, SSE stream, abort."""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.errors import api_error
from app.schemas import OkOut, RaceIn, RaceOut
from app.sse import SSE_HEADERS, sse_event
from app.training.race import resolve_lanes

router = APIRouter()


@router.post("/api/demo/races", response_model=RaceOut)
def create_race(body: RaceIn, request: Request) -> RaceOut:
    st = request.app.state
    if body.max_seconds > st.settings.race_max_seconds:
        raise api_error(422, "validation_error", f"max_seconds must be <= {st.settings.race_max_seconds:g}")
    lanes_raw = [{"id": l.id, "config": l.config} for l in body.lanes]
    lanes, errors = resolve_lanes(lanes_raw)
    if errors:
        raise api_error(422, "validation_error", "; ".join(errors))
    if not st.scheduler.doodle_data_available():
        # The frontend treats this like "server offline" for races and plays the cached stream instead.
        raise api_error(503, "data_unavailable", "doodle tensors are not built yet (run scripts/rasterize_quickdraw.py)")
    race_id = uuid.uuid4().hex[:12]
    st.scheduler.enqueue_race(race_id, body.kind, [l.model_dump() for l in lanes], body.max_seconds)
    return RaceOut(race_id=race_id)


@router.get("/api/demo/races/{race_id}/stream")
async def race_stream(race_id: str, request: Request) -> StreamingResponse:
    state = request.app.state.scheduler.races.get(race_id)
    if state is None:
        raise api_error(404, "not_found", "unknown race (the server may have restarted)")

    async def gen():
        idx, idle = 0, 0
        while True:
            if await request.is_disconnected():
                return
            new = state.since(idx)
            for event, data in new:
                yield sse_event(event, data)
                idx += 1
                idle = 0
                if event in ("done", "error"):
                    return
            if not new:
                idle += 1
                if idle % 10 == 0:
                    yield ": waiting\n\n"  # keep-alive comment while the race waits for the GPU
                await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/api/demo/races/{race_id}/abort", response_model=OkOut)
def abort_race(race_id: str, request: Request) -> OkOut:
    if not request.app.state.scheduler.abort_race(race_id):
        raise api_error(404, "not_found", "unknown race")
    return OkOut(ok=True)
