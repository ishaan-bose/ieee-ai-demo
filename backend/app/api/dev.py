"""Development endpoints. GET /api/dev/hello-stream: an SSE counter for testing the tunnel."""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.sse import SSE_HEADERS, sse_event

router = APIRouter()


@router.get("/api/dev/hello-stream")
async def hello_stream(
    request: Request,
    limit: int | None = Query(None, ge=1, le=100_000, description="stop after this many ticks (default: forever)"),
    interval: float = Query(1.0, ge=0.01, le=60, description="seconds between ticks"),
) -> StreamingResponse:
    """Streams `event: tick` with data {"count": n, "server_time": unix_seconds} once per interval."""

    async def gen():
        n = 0
        while limit is None or n < limit:
            if await request.is_disconnected():
                return
            yield sse_event("tick", {"count": n, "server_time": round(time.time(), 3)})
            n += 1
            if limit is not None and n >= limit:
                break
            await asyncio.sleep(interval)
        yield sse_event("done", {"count": n})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)
