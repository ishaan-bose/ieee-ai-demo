"""Server-Sent Events helpers (plain text/event-stream, no extra dependency)."""

from __future__ import annotations

import json

# no-cache + no proxy buffering so events reach the browser immediately through the tunnel
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"
