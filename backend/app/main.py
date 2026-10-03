"""FastAPI app. Start with:  python -m app.main   (from backend/)

Binds 127.0.0.1:8000 only; the laptop reaches it through the SSH tunnel.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__, db
from app.api import dev, health
from app.config import get_settings
from app.device import detect_device
from app.selfcheck import run_self_check

log = logging.getLogger("app")

GRACEFUL_SHUTDOWN_SECONDS = 3


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    db.init_db(settings.db_path)
    app.state.settings = settings
    app.state.device = detect_device()
    app.state.self_check = (
        run_self_check(settings, app.state.device, print_fn=lambda s: print(s, flush=True))
        if settings.run_self_check else None
    )
    # Phase 6: re-queue interrupted jobs from their checkpoints here (SPEC 6.3).
    yield


app = FastAPI(title="Build Your Own AI backend", version=__version__, lifespan=lifespan)


# Errors use the SPEC 7 shape: {"error": "code", "message": "..."}.
_CODES = {400: "bad_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
          405: "method_not_allowed", 409: "conflict", 422: "validation_error", 500: "internal_error"}


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        body = {"error": detail["error"], "message": str(detail.get("message", ""))}
    else:
        body = {"error": _CODES.get(exc.status_code, "error"), "message": str(detail)}
    return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    msgs = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors())
    return JSONResponse({"error": "validation_error", "message": msgs}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error")
    return JSONResponse({"error": "internal_error", "message": f"{type(exc).__name__}: {exc}"}, status_code=500)


app.include_router(health.router)
app.include_router(dev.router)


def main() -> None:
    import uvicorn

    settings = get_settings()
    # Open SSE streams never end on their own, and uvicorn's graceful shutdown would
    # wait for them forever (Ctrl+C / restart would hang while a browser is connected).
    # Cap it: after this many seconds, remaining connections are closed.
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level="info",
                timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS)


if __name__ == "__main__":
    main()
