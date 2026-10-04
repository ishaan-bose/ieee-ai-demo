"""Presenter sync (extension of SPEC 7): laptop B publishes its current stage, the phone's /presenter page reads it.
Ephemeral on purpose (in memory): after a server restart the next stage change republishes it."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.schemas import PresenterState

router = APIRouter()


@router.get("/api/presenter", response_model=PresenterState)
def get_presenter(request: Request) -> PresenterState:
    return getattr(request.app.state, "presenter", PresenterState())


@router.put("/api/presenter", response_model=PresenterState)
def put_presenter(body: PresenterState, request: Request) -> PresenterState:
    body.updated_at = time.time()
    request.app.state.presenter = body
    return body
