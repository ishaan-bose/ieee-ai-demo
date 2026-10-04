"""Pydantic models for every request and response (SPEC 7). `python -m app.export_openapi` turns these into shared/openapi.json."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorOut(BaseModel):
    error: str
    message: str


# ---- demo races
class LaneIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1, max_length=40)
    config: dict[str, Any] = {}


class RaceIn(BaseModel):
    kind: Literal["loss", "lr", "batch"]
    lanes: list[LaneIn] = Field(min_length=1, max_length=3)
    max_seconds: float = Field(30.0, gt=0)


class RaceOut(BaseModel):
    race_id: str


class OkOut(BaseModel):
    ok: bool


class LaneTick(BaseModel):
    id: str
    step: int
    samples_seen: int
    updates: int
    loss: float | None
    acc: float
    probe_preds: list[int]


class TickEvent(BaseModel):  # documents the SSE `tick` event payload
    t: float
    lanes: list[LaneTick]


class DoneEvent(BaseModel):  # documents the SSE `done` event payload
    reason: Literal["time", "complete", "aborted"]
    final: list[LaneTick]


class ErrorEvent(BaseModel):  # documents the SSE `error` event payload
    message: str


# ---- submissions
class PreviewOut(BaseModel):
    valid: bool
    errors: list[str]
    param_count: int | None = None
    tier: str | None = None
    search_depth_full_moves: int | None = None
    est_flops_budget: float | None = None
    flops_per_sample: int | None = None
    est_minutes: float | None = None


class SubmissionIn(BaseModel):
    model_config = ConfigDict(extra="ignore")
    nickname: str = Field(min_length=1, max_length=24)
    model_name: str = Field(min_length=1, max_length=32)
    contact: str | None = Field(None, max_length=120)
    consent: bool
    config: dict[str, Any] = {}
    client_id: str | None = Field(None, max_length=64, description="random id made by the browser; makes retried uploads idempotent")


class SubmissionOut(BaseModel):
    submission_id: str
    participant_code: str
    param_count: int
    tier: str
    status: Literal["queued"] = "queued"
    queue_position: int | None


class ProgressOut(BaseModel):
    fraction: float
    samples_seen: int
    flops_used: float
    flops_budget: float
    active_gpu_seconds: float
    preemptions: int


class SubmissionStatusOut(BaseModel):
    status: Literal["queued", "running", "done", "killed", "removed", "error"]
    participant_code: str
    nickname: str
    model_name: str
    tier: str
    param_count: int
    queue_position: int | None = None
    progress: ProgressOut | None = None
    stop_reason: Literal["budget", "time", "diverged", "killed", "error"] | None = None
    curves: dict[str, list[dict[str, Any]]] | None = None
    val_mse: float | None = None


# ---- admin
class DemoModeIn(BaseModel):
    enabled: bool


class ResetIn(BaseModel):
    confirm: str


class ResetOut(BaseModel):
    ok: bool
    snapshot: str


class JobOut(BaseModel):
    id: int
    kind: str
    status: str
    nickname: str | None = None
    model_name: str | None = None
    participant_code: str | None = None
    tier: str | None = None
    param_count: int | None = None
    config_summary: str | None = None
    priority: int
    stop_reason: str | None = None
    progress: float | None = None
    samples_seen: int
    flops_used: float
    active_gpu_seconds: float
    preemptions: int
    error_message: str | None = None
    concurrency: int | None = None  # most competition jobs that were training at the same time while this one ran
    samples_per_s: float | None = None
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None


class QueueOut(BaseModel):
    paused: bool
    demo_mode: bool
    current_job_id: int | None
    running_job_ids: list[int] = []  # every competition job training right now (up to MAX_CONCURRENT_JOBS)
    budget_flops: float
    jobs: list[JobOut]


class JobActionOut(BaseModel):
    ok: bool
    status: str


class PresenterState(BaseModel):
    stage_id: str = ""
    stage_index: int = 0
    title: str = ""
    updated_at: float = 0.0
