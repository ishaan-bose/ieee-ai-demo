"""Submissions (SPEC 7): preview, submit, status, config schema, leaderboard placeholder."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Body, Request

from app.api.errors import api_error
from app.chess_net.config import CompetitionConfig, DEFAULT_CONFIG, MAX_PARAMS, flops_per_sample, resolve_config, tier_table
from app.activations import ACTIVATION_NAMES
from app.schemas import PreviewOut, ProgressOut, SubmissionIn, SubmissionOut, SubmissionStatusOut

router = APIRouter()


def _preview(cfg_raw: dict, budget: float, nominal_flops_per_s: float | None = None) -> PreviewOut:
    cfg, errors = resolve_config(cfg_raw)
    if errors:
        return PreviewOut(valid=False, errors=errors)
    fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
    return PreviewOut(valid=True, errors=[], param_count=cfg["param_count"], tier=cfg["tier"],
                      search_depth_full_moves=cfg["search_depth_full_moves"], est_flops_budget=budget, flops_per_sample=fps,
                      est_minutes=None)


@router.post("/api/submissions/preview", response_model=PreviewOut)
def preview(request: Request, config: dict[str, Any] = Body(default_factory=dict)) -> PreviewOut:
    return _preview(config, request.app.state.settings.budget_flops)


@router.post("/api/submissions", response_model=SubmissionOut)
def submit(body: SubmissionIn, request: Request) -> SubmissionOut:
    st = request.app.state
    if not body.consent:
        raise api_error(422, "consent_required", "please tick the consent box to enter")
    nickname, model_name = body.nickname.strip(), body.model_name.strip()
    if not nickname or not model_name:
        raise api_error(422, "validation_error", "nickname and model name cannot be blank")
    cfg, errors = resolve_config(body.config)
    if errors:
        raise api_error(422, "invalid_config", "; ".join(errors))
    contact = (body.contact or "").strip() or None
    res = st.store.create_submission(nickname=nickname, model_name=model_name, contact=contact, consent=body.consent, config=cfg,
                                     client_id=body.client_id)
    st.scheduler.wake()
    return SubmissionOut(submission_id=res["submission_id"], participant_code=res["participant_code"], param_count=res["param_count"],
                         tier=res["tier"], queue_position=st.store.queue_position(res["job_id"]) if res["job_id"] else None)


@router.get("/api/submissions/{submission_id}", response_model=SubmissionStatusOut)
def submission_status(submission_id: str, request: Request) -> SubmissionStatusOut:
    st = request.app.state
    sub = st.store.get_submission(submission_id)
    job = st.store.submission_job(submission_id) if sub else None
    if sub is None or job is None:
        raise api_error(404, "not_found", "unknown submission")
    budget = st.settings.budget_flops
    progress = ProgressOut(fraction=min(1.0, job["flops_used"] / budget) if budget else 0.0, samples_seen=job["samples_seen"],
                           flops_used=job["flops_used"], flops_budget=budget, active_gpu_seconds=job["active_gpu_seconds"],
                           preemptions=job["preemptions"])
    val_mse = None
    if job["status"] == "done":
        try:
            val_mse = json.loads((st.settings.models_dir / submission_id / "meta.json").read_text()).get("val_mse")
        except (OSError, ValueError):
            pass
    return SubmissionStatusOut(
        status=job["status"], participant_code=sub["participant_code"], nickname=sub["nickname"], model_name=sub["model_name"],
        tier=sub["tier"], param_count=sub["param_count"], queue_position=st.store.queue_position(job["id"]),
        progress=progress if job["status"] != "queued" or job["flops_used"] > 0 else None,
        stop_reason=job["stop_reason"], curves=st.store.curves(job["id"]) if job["status"] in ("running", "done", "killed") else None,
        val_mse=val_mse)


@router.get("/api/config/schema")
def config_schema(request: Request) -> dict:
    """Extension of SPEC 7 for the /build form: defaults, JSON schema of every knob, the tier table, the parameter cap."""
    return {"defaults": DEFAULT_CONFIG, "json_schema": CompetitionConfig.model_json_schema(), "tiers": tier_table(),
            "max_params": MAX_PARAMS, "activations": list(ACTIVATION_NAMES), "budget_flops": request.app.state.settings.budget_flops}


@router.get("/api/leaderboard")
def leaderboard() -> list:
    """Reserved (SPEC 7). No ranking logic on the server."""
    return []
