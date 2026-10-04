"""One competition job in its OWN OS process (spawned: one CUDA context each, no GIL sharing).

The parent (scheduler.worker.Scheduler) decides WHEN a job runs and what happens when it is interrupted; this module only trains it:

  worker_main(...)   entry point of the child process. Talks to the parent through a Queue ("log" lines and ONE final "result") and a shared
                     integer abort flag (0 = keep going, otherwise an abort code the parent set).
  run_job(...)       the training itself. Terminal outcomes (finished / diverged / time / error) are written to the database HERE, exactly as the
                     single-threaded worker used to; an ABORT or an out-of-memory error is only REPORTED, because what to do next (put the job back at the
                     front in the right order, lower the concurrency, kill, redo) is decided by the parent, which sees all running jobs at once.

If the parent dies (kill -9, power loss of the process, not of the box) the child notices (its parent pid changes) and stops without saving: the next server
start re-queues the job from its last checkpoint like any other interrupted job, so two copies of a job never train at once.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Callable

from app.config import Settings
from app.store import Store

# abort codes shared with the parent; the numeric order is the priority (a later kill beats an earlier preemption)
ABORT_CODES = {"shutdown": 1, "preempt": 2, "redo": 3, "kill": 4}
ABORT_NAMES = {v: k for k, v in ABORT_CODES.items()}
INJECT_OOM_ENV = "BYOAI_INJECT_OOM"  # test hook: comma-separated job ids that raise ONE CUDA out-of-memory error at their first start


def is_oom(e: BaseException) -> bool:
    try:
        import torch

        if isinstance(e, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def _maybe_inject_oom(settings: Settings, job_id: int) -> None:
    ids = {x.strip() for x in os.environ.get(INJECT_OOM_ENV, "").split(",") if x.strip()}
    marker = settings.state_dir / f"oom_injected_{job_id}"
    if str(job_id) in ids and not marker.exists():
        marker.write_text("1")
        import torch

        raise torch.cuda.OutOfMemoryError(f"CUDA out of memory (injected for job {job_id})")


def run_job(settings: Settings, device: str, job_id: int, should_abort: Callable[[], bool], log: Callable[[int, str], None]) -> dict:
    """Train competition job `job_id` to a stop. Returns {"outcome": "done" | "aborted" | "oom", ...stats}. Raises on any other error."""
    import torch

    from app.chess_net.config import resolve_config
    from app.data.loaders import LichessData
    from app.training.chess_data import ChessData
    from app.training.trainer import CompetitionTrainer

    store = Store(settings.db_path)
    job = store.get_job(job_id)
    sub = store.get_submission(job["submission_id"])
    cfg, errors = resolve_config(json.loads(sub["config_json"]))
    if errors:
        raise ValueError("; ".join(errors))
    _maybe_inject_oom(settings, job_id)
    data = ChessData(LichessData(settings.data_dir / "lichess"), device, val_rows=settings.val_rows)
    trainer = CompetitionTrainer(cfg, data, device, budget_flops=settings.budget_flops, time_cap_s=settings.time_cap_seconds,
                                 ckpt_every_s=settings.checkpoint_seconds, metrics_every_s=settings.metrics_seconds)
    ck = store.latest_checkpoint(job_id)
    if ck and Path(ck["path"]).is_file():
        trainer.load_checkpoint(ck["path"])
        log(logging.INFO, f"job {job_id} resumed from checkpoint at step {trainer.step} ({trainer.samples_seen} samples)")
    else:
        log(logging.INFO, f"job {job_id} starting ({sub['nickname']}, {cfg['param_count']:,} params)")
    store.mark_running(job_id)
    job = store.get_job(job_id)

    def on_metrics(rec: dict) -> None:
        store.add_metrics(job_id, rec)
        store.update_progress(job_id, samples_seen=trainer.samples_seen, flops_used=trainer.flops_used, active_gpu_seconds=trainer.active_s)

    def on_checkpoint() -> None:
        path = settings.checkpoints_dir / str(job_id) / f"ckpt_{trainer.step}.pt"
        info = trainer.save_checkpoint(path)
        for p in store.add_checkpoint(job_id, info):
            Path(p).unlink(missing_ok=True)

    reason = trainer.run(should_abort, on_metrics, on_checkpoint)
    stats = {"samples_seen": trainer.samples_seen, "active_s": trainer.active_s, "steps": trainer.step, "flops_used": trainer.flops_used}
    if reason == "aborted":
        return {"outcome": "aborted", **stats}

    identity = {"nickname": sub["nickname"], "model_name": sub["model_name"], "participant_code": sub["participant_code"], "submission_id": sub["id"]}
    out_dir = settings.models_dir / sub["id"]
    if reason == "diverged":
        last = store.latest_checkpoint(job_id)
        if last and Path(last["path"]).is_file():  # ship the last finite weights, not NaNs
            trainer.load_checkpoint(last["path"])
            trainer.export(out_dir, "diverged", preemptions=job["preemptions"], identity=identity, started_at=job["started_at"])
        else:
            log(logging.WARNING, f"job {job_id} diverged before any checkpoint: no model saved")
    else:
        meta = trainer.export(out_dir, reason, preemptions=job["preemptions"], identity=identity, started_at=job["started_at"])
        log(logging.INFO, f"job {job_id} done: {reason}, val_mse {meta['val_mse']:.4f}")
    store.update_progress(job_id, samples_seen=trainer.samples_seen, flops_used=trainer.flops_used, active_gpu_seconds=trainer.active_s)
    store.finish_job(job_id, "done", reason)
    for p in store.drop_checkpoints(job_id):
        Path(p).unlink(missing_ok=True)
    shutil.rmtree(settings.checkpoints_dir / str(job_id), ignore_errors=True)
    del trainer, data
    return {"outcome": "done", "reason": reason, **stats}


def worker_main(settings: Settings, device: str, job_id: int, abort_flag, out_q, parent_pid: int) -> None:
    """Entry point of the spawned process."""

    def log(level: int, msg: str) -> None:
        out_q.put(("log", level, msg))

    def should_abort() -> bool:
        return abort_flag.value != 0 or os.getppid() != parent_pid  # the second clause: the server died, do not train on as an orphan

    try:
        if device == "cpu":  # several CPU processes would each grab every core and thrash: split the cores between the jobs that can run together
            import torch

            torch.set_num_threads(max(1, torch.get_num_threads() // max(1, settings.max_concurrent_jobs)))
        result = run_job(settings, device, job_id, should_abort, log)
    except BaseException as e:  # noqa: BLE001  nothing may escape without a result message
        if is_oom(e):
            try:
                import torch

                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            result = {"outcome": "oom", "message": str(e)[:300]}
        else:
            msg = f"{type(e).__name__}: {e}"
            log(logging.ERROR, f"job {job_id} failed: {traceback.format_exc()}")
            try:
                Store(settings.db_path).finish_job(job_id, "error", "error", msg)
            except Exception:  # noqa: BLE001
                pass
            result = {"outcome": "error", "message": msg}
    out_q.put(("result", result))
    out_q.close()
    out_q.join_thread()  # make sure the result is flushed before the process exits
