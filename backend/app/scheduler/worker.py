"""The scheduler (SPEC 6.1): ONE worker thread, FCFS queue, demo races preempt competition training by killing it.

- A demo race sets the abort flag of a running competition job. The trainer stops within a step or two WITHOUT saving; the job
  returns to the FRONT of the queue and later resumes from its last periodic checkpoint (no samples are double counted).
- Demo Mode (and pause) keep competition jobs off the GPU entirely.
- Everything durable is in SQLite; on start, interrupted jobs are re-queued (a crash and a clean shutdown are treated the same).
- When the queue is empty the worker idles and does nothing.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
from pathlib import Path

import torch

from app.chess_net.config import resolve_config
from app.config import Settings
from app.data.loaders import LichessData, QuickDrawData
from app.device import DeviceInfo
from app.store import Store
from app.training.chess_data import ChessData
from app.training.race import DoodleData, Lane, resolve_lanes, run_race
from app.training.trainer import CompetitionTrainer

log = logging.getLogger("scheduler")


class RaceState:
    """In-memory event buffer of one race (SSE consumers replay from the start, so late subscribers miss nothing)."""

    def __init__(self, race_id: str):
        self.race_id = race_id
        self.events: list[tuple[str, dict]] = []
        self.finished = False
        self.abort = threading.Event()
        self._lock = threading.Lock()

    def emit(self, event: str, data: dict) -> None:
        with self._lock:
            self.events.append((event, data))
            if event in ("done", "error"):
                self.finished = True

    def since(self, idx: int) -> list[tuple[str, dict]]:
        with self._lock:
            return self.events[idx:]


ABORT_PRIORITY = {"shutdown": 0, "preempt": 1, "redo": 2, "kill": 3}  # an admin's kill/redo beats an automatic preemption


class Running:
    def __init__(self, job: dict):
        self.job, self.job_id, self.kind = job, job["id"], job["kind"]
        self.reason: str | None = None  # abort reason: preempt | kill | redo | shutdown

    def abort(self, reason: str) -> None:
        if self.reason is None or ABORT_PRIORITY[reason] > ABORT_PRIORITY[self.reason]:
            self.reason = reason

    def should_abort(self) -> bool:
        return self.reason is not None


class Scheduler:
    def __init__(self, settings: Settings, device: DeviceInfo, store: Store):
        self.s, self.store, self.device = settings, store, torch.device(device.device)
        self.current: Running | None = None
        self.races: dict[str, RaceState] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._chess: ChessData | None = None
        self._doodles: DoodleData | None = None
        self._guard = threading.Lock()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> dict:
        rec = self.store.recover_after_restart()
        if rec["requeued"] or rec["races_failed"]:
            log.info("recovered after restart: re-queued jobs %s, failed races %s", rec["requeued"], rec["races_failed"])
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._thread.start()
        return rec

    def stop(self, timeout: float = 20.0) -> None:
        self._stop.set()
        cur = self.current
        if cur:
            cur.abort("shutdown")
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------ data (loaded lazily, once)
    def chess_data(self) -> ChessData:
        if self._chess is None:
            self._chess = ChessData(LichessData(self.s.data_dir / "lichess"), self.device, val_rows=self.s.val_rows)
        return self._chess

    def doodle_data(self) -> DoodleData:
        if self._doodles is None:
            self._doodles = DoodleData.from_dir(QuickDrawData(self.s.data_dir / "quickdraw"), self.device)
        return self._doodles

    def doodle_data_available(self) -> bool:
        return QuickDrawData(self.s.data_dir / "quickdraw").has_tensors(("train", "val")) if (self.s.data_dir / "quickdraw").is_dir() else False

    # ------------------------------------------------------------ control (called from API threads)
    def enqueue_race(self, race_id: str, kind: str, lanes: list[dict], max_seconds: float) -> None:
        self.races[race_id] = RaceState(race_id)
        for old in list(self.races)[:-30]:  # bounded memory
            if self.races[old].finished:
                del self.races[old]
        self.store.create_race(race_id, kind, lanes, max_seconds)
        self.preempt_competition()
        self.wake()

    def preempt_competition(self) -> bool:
        cur = self.current
        if cur and cur.kind == "competition":
            log.info("preempting competition job %s", cur.job_id)
            cur.abort("preempt")
            return True
        return False

    def abort_race(self, race_id: str) -> bool:
        st = self.races.get(race_id)
        job = self.store.race_job(race_id)
        if st is None or job is None:
            return False
        st.abort.set()
        if job["status"] == "queued":  # never started: finish it right away
            self.store.finish_job(job["id"], "killed", "killed")
            self.store.update_race(race_id, status="done", done_reason="aborted", finished_at=time.time())
            st.emit("done", {"reason": "aborted", "final": []})
        return True

    def apply_queue_state(self) -> None:
        """Called after pause / demo-mode changes: competition training must leave the GPU."""
        if self.store.is_paused() or self.store.is_demo_mode():
            self.preempt_competition()
        self.wake()

    def kill_job(self, job_id: int) -> str:
        cur = self.current
        if cur and cur.job_id == job_id:
            cur.abort("kill")
            return "killing"
        job = self.store.get_job(job_id)
        if job and job["status"] == "queued":
            self.store.finish_job(job_id, "killed", "killed")
            self._cleanup_job_files(job)
            return "killed"
        return "not_running"

    def redo_job(self, job_id: int) -> str:
        cur = self.current
        if cur and cur.job_id == job_id:
            cur.abort("redo")
            return "redoing"
        job = self.store.get_job(job_id)
        paths = self.store.reset_job_from_scratch(job_id)
        self._delete_files(paths)
        self._cleanup_job_files(job, keep_checkpoints=True)
        self.wake()
        return "queued"

    def remove_job(self, job_id: int) -> None:
        job = self.store.get_job(job_id)
        self.store.set_removed(job_id)
        self._cleanup_job_files(job)

    def wait_idle(self, timeout: float = 15.0) -> bool:
        t0 = time.time()
        while self.current is not None and time.time() - t0 < timeout:
            time.sleep(0.05)
        return self.current is None

    # ------------------------------------------------------------ files
    @staticmethod
    def _delete_files(paths) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass

    def _cleanup_job_files(self, job: dict | None, keep_checkpoints: bool = False) -> None:
        if not job:
            return
        if not keep_checkpoints:
            self._delete_files(self.store.drop_checkpoints(job["id"]))
        shutil.rmtree(self.s.checkpoints_dir / str(job["id"]), ignore_errors=True) if not keep_checkpoints else None
        if job.get("submission_id"):
            shutil.rmtree(self.s.models_dir / job["submission_id"], ignore_errors=True)

    # ------------------------------------------------------------ the worker
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                allow = not (self.store.is_paused() or self.store.is_demo_mode())
                job = self.store.next_job(allow)
            except Exception:  # noqa: BLE001  the worker must never die
                log.error("scheduler DB error: %s", traceback.format_exc())
                time.sleep(1.0)
                continue
            if job is None:
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            run = Running(job)
            with self._guard:
                self.current = run
            try:
                if job["kind"] == "demo_race":
                    self._run_race(run)
                else:
                    self._run_competition(run)
            except Exception as e:  # noqa: BLE001
                log.error("job %s failed: %s", job["id"], traceback.format_exc())
                self.store.finish_job(job["id"], "error", "error", f"{type(e).__name__}: {e}")
                if job["kind"] == "demo_race":
                    self._fail_race(job["race_id"], f"{type(e).__name__}: {e}")
            finally:
                with self._guard:
                    self.current = None

    # ------------------------------------------------------------ demo race
    def _fail_race(self, race_id: str, message: str) -> None:
        self.store.update_race(race_id, status="error", finished_at=time.time())
        st = self.races.get(race_id)
        if st and not st.finished:
            st.emit("error", {"message": message})

    def _run_race(self, run: Running) -> None:
        job = run.job
        race = self.store.get_race(job["race_id"])
        st = self.races.setdefault(race["id"], RaceState(race["id"]))
        lanes_cfg, errors = resolve_lanes(json.loads(race["lanes_json"]))
        if errors:
            raise ValueError("; ".join(errors))
        self.store.mark_running(job["id"])
        self.store.update_race(race["id"], status="running", started_at=time.time())
        log.info("race %s (%s) starting, %d lanes, %.0fs", race["id"], race["kind"], len(lanes_cfg), race["max_seconds"])
        try:
            data = self.doodle_data()
        except FileNotFoundError as e:
            raise RuntimeError(str(e)) from e
        lanes = [Lane(c, data, seed=0) for c in lanes_cfg]
        reason = run_race(lanes, race["max_seconds"], st.emit, lambda: st.abort.is_set() or run.should_abort())
        final = [l.snapshot() for l in lanes]
        self.store.update_race(race["id"], status="done", done_reason=reason, final_json=json.dumps(final), finished_at=time.time())
        self.store.finish_job(job["id"], "done", "budget" if reason != "aborted" else "killed")
        log.info("race %s finished: %s", race["id"], reason)

    # ------------------------------------------------------------ competition training
    def _run_competition(self, run: Running) -> None:
        job = run.job
        jid = job["id"]
        sub = self.store.get_submission(job["submission_id"])
        cfg, errors = resolve_config(json.loads(sub["config_json"]))
        if errors:
            raise ValueError("; ".join(errors))
        trainer = CompetitionTrainer(cfg, self.chess_data(), self.device, budget_flops=self.s.budget_flops,
                                     time_cap_s=self.s.time_cap_seconds, ckpt_every_s=self.s.checkpoint_seconds,
                                     metrics_every_s=self.s.metrics_seconds)
        ck = self.store.latest_checkpoint(jid)
        if ck and Path(ck["path"]).is_file():
            trainer.load_checkpoint(ck["path"])
            log.info("job %s resumed from checkpoint at step %d (%d samples)", jid, trainer.step, trainer.samples_seen)
        else:
            log.info("job %s starting (%s, %s params)", jid, sub["nickname"], f"{cfg['param_count']:,}")
        self.store.mark_running(jid)
        job = self.store.get_job(jid)

        def on_metrics(rec: dict) -> None:
            self.store.add_metrics(jid, rec)
            self.store.update_progress(jid, samples_seen=trainer.samples_seen, flops_used=trainer.flops_used,
                                       active_gpu_seconds=trainer.active_s)

        def on_checkpoint() -> None:
            path = self.s.checkpoints_dir / str(jid) / f"ckpt_{trainer.step}.pt"
            info = trainer.save_checkpoint(path)
            self._delete_files(self.store.add_checkpoint(jid, info))

        reason = trainer.run(run.should_abort, on_metrics, on_checkpoint)

        if reason == "aborted":
            why = run.reason
            log.info("job %s aborted (%s) at step %d", jid, why, trainer.step)
            if why in ("preempt", "shutdown"):
                self.store.requeue_front(jid, count_preemption=(why == "preempt"))
            elif why == "kill":
                self.store.finish_job(jid, "killed", "killed")
                self._delete_files(self.store.drop_checkpoints(jid))
            elif why == "redo":
                self._delete_files(self.store.reset_job_from_scratch(jid))
            return

        identity = {"nickname": sub["nickname"], "model_name": sub["model_name"], "participant_code": sub["participant_code"],
                    "submission_id": sub["id"]}
        out_dir = self.s.models_dir / sub["id"]
        if reason == "diverged":
            last = self.store.latest_checkpoint(jid)
            if last and Path(last["path"]).is_file():  # ship the last finite weights, not NaNs
                trainer.load_checkpoint(last["path"])
                trainer.export(out_dir, "diverged", preemptions=job["preemptions"], identity=identity, started_at=job["started_at"])
            else:
                log.warning("job %s diverged before any checkpoint: no model saved", jid)
        else:
            meta = trainer.export(out_dir, reason, preemptions=job["preemptions"], identity=identity, started_at=job["started_at"])
            log.info("job %s done: %s, val_mse %.4f", jid, reason, meta["val_mse"])
        self.store.update_progress(jid, samples_seen=trainer.samples_seen, flops_used=trainer.flops_used, active_gpu_seconds=trainer.active_s)
        self.store.finish_job(jid, "done", reason)
        self._delete_files(self.store.drop_checkpoints(jid))
        shutil.rmtree(self.s.checkpoints_dir / str(jid), ignore_errors=True)
