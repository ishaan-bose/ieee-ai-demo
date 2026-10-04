"""The scheduler (SPEC 6.1): a dispatcher thread, a FCFS queue, up to MAX_CONCURRENT_JOBS competition jobs at once, each in its OWN process.

- FCFS: jobs START in queue order (priority, then queue_order). A job whose estimated VRAM does not fit the GPU right now makes the ones behind it wait
  (nothing overtakes it); fewer jobs run until memory frees up. After a CUDA out-of-memory error the job goes back to the FRONT of the queue and the live
  concurrency drops by one for a while (it climbs back one step per cool-down).
- A demo race (priority 100) takes the whole GPU: ALL running competition jobs are aborted (they stop within a step or two WITHOUT saving), the race runs,
  and the interrupted jobs return to the FRONT of the queue in their ORIGINAL order, each resuming from its last periodic checkpoint (no samples are
  double counted). The race itself runs in the dispatcher thread (it needs no process of its own: nothing else is on the GPU).
- Demo Mode and pause keep competition jobs off the GPU entirely; admin kill / redo / remove act on one job, reset on all.
- Everything durable is in SQLite; on start, interrupted jobs are re-queued (a crash and a clean shutdown are treated the same).
- When the queue is empty the dispatcher idles and does nothing.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue as queue_mod
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
from app.scheduler import vram
from app.scheduler.job_process import ABORT_CODES, worker_main
from app.store import Store
from app.training.race import DoodleData, Lane, resolve_lanes, run_race

log = logging.getLogger("scheduler")

OOM_COOLDOWN_S = 120.0  # after an out-of-memory error: run one job fewer for this long, then climb back one step at a time
MAX_CRASHES = 3  # a job whose process dies without a result this many times is failed instead of retried forever


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
    """A job that is on the GPU: a competition job in a child process (`proc`), or the demo race in the dispatcher thread."""

    def __init__(self, job: dict, proc=None, flag=None, out_q=None, est_vram: int = 0):
        self.job, self.job_id, self.kind = job, job["id"], job["kind"]
        self.reason: str | None = None  # abort reason: preempt | kill | redo | shutdown
        self.proc, self.flag, self.out_q, self.est_vram = proc, flag, out_q, est_vram
        self.result: dict | None = None
        self.peak = 1  # most competition jobs that were running together while this one ran

    def abort(self, reason: str) -> None:
        if self.reason is None or ABORT_PRIORITY[reason] > ABORT_PRIORITY[self.reason]:
            self.reason = reason
            if self.flag is not None:
                self.flag.value = ABORT_CODES[reason]

    def should_abort(self) -> bool:
        return self.reason is not None


class Scheduler:
    def __init__(self, settings: Settings, device: DeviceInfo, store: Store):
        self.s, self.store, self.device = settings, store, torch.device(device.device)
        self.running: dict[int, Running] = {}  # competition jobs, one child process each
        self.race_run: Running | None = None  # the demo race currently on the GPU
        self.races: dict[str, RaceState] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._doodles: DoodleData | None = None
        self._guard = threading.Lock()
        self._ctx = mp.get_context("spawn")
        self._pending_requeue: list[tuple[float, int, bool]] = []  # preempted jobs waiting for their siblings to stop, so they go back in order
        self._crashes: dict[int, int] = {}
        self._oom_penalty = 0  # how many slots are withheld after out-of-memory errors
        self._penalty_until = 0.0
        self.oom_cooldown_s = OOM_COOLDOWN_S
        self._data_bytes: int | None = None
        self._wait_logged: int | None = None
        self.mem_info = self._cuda_mem_info if self.device.type == "cuda" else None  # () -> (free, total) bytes; tests replace it

    # ------------------------------------------------------------ properties used by the API
    @property
    def current(self) -> Running | None:
        """One running job (the first competition job, else the race): what /admin shows as 'the' current job."""
        with self._guard:
            return next(iter(self.running.values()), None) or self.race_run

    def running_jobs(self) -> list[Running]:
        with self._guard:
            return list(self.running.values())

    @property
    def max_jobs(self) -> int:
        if self.device.type == "cuda" or self.s.max_concurrent_explicit:
            return max(1, self.s.max_concurrent_jobs)
        return 1  # a CPU-only box: processes would only fight over the same cores

    def effective_limit(self) -> int:
        """MAX_CONCURRENT_JOBS minus the slots withheld after out-of-memory errors (restored one per cool-down)."""
        now = time.time()
        while self._oom_penalty > 0 and now >= self._penalty_until:
            self._oom_penalty -= 1
            self._penalty_until = now + self.oom_cooldown_s
            log.info("concurrency raised to %d after the out-of-memory cool-down", max(1, self.max_jobs - self._oom_penalty))
        return max(1, self.max_jobs - self._oom_penalty)

    def _cuda_mem_info(self) -> tuple[int, int]:
        return torch.cuda.mem_get_info(self.device)

    # ------------------------------------------------------------ lifecycle
    def start(self) -> dict:
        rec = self.store.recover_after_restart()
        if rec["requeued"] or rec["races_failed"]:
            log.info("recovered after restart: re-queued jobs %s, failed races %s", rec["requeued"], rec["races_failed"])
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
        self._thread.start()
        return rec

    def stop(self, timeout: float = 25.0) -> None:
        self._stop.set()
        for r in self.running_jobs():
            r.abort("shutdown")
        if self.race_run:
            self.race_run.abort("shutdown")
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    # ------------------------------------------------------------ data (loaded lazily, once)
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
        """The GPU is needed for something else: every running competition job leaves it (and comes back later, in order)."""
        runs = self.running_jobs()
        for r in runs:
            if r.reason is None:  # (the dispatcher repeats this every tick until the children have stopped: log once)
                log.info("preempting competition job %s", r.job_id)
            r.abort("preempt")
        return bool(runs)

    def kill_all(self) -> None:
        for r in self.running_jobs():
            r.abort("kill")
        if self.race_run:
            self.race_run.abort("kill")

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

    def _running(self, job_id: int) -> Running | None:
        """The live Running of a job that is training right now. A child writes 'done' a moment BEFORE the parent notices its process exited: such a job
        is finished, not running."""
        with self._guard:
            run = self.running.get(job_id)
        if run is not None:
            job = self.store.get_job(job_id)
            if job is None or job["status"] != "running":
                return None
        return run

    def kill_job(self, job_id: int) -> str:
        run = self._running(job_id)
        if run:
            run.abort("kill")
            return "killing"
        job = self.store.get_job(job_id)
        if job and job["status"] == "queued":
            self.store.finish_job(job_id, "killed", "killed")
            self._cleanup_job_files(job)
            return "killed"
        return "not_running"

    def redo_job(self, job_id: int) -> str:
        run = self._running(job_id)
        if run:
            run.abort("redo")
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
        while (self.running or self.race_run is not None) and time.time() - t0 < timeout:
            time.sleep(0.05)
        return not self.running and self.race_run is None

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

    # ------------------------------------------------------------ the dispatcher
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_children()
                paused = self.store.is_paused() or self.store.is_demo_mode()
                if paused and self.running:
                    self.preempt_competition()
                if self.store.has_queued("demo_race"):
                    if self.running or self._pending_requeue:
                        self.preempt_competition()  # make room: the race gets the whole GPU
                    else:
                        job = self.store.next_job(False)
                        if job is not None:
                            self._run_race_job(job)
                            continue
                elif not paused and not self._pending_requeue:
                    self._fill_slots()
            except Exception:  # noqa: BLE001  the dispatcher must never die
                log.error("scheduler error: %s", traceback.format_exc())
                time.sleep(1.0)
                continue
            self._wake.wait(0.1 if (self.running or self._pending_requeue) else 0.5)
            self._wake.clear()
        self._shutdown_children()

    # ------------------------------------------------------------ starting competition jobs
    def _data_resident_bytes(self) -> int:
        if self._data_bytes is None:
            try:
                from app.training.chess_data import FIELDS

                self._data_bytes = int(LichessData(self.s.data_dir / "lichess").nbytes(FIELDS)) if self.device.type == "cuda" else 0
            except Exception:  # noqa: BLE001
                self._data_bytes = 0
        return self._data_bytes

    def _estimate(self, job: dict) -> int:
        sub = self.store.get_submission(job["submission_id"])
        cfg, errors = resolve_config(json.loads(sub["config_json"]))
        if errors:
            return 0  # the child will fail the job with the validation message
        return vram.estimate_job_vram(cfg, self._data_resident_bytes())

    def _fits(self, job: dict, est: int) -> bool:
        if self.mem_info is None or not self.running:
            return True  # CPU box, or nothing else is running: the job gets the GPU (a job that cannot fit even alone shrinks its micro-batches)
        free, total = self.mem_info()
        committed = sum(r.est_vram for r in self.running_jobs())
        return vram.admit(est, free, total, committed)

    def _fill_slots(self) -> None:
        limit = self.effective_limit()
        while len(self.running) < limit and not self._stop.is_set():
            job = self.store.next_job(True)
            if job is None:
                self._wait_logged = None
                return
            if job["id"] in self.running:  # re-queued (redo) while its old process is still being reaped: start it once that is done
                return
            est = self._estimate(job)
            if not self._fits(job, est):
                if self._wait_logged != job["id"]:  # once per job, not once per poll
                    self._wait_logged = job["id"]
                    log.info("job %s waits: its estimated %.1f GB do not fit next to the %d running job(s) (strict FCFS: nothing overtakes it)",
                             job["id"], est / 2**30, len(self.running))
                return
            self._start_child(job, est)

    def _start_child(self, job: dict, est: int) -> None:
        flag = self._ctx.Value("i", 0)
        out_q = self._ctx.Queue()
        self.store.mark_running(job["id"])
        proc = self._ctx.Process(target=worker_main, args=(self.s, str(self.device), job["id"], flag, out_q, os.getpid()), daemon=True,
                                 name=f"job-{job['id']}")
        run = Running(job, proc, flag, out_q, est)
        with self._guard:
            self.running[job["id"]] = run
        proc.start()
        n = len(self.running)
        for r in self.running_jobs():
            r.peak = max(r.peak, n)
        log.info("job %s started in process %s (%d running, limit %d)", job["id"], proc.pid, n, self.effective_limit())

    # ------------------------------------------------------------ watching them
    def _drain(self, run: Running) -> None:
        while True:
            try:
                msg = run.out_q.get_nowait()
            except queue_mod.Empty:
                return
            except (EOFError, OSError):
                return
            if msg[0] == "log":
                log.log(msg[1], "%s", msg[2])
            elif msg[0] == "result":
                run.result = msg[1]

    def _poll_children(self) -> None:
        for run in self.running_jobs():
            self._drain(run)
            if not run.proc.is_alive():
                run.proc.join(2.0)
                self._drain(run)
                self._reap(run)
        self._flush_requeue()

    def _flush_requeue(self) -> None:
        """Preempted jobs go back to the front only once every sibling that is also being preempted has stopped, so that their ORIGINAL order is kept."""
        if self._pending_requeue and not any(r.reason in ("preempt", "shutdown") for r in self.running_jobs()):
            self.store.requeue_front_in_order(self._pending_requeue)
            log.info("re-queued at the front, in order: jobs %s", [j for _o, j, _p in sorted(self._pending_requeue)])
            self._pending_requeue = []

    def _reap(self, run: Running) -> None:
        jid = run.job_id
        with self._guard:
            self.running.pop(jid, None)
        res = run.result
        try:
            if res is None:  # the process died without a word (segfault, CUDA fault, killed from outside)
                if run.reason in ("preempt", "shutdown"):
                    self._pending_requeue.append((run.job["queue_order"], jid, run.reason == "preempt"))
                elif run.reason == "kill":
                    self._finish_killed(jid)
                elif run.reason == "redo":
                    self._delete_files(self.store.reset_job_from_scratch(jid))
                else:
                    n = self._crashes[jid] = self._crashes.get(jid, 0) + 1
                    log.error("job %s: its process died (exit code %s), attempt %d of %d", jid, run.proc.exitcode, n, MAX_CRASHES)
                    if n >= MAX_CRASHES:
                        self.store.finish_job(jid, "error", "error", f"worker process died (exit code {run.proc.exitcode})")
                        self._log_finished(run, {"outcome": "error"}, "error")
                    else:
                        self.store.requeue_front(jid, count_preemption=False)
            elif res["outcome"] == "aborted":
                log.info("job %s aborted (%s) at step %d", jid, run.reason, res.get("steps", 0))
                if run.reason == "kill":
                    self._finish_killed(jid)
                    self._log_finished(run, res, "killed")
                elif run.reason == "redo":
                    self._delete_files(self.store.reset_job_from_scratch(jid))
                else:  # preempt / shutdown (or an orphan check that fired for some other reason): back to the front, in order
                    self._pending_requeue.append((run.job["queue_order"], jid, run.reason == "preempt"))
            elif res["outcome"] == "oom":
                self._oom_penalty = min(self._oom_penalty + 1, self.max_jobs - 1)
                self._penalty_until = time.time() + self.oom_cooldown_s
                self.store.requeue_front(jid, count_preemption=False)
                log.warning("job %s ran out of GPU memory (%s): back to the FRONT of the queue; concurrency lowered to %d for %.0f s",
                            jid, res.get("message", ""), self.effective_limit(), self.oom_cooldown_s)
            else:  # done or error: the child already wrote the terminal state
                self._log_finished(run, res, res.get("reason") or "error")
        finally:
            self._wake.set()

    def _finish_killed(self, jid: int) -> None:
        self.store.finish_job(jid, "killed", "killed")
        self._delete_files(self.store.drop_checkpoints(jid))

    def _log_finished(self, run: Running, res: dict, stop: str) -> None:
        """One line per finished job (samples/s, active GPU seconds, the concurrency it ran under, why it stopped), and a warning for 'time'."""
        active, samples = float(res.get("active_s", 0.0)), int(res.get("samples_seen", 0))
        sps = samples / active if active > 0 else None
        self.store.set_job_stats(run.job_id, concurrency=run.peak, samples_per_s=sps)
        log.info("job %s finished: stop reason %s, %s samples/s, %.1f active GPU s, ran with up to %d job(s) at once", run.job_id, stop,
                 f"{sps:,.0f}" if sps else "n/a", active, run.peak)
        if stop == "time":
            done = min(1.0, float(res.get("flops_used", 0.0)) / self.s.budget_flops) if self.s.budget_flops else 0.0
            log.warning("job %s hit the %.0f s active-GPU-time cap (stop reason \"time\") after %.0f%% of its FLOPs budget: concurrency (up to %d jobs at "
                        "once, limit %d) may be too high. Lower MAX_CONCURRENT_JOBS in backend/.env.", run.job_id, self.s.time_cap_seconds, 100 * done,
                        run.peak, self.max_jobs)

    def _shutdown_children(self) -> None:
        """Server stopping: every child stops without saving, then the interrupted jobs go back to the front in order (recovery would do the same)."""
        for r in self.running_jobs():
            r.abort("shutdown")
        t0 = time.time()
        while self.running and time.time() - t0 < 20.0:
            self._poll_children()
            time.sleep(0.05)
        for r in self.running_jobs():  # did not stop in time: end the process; the job is still 'running' in the database and is recovered at the next start
            log.warning("job %s did not stop in time: terminating its process", r.job_id)
            r.proc.terminate()
            r.proc.join(2.0)
            with self._guard:
                self.running.pop(r.job_id, None)
        self._flush_requeue()

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

    def _run_race_job(self, job: dict) -> None:
        run = Running(job)
        self.race_run = run
        try:
            self._run_race(run)
        except Exception as e:  # noqa: BLE001
            log.error("job %s failed: %s", job["id"], traceback.format_exc())
            self.store.finish_job(job["id"], "error", "error", f"{type(e).__name__}: {e}")
            self._fail_race(job["race_id"], f"{type(e).__name__}: {e}")
        finally:
            self.race_run = None
