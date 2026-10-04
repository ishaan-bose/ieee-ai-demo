"""All database operations for submissions, jobs, races, metrics and checkpoints (SPEC 6.3).

Every method opens its own short-lived connection (SQLite WAL handles the API threads and the worker thread). Nothing
depends on in-memory state surviving a restart.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from app import db

CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O/1/I/L
DEMO_PRIORITY = 100


def _row(r: sqlite3.Row | None) -> dict | None:
    return dict(r) if r is not None else None


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    @contextmanager
    def tx(self):
        conn = db.connect(self.path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @contextmanager
    def conn(self):
        c = db.connect(self.path)
        try:
            yield c
        finally:
            c.close()

    def _next_order(self, c) -> float:
        return float(c.execute("SELECT COALESCE(MAX(queue_order), 0) + 1 FROM jobs").fetchone()[0])

    # ------------------------------------------------------------ submissions
    def create_submission(self, *, nickname: str, model_name: str, contact: str | None, consent: bool, config: dict,
                          client_id: str | None = None) -> dict:
        """Insert submission + its competition job. A repeated client_id returns the existing one (idempotent upload)."""
        with self.tx() as c:
            if client_id:
                old = c.execute("SELECT id FROM submissions WHERE client_id = ?", (client_id,)).fetchone()
                if old:
                    return {**self._submission_summary(c, old["id"]), "duplicate": True}
            sid = uuid.uuid4().hex
            for _ in range(50):
                code = "AI-" + "".join(secrets.choice(CODE_ALPHABET) for _ in range(5))
                if not c.execute("SELECT 1 FROM submissions WHERE participant_code = ?", (code,)).fetchone():
                    break
            now = time.time()
            c.execute("INSERT INTO submissions (id, participant_code, nickname, model_name, contact, consent, config_json, "
                      "param_count, tier, created_at, client_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (sid, code, nickname, model_name, contact, int(consent), json.dumps(config), config["param_count"],
                       config["tier"], now, client_id))
            c.execute("INSERT INTO jobs (kind, submission_id, status, priority, queue_order, created_at, updated_at) "
                      "VALUES ('competition', ?, 'queued', 0, ?, ?, ?)", (sid, self._next_order(c), now, now))
            return self._submission_summary(c, sid)

    def _submission_summary(self, c, sid: str) -> dict:
        s = c.execute("SELECT * FROM submissions WHERE id = ?", (sid,)).fetchone()
        j = c.execute("SELECT id FROM jobs WHERE submission_id = ? ORDER BY id DESC LIMIT 1", (sid,)).fetchone()
        return {"submission_id": sid, "participant_code": s["participant_code"], "param_count": s["param_count"], "tier": s["tier"],
                "job_id": j["id"] if j else None}

    def get_submission(self, sid: str) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM submissions WHERE id = ?", (sid,)).fetchone())

    def submission_job(self, sid: str) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM jobs WHERE submission_id = ? ORDER BY id DESC LIMIT 1", (sid,)).fetchone())

    # ------------------------------------------------------------ races
    def create_race(self, race_id: str, kind: str, lanes: list[dict], max_seconds: float) -> int:
        with self.tx() as c:
            now = time.time()
            c.execute("INSERT INTO races (id, kind, lanes_json, max_seconds, status, created_at) VALUES (?,?,?,?, 'queued', ?)",
                      (race_id, kind, json.dumps(lanes), max_seconds, now))
            cur = c.execute("INSERT INTO jobs (kind, race_id, status, priority, queue_order, created_at, updated_at) "
                            "VALUES ('demo_race', ?, 'queued', ?, ?, ?, ?)", (race_id, DEMO_PRIORITY, self._next_order(c), now, now))
            return int(cur.lastrowid)

    def get_race(self, race_id: str) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM races WHERE id = ?", (race_id,)).fetchone())

    def update_race(self, race_id: str, **fields) -> None:
        if not fields:
            return
        with self.tx() as c:
            c.execute(f"UPDATE races SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?", (*fields.values(), race_id))

    def race_job(self, race_id: str) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM jobs WHERE race_id = ?", (race_id,)).fetchone())

    # ------------------------------------------------------------ jobs
    def get_job(self, job_id: int) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def next_job(self, allow_competition: bool) -> dict | None:
        with self.conn() as c:
            q = "SELECT * FROM jobs WHERE status = 'queued' AND (kind = 'demo_race' OR ?) ORDER BY priority DESC, queue_order ASC LIMIT 1"
            return _row(c.execute(q, (1 if allow_competition else 0,)).fetchone())

    def has_queued(self, kind: str | None = None) -> bool:
        with self.conn() as c:
            if kind:
                return c.execute("SELECT 1 FROM jobs WHERE status='queued' AND kind=? LIMIT 1", (kind,)).fetchone() is not None
            return c.execute("SELECT 1 FROM jobs WHERE status='queued' LIMIT 1").fetchone() is not None

    def mark_running(self, job_id: int) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?), updated_at=? WHERE id=?", (now, now, job_id))

    def update_progress(self, job_id: int, *, samples_seen: int, flops_used: float, active_gpu_seconds: float) -> None:
        with self.tx() as c:
            c.execute("UPDATE jobs SET samples_seen=?, flops_used=?, active_gpu_seconds=?, updated_at=? WHERE id=?",
                      (samples_seen, flops_used, active_gpu_seconds, time.time(), job_id))

    def finish_job(self, job_id: int, status: str, stop_reason: str | None = None, error_message: str | None = None) -> None:
        now = time.time()
        with self.tx() as c:
            c.execute("UPDATE jobs SET status=?, stop_reason=?, error_message=?, finished_at=?, updated_at=? WHERE id=?",
                      (status, stop_reason, error_message, now, now, job_id))

    def requeue_front(self, job_id: int, *, count_preemption: bool) -> None:
        """Back to the FRONT of the queue; counters restored from the last checkpoint (or zero)."""
        now = time.time()
        with self.tx() as c:
            front = c.execute("SELECT COALESCE(MIN(queue_order), 1) - 1 FROM jobs WHERE status='queued'").fetchone()[0]
            ck = c.execute("SELECT * FROM checkpoints WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,)).fetchone()
            s, f, a = (ck["samples_seen"], ck["flops_used"], ck["active_gpu_seconds"]) if ck else (0, 0.0, 0.0)
            c.execute("UPDATE jobs SET status='queued', queue_order=?, samples_seen=?, flops_used=?, active_gpu_seconds=?, "
                      "preemptions=preemptions+?, updated_at=? WHERE id=?", (front, s, f, a, 1 if count_preemption else 0, now, job_id))

    def reset_job_from_scratch(self, job_id: int) -> list[str]:
        """Redo: wipe metrics and checkpoints, queue at the END. Returns checkpoint file paths to delete."""
        now = time.time()
        with self.tx() as c:
            paths = [r["path"] for r in c.execute("SELECT path FROM checkpoints WHERE job_id=?", (job_id,))]
            c.execute("DELETE FROM checkpoints WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM job_metrics WHERE job_id=?", (job_id,))
            c.execute("UPDATE jobs SET status='queued', queue_order=?, stop_reason=NULL, error_message=NULL, samples_seen=0, "
                      "flops_used=0, active_gpu_seconds=0, preemptions=0, started_at=NULL, finished_at=NULL, updated_at=? WHERE id=?",
                      (self._next_order(c), now, job_id))
            return paths

    def queue_position(self, job_id: int) -> int | None:
        """1-based position among competition work (running counts as ahead); None if not queued."""
        with self.conn() as c:
            j = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if j is None or j["status"] != "queued":
                return None
            ahead = c.execute("SELECT COUNT(*) FROM jobs WHERE kind='competition' AND (status='running' OR (status='queued' AND "
                              "(priority > ? OR (priority = ? AND queue_order < ?))))", (j["priority"], j["priority"], j["queue_order"])).fetchone()[0]
            return int(ahead) + 1

    def recover_after_restart(self) -> dict:
        """A clean shutdown and a crash are handled the same way (SPEC 6.3): interrupted jobs go back to the front."""
        with self.tx() as c:
            running = [r["id"] for r in c.execute("SELECT id FROM jobs WHERE status='running' AND kind='competition' ORDER BY id")]
            races = c.execute("SELECT id, race_id FROM jobs WHERE kind='demo_race' AND status IN ('running','queued')").fetchall()
            now = time.time()
            for r in races:
                c.execute("UPDATE jobs SET status='error', stop_reason='error', error_message='server restarted', finished_at=?, updated_at=? WHERE id=?",
                          (now, now, r["id"]))
                c.execute("UPDATE races SET status='error', finished_at=? WHERE id=?", (now, r["race_id"]))
        for jid in reversed(running):
            self.requeue_front(jid, count_preemption=False)
        return {"requeued": running, "races_failed": [r["race_id"] for r in races]}

    # ------------------------------------------------------------ metrics and checkpoints
    def add_metrics(self, job_id: int, rec: dict) -> None:
        now = time.time()
        with self.tx() as c:
            for split, loss in (("train", rec.get("train_loss")), ("val", rec.get("val_loss"))):
                if loss is not None:
                    c.execute("INSERT INTO job_metrics (job_id, split, t, step, samples_seen, flops_used, loss, acc, created_at) "
                              "VALUES (?,?,?,?,?,?,?,?,?)", (job_id, split, rec["t"], rec["step"], rec["samples_seen"], rec["flops"],
                                                              float(loss), rec.get("val_mse") if split == "val" else None, now))

    def curves(self, job_id: int, limit: int = 400) -> dict:
        """{"train": [{t, step, loss}], "val": [{t, step, loss, val_mse}]} (the `acc` column stores val_mse for val rows)."""
        out: dict[str, list] = {"train": [], "val": []}
        with self.conn() as c:
            for split in out:
                rows = c.execute("SELECT t, step, samples_seen, loss, acc FROM job_metrics WHERE job_id=? AND split=? ORDER BY step DESC LIMIT ?",
                                 (job_id, split, limit)).fetchall()
                for r in reversed(rows):
                    item = {"t": r["t"], "step": r["step"], "samples_seen": r["samples_seen"], "loss": r["loss"]}
                    if split == "val":
                        item["val_mse"] = r["acc"]
                    out[split].append(item)
        return out

    def add_checkpoint(self, job_id: int, info: dict) -> list[str]:
        """Record a checkpoint, keep the newest two rows; returns paths that can be deleted."""
        with self.tx() as c:
            c.execute("INSERT INTO checkpoints (job_id, path, step, samples_seen, flops_used, active_gpu_seconds, created_at) VALUES (?,?,?,?,?,?,?)",
                      (job_id, info["path"], info["step"], info["samples_seen"], info["flops_used"], info["active_gpu_seconds"], time.time()))
            old = c.execute("SELECT id, path FROM checkpoints WHERE job_id=? ORDER BY id DESC LIMIT -1 OFFSET 2", (job_id,)).fetchall()
            for r in old:
                c.execute("DELETE FROM checkpoints WHERE id=?", (r["id"],))
            return [r["path"] for r in old if r["path"] != info["path"]]

    def latest_checkpoint(self, job_id: int) -> dict | None:
        with self.conn() as c:
            return _row(c.execute("SELECT * FROM checkpoints WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,)).fetchone())

    def checkpoint_paths(self, job_id: int) -> list[str]:
        with self.conn() as c:
            return [r["path"] for r in c.execute("SELECT path FROM checkpoints WHERE job_id=?", (job_id,))]

    def drop_checkpoints(self, job_id: int) -> list[str]:
        with self.tx() as c:
            paths = [r["path"] for r in c.execute("SELECT path FROM checkpoints WHERE job_id=?", (job_id,))]
            c.execute("DELETE FROM checkpoints WHERE job_id=?", (job_id,))
            return paths

    # ------------------------------------------------------------ admin views
    def list_jobs(self, include_removed: bool = False) -> list[dict]:
        q = ("SELECT j.*, s.nickname, s.model_name, s.participant_code, s.tier, s.param_count, s.config_json "
             "FROM jobs j LEFT JOIN submissions s ON s.id = j.submission_id "
             + ("" if include_removed else "WHERE j.status != 'removed' ")
             + "ORDER BY CASE j.status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END, j.priority DESC, j.queue_order, j.id")
        with self.conn() as c:
            return [dict(r) for r in c.execute(q)]

    def set_removed(self, job_id: int) -> None:
        with self.tx() as c:
            c.execute("UPDATE jobs SET status='removed', updated_at=? WHERE id=?", (time.time(), job_id))

    def export_rows(self, include_contact: bool = False) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT s.*, j.id AS job_id, j.status, j.stop_reason, j.samples_seen, j.flops_used, j.active_gpu_seconds, "
                             "j.preemptions, j.started_at, j.finished_at FROM submissions s LEFT JOIN jobs j ON j.submission_id = s.id "
                             "ORDER BY s.created_at").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d.pop("config_json"))
                d.pop("client_id", None)
                if not include_contact:
                    d.pop("contact", None)  # SPEC 13: contact info never leaves in an export unless explicitly requested
                last = c.execute("SELECT loss FROM job_metrics WHERE job_id=? AND split='val' ORDER BY step DESC LIMIT 1", (d["job_id"],)).fetchone()
                d["final_val_loss"] = last["loss"] if last else None
                out.append(d)
            return out

    def snapshot(self, dest: Path) -> Path:
        """Consistent copy of the database (SQLite online backup API)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = db.connect(self.path)
        try:
            dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        return dest

    def clear_runs(self) -> None:
        with self.tx() as c:
            for t in ("job_metrics", "checkpoints", "jobs", "races", "submissions"):
                c.execute(f"DELETE FROM {t}")
            c.execute("DELETE FROM sqlite_sequence WHERE name IN ('jobs','job_metrics','checkpoints')")

    # ------------------------------------------------------------ queue toggles (stored as admin events)
    def is_paused(self) -> bool:
        with self.conn() as c:
            return db.is_paused(c)

    def is_demo_mode(self) -> bool:
        with self.conn() as c:
            return db.is_demo_mode(c)

    def log_event(self, kind: str, payload: dict | None = None) -> None:
        with self.tx() as c:
            db.log_admin_event(c, kind, payload)
