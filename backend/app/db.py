"""SQLite persistence (SPEC 6.3). WAL mode; schema only in phase 1.

Every piece of state that must survive a power cycle lives here: the server is
powered down and up during the event, and a crash must be handled exactly like a
clean shutdown. Open a new connection per request/thread with `connect()`.

Timestamps are Unix seconds (REAL). JSON blobs are stored as TEXT.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SCHEMA_VERSION = 1

TABLES = ("submissions", "jobs", "job_metrics", "checkpoints", "races", "admin_events")

SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id               TEXT PRIMARY KEY,
    participant_code TEXT NOT NULL UNIQUE,
    nickname         TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    contact          TEXT,                 -- private, never exported
    consent          INTEGER NOT NULL DEFAULT 0,
    config_json      TEXT NOT NULL,        -- full resolved config
    param_count      INTEGER NOT NULL,
    tier             TEXT NOT NULL,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS races (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('loss', 'lr', 'batch')),
    lanes_json   TEXT NOT NULL,
    max_seconds  REAL NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued', 'running', 'done', 'error')),
    done_reason  TEXT CHECK (done_reason IN ('time', 'complete', 'aborted')),
    final_json   TEXT,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);

CREATE TABLE IF NOT EXISTS jobs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    kind               TEXT NOT NULL CHECK (kind IN ('demo_race', 'competition')),
    submission_id      TEXT REFERENCES submissions(id) ON DELETE CASCADE,
    race_id            TEXT REFERENCES races(id) ON DELETE CASCADE,
    status             TEXT NOT NULL DEFAULT 'queued'
                       CHECK (status IN ('queued', 'running', 'done', 'killed', 'removed', 'error')),
    priority           INTEGER NOT NULL DEFAULT 0,   -- higher runs first (demo races)
    queue_order        REAL NOT NULL,                -- FCFS key; preempted jobs go to the front
    stop_reason        TEXT CHECK (stop_reason IN ('budget', 'time', 'diverged', 'killed', 'error')),
    samples_seen       INTEGER NOT NULL DEFAULT 0,
    flops_used         REAL NOT NULL DEFAULT 0,
    active_gpu_seconds REAL NOT NULL DEFAULT 0,
    preemptions        INTEGER NOT NULL DEFAULT 0,
    error_message      TEXT,
    created_at         REAL NOT NULL,
    started_at         REAL,
    finished_at        REAL,
    updated_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, priority DESC, queue_order);

CREATE TABLE IF NOT EXISTS job_metrics (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id       INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    split        TEXT NOT NULL CHECK (split IN ('train', 'val')),
    t            REAL NOT NULL,          -- active GPU seconds at this point
    step         INTEGER NOT NULL,
    samples_seen INTEGER NOT NULL,
    flops_used   REAL NOT NULL,
    loss         REAL,
    acc          REAL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS job_metrics_job ON job_metrics (job_id, split, step);

CREATE TABLE IF NOT EXISTS checkpoints (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id             INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    path               TEXT NOT NULL,
    step               INTEGER NOT NULL,
    samples_seen       INTEGER NOT NULL,
    flops_used         REAL NOT NULL,
    active_gpu_seconds REAL NOT NULL,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS checkpoints_job ON checkpoints (job_id, id DESC);

-- Audit log of admin actions. Also the durable source of the queue toggles:
-- the latest 'pause'/'resume' event decides `paused`, the latest 'demo_mode'
-- event (payload {"enabled": bool}) decides `demo_mode`.
CREATE TABLE IF NOT EXISTS admin_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    payload_json TEXT,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS admin_events_kind ON admin_events (kind, id DESC);
"""

# Jobs that still count as "in the queue" (waiting or running).
OPEN_JOB_STATUSES = ("queued", "running")


def connect(path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)  # autocommit; use BEGIN explicitly
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(path: Path | str) -> None:
    """Create the database and tables if missing. Safe to call on every start."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            raise RuntimeError(f"SQLite refused WAL mode (got {mode!r}); is {path} on a network filesystem?")
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    finally:
        conn.close()


def log_admin_event(conn: sqlite3.Connection, kind: str, payload: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO admin_events (kind, payload_json, created_at) VALUES (?, ?, ?)",
        (kind, json.dumps(payload) if payload is not None else None, time.time()),
    )


def queue_length(conn: sqlite3.Connection) -> int:
    marks = ",".join("?" * len(OPEN_JOB_STATUSES))
    row = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE status IN ({marks})", OPEN_JOB_STATUSES).fetchone()
    return int(row[0])


def is_paused(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT kind FROM admin_events WHERE kind IN ('pause', 'resume') ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row is not None and row["kind"] == "pause"


def is_demo_mode(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT payload_json FROM admin_events WHERE kind = 'demo_mode' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None or row["payload_json"] is None:
        return False
    return bool(json.loads(row["payload_json"]).get("enabled", False))
