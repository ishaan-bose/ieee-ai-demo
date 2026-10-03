import sqlite3

import pytest

from app import db


def test_wal_and_tables(tmp_path):
    path = tmp_path / "x" / "demo.db"
    db.init_db(path)
    db.init_db(path)  # idempotent: safe on every start
    conn = db.connect(path)
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert set(db.TABLES) <= names
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_state_helpers(tmp_path):
    path = tmp_path / "demo.db"
    db.init_db(path)
    conn = db.connect(path)
    assert db.queue_length(conn) == 0 and not db.is_paused(conn) and not db.is_demo_mode(conn)

    now = 1.0
    conn.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('competition','queued',1,?,?)", (now, now))
    conn.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('competition','running',2,?,?)", (now, now))
    conn.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('competition','done',3,?,?)", (now, now))
    assert db.queue_length(conn) == 2

    db.log_admin_event(conn, "pause")
    db.log_admin_event(conn, "demo_mode", {"enabled": True})
    assert db.is_paused(conn) and db.is_demo_mode(conn)
    db.log_admin_event(conn, "resume")
    db.log_admin_event(conn, "demo_mode", {"enabled": False})
    assert not db.is_paused(conn) and not db.is_demo_mode(conn)
    conn.close()

    # state survives a "power cycle" (new connection = new process)
    conn = db.connect(path)
    assert db.queue_length(conn) == 2 and not db.is_paused(conn)


def test_checks_reject_bad_values(tmp_path):
    path = tmp_path / "demo.db"
    db.init_db(path)
    conn = db.connect(path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('nope','queued',1,0,0)")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO jobs (kind, status, stop_reason, queue_order, created_at, updated_at) "
                     "VALUES ('competition','done','bored',1,0,0)")
