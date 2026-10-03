"""Phase 6 backend: submissions, scheduler (preemption + resume), admin endpoints, restart recovery, race SSE."""

import json
import shutil
import sqlite3
import time

import pytest

from app import db
from app.chess_net.model_io import load_model
from app.store import Store
from tests.conftest import requires_torch

pytestmark = requires_torch
TOKEN = "test-token"
H = {"X-Admin-Token": TOKEN}
SMALL = {"layers": 2, "width": 32, "batch_size": 64}


@pytest.fixture(scope="session")
def server_data(tmp_path_factory, fake_data_dir):
    """Fake data + rasterized quickdraw tensors, so races can run."""
    import os

    from scripts import rasterize_quickdraw as rq

    root = tmp_path_factory.mktemp("serverdata")
    shutil.copytree(fake_data_dir, root, dirs_exist_ok=True)
    os.environ["LOG_DIR"] = str(root / "logs")
    rq.main(["--data-dir", str(root), "--workers", "1"])
    return root


@pytest.fixture
def env(monkeypatch, tmp_path, server_data):
    monkeypatch.setenv("DATA_DIR", str(server_data))
    monkeypatch.setenv("STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ADMIN_TOKEN", TOKEN)
    monkeypatch.setenv("SELF_CHECK", "0")
    monkeypatch.setenv("BUDGET_FLOPS", "2e10")
    monkeypatch.setenv("CHECKPOINT_SECONDS", "0.3")
    monkeypatch.setenv("METRICS_SECONDS", "0.2")
    monkeypatch.setenv("VAL_ROWS", "200")
    return tmp_path / "state"


@pytest.fixture
def client(env, request, monkeypatch):
    """ONE app (one scheduler) per test. @pytest.mark.budget("6e10") overrides the FLOPs budget for long-running jobs."""
    from fastapi.testclient import TestClient

    from app.main import app

    m = request.node.get_closest_marker("budget")
    if m:
        monkeypatch.setenv("BUDGET_FLOPS", m.args[0])
    with TestClient(app) as c:
        yield c


def submit(client, name="alice", config=None, consent=True, **kw):
    body = {"nickname": name, "model_name": f"{name}-net", "consent": consent, "config": config or SMALL, **kw}
    return client.post("/api/submissions", json=body)


def wait_for(fn, timeout=60.0, every=0.1, msg="condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {msg}")


def status(client, sid):
    return client.get(f"/api/submissions/{sid}").json()


# ---------------------------------------------------------------- migration / recovery

def test_migration_from_v1_database(tmp_path):
    """The owner's server already has a phase-1 (v1) database: it must be upgraded in place."""
    p = tmp_path / "demo.db"
    conn = sqlite3.connect(p)
    conn.executescript(db.SCHEMA)
    conn.execute("INSERT INTO submissions (id, participant_code, nickname, model_name, consent, config_json, param_count, tier, created_at) "
                 "VALUES ('x','AI-X','n','m',1,'{}',1,'Light',0)")
    conn.commit()
    conn.execute("PRAGMA user_version=1")
    conn.close()
    db.init_db(p)
    c = db.connect(p)
    assert "client_id" in [r[1] for r in c.execute("PRAGMA table_info(submissions)")]
    assert c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1  # data survived
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    db.init_db(p)  # idempotent


def test_recovery_requeues_interrupted_jobs_at_the_front(tmp_path):
    db.init_db(tmp_path / "d.db")
    st = Store(tmp_path / "d.db")
    cfg = {"param_count": 1, "tier": "Light"}
    a = st.create_submission(nickname="a", model_name="a", contact=None, consent=True, config=cfg)
    b = st.create_submission(nickname="b", model_name="b", contact=None, consent=True, config=cfg)
    c = st.create_submission(nickname="c", model_name="c", contact=None, consent=True, config=cfg)
    st.mark_running(b["job_id"])  # b was running when the box lost power
    st.create_race("r1", "loss", [], 5)
    st.mark_running(st.race_job("r1")["id"])
    rec = st.recover_after_restart()
    assert rec["requeued"] == [b["job_id"]] and rec["races_failed"] == ["r1"]
    assert st.next_job(True)["id"] == b["job_id"]  # back at the front, ahead of a
    assert st.get_job(b["job_id"])["preemptions"] == 0  # a crash is not a preemption
    assert st.get_race("r1")["status"] == "error"
    assert {a["job_id"], c["job_id"]} <= {j["id"] for j in st.list_jobs()}


# ---------------------------------------------------------------- submissions

def test_preview(client):
    ok = client.post("/api/submissions/preview", json=SMALL).json()
    assert ok["valid"] and ok["errors"] == [] and ok["tier"] == "Light" and ok["search_depth_full_moves"] == 3
    assert ok["param_count"] == 773 * 32 + 32 + 32 * 32 + 32 + 32 + 1 and ok["est_flops_budget"] == 2e10
    bad = client.post("/api/submissions/preview", json={"layers": 16, "width": 4096}).json()
    assert not bad["valid"] and any("too many parameters" in e for e in bad["errors"])
    assert not client.post("/api/submissions/preview", json={"layers": 0}).json()["valid"]
    assert client.post("/api/submissions/preview", json={}).json()["valid"]  # defaults apply
    # removed knobs are ignored silently
    assert client.post("/api/submissions/preview", json={"weight_decay": 0.1, "dropout": 0.3}).json()["valid"]


def test_config_schema_endpoint(client):
    s = client.get("/api/config/schema").json()
    assert s["max_params"] == 30_000_000 and [t["name"] for t in s["tiers"]] == ["Light", "Medium", "Heavy"]
    assert "relu" in s["activations"] and s["defaults"]["optimizer"] == "adam"


def test_submit_validation_and_errors(client):
    assert submit(client, consent=False).json()["error"] == "consent_required"
    r = submit(client, config={"layers": 99})
    assert r.status_code == 422 and r.json()["error"] == "invalid_config"
    assert submit(client, name="   ").status_code == 422
    assert client.post("/api/submissions", json={"nickname": "x"}).json()["error"] == "validation_error"
    assert client.get("/api/submissions/nope").json() == {"error": "not_found", "message": "unknown submission"}


def test_submit_is_idempotent_with_client_id(client, env):
    a = submit(client, client_id="abc").json()
    b = submit(client, client_id="abc").json()
    assert a["submission_id"] == b["submission_id"] and a["participant_code"] == b["participant_code"]
    assert a["participant_code"].startswith("AI-") and len(a["participant_code"]) == 8
    assert len(client.get("/admin/queue", headers=H).json()["jobs"]) == 1


def test_competition_job_runs_to_completion(client, env):
    r = submit(client, config={**SMALL, "ema": {"enabled": True, "decay": 0.99}}).json()
    assert r["status"] == "queued" and r["tier"] == "Light" and r["queue_position"] >= 1
    sid = r["submission_id"]
    done = wait_for(lambda: (s := status(client, sid))["status"] == "done" and s, msg="job done")
    assert done["stop_reason"] == "budget" and done["progress"]["fraction"] >= 0.99
    assert done["val_mse"] is not None and done["curves"]["val"], done
    m = env / "models" / sid
    assert sorted(p.name for p in m.iterdir()) == ["config.json", "curves.json", "meta.json", "weights.safetensors"]
    meta = json.loads((m / "meta.json").read_text())
    assert meta["stop_reason"] == "budget" and meta["flops_used"] >= 2e10 and meta["preemptions"] == 0
    assert load_model(m).name == "alice-net"
    assert not (env / "checkpoints" / "1").exists()  # temporary checkpoints are cleaned up
    assert client.get("/api/health").json()["queue_length"] == 0


def test_fcfs_order(client, env):
    client.post("/admin/demo-mode", json={"enabled": True}, headers=H)  # hold the queue while submitting
    ids = [submit(client, name=n).json()["submission_id"] for n in ("a", "b", "c")]
    jobs = client.get("/admin/queue", headers=H).json()["jobs"]
    assert [j["nickname"] for j in jobs] == ["a", "b", "c"]
    assert [status(client, i)["queue_position"] for i in ids] == [1, 2, 3]
    client.post("/admin/demo-mode", json={"enabled": False}, headers=H)
    for i in ids:
        wait_for(lambda i=i: status(client, i)["status"] == "done", msg="fcfs job")
    t = [json.loads((env / "models" / i / "meta.json").read_text())["finished_at"] for i in ids]
    assert t == sorted(t)


# ---------------------------------------------------------------- races

def read_sse(client, path, until=("done", "error"), limit=200):
    events = []
    with client.stream("GET", path) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                if block.startswith(":"):
                    continue
                f = dict(l.split(": ", 1) for l in block.splitlines())
                events.append((f["event"], json.loads(f["data"])))
                if f["event"] in until or len(events) >= limit:
                    return events
    return events


def race_body(seconds=1.0, **lane):
    return {"kind": "loss", "max_seconds": seconds, "lanes": [{"id": "a", "config": {"loss": "ce", **lane}}, {"id": "b", "config": {"loss": "mse"}}]}


def test_race_api_end_to_end(client):
    r = client.post("/api/demo/races", json=race_body(1.0))
    assert r.status_code == 200
    ev = read_sse(client, f"/api/demo/races/{r.json()['race_id']}/stream")
    kinds = [k for k, _ in ev]
    assert kinds[-1] == "done" and kinds.count("tick") >= 2 and ev[-1][1]["reason"] == "time"
    tick = [d for k, d in ev if k == "tick"][-1]
    assert [l["id"] for l in tick["lanes"]] == ["a", "b"] and len(tick["lanes"][0]["probe_preds"]) == 16
    assert len(ev[-1][1]["final"]) == 2


def test_race_validation_and_unknown(client):
    assert client.post("/api/demo/races", json={**race_body(), "kind": "weird"}).status_code == 422
    assert client.post("/api/demo/races", json={**race_body(), "max_seconds": 9999}).status_code == 422
    assert client.post("/api/demo/races", json={**race_body(), "lanes": []}).status_code == 422
    bad = client.post("/api/demo/races", json=race_body(1.0, lr=0))
    assert bad.status_code == 422 and bad.json()["error"] == "validation_error"
    assert client.get("/api/demo/races/nope/stream").status_code == 404
    assert client.post("/api/demo/races/nope/abort").status_code == 404


def test_race_abort(client):
    rid = client.post("/api/demo/races", json=race_body(30.0)).json()["race_id"]
    wait_for(lambda: client.app.state.scheduler.races[rid].events, msg="first event")
    assert client.post(f"/api/demo/races/{rid}/abort").json() == {"ok": True}
    ev = read_sse(client, f"/api/demo/races/{rid}/stream")
    assert ev[-1][0] == "done" and ev[-1][1]["reason"] == "aborted"


def test_races_unavailable_without_tensors(monkeypatch, tmp_path, fake_data_dir):
    from fastapi.testclient import TestClient

    from app.main import app

    for k, v in {"DATA_DIR": str(fake_data_dir), "STATE_DIR": str(tmp_path / "s"), "LOG_DIR": str(tmp_path / "l"), "SELF_CHECK": "0"}.items():
        monkeypatch.setenv(k, v)
    with TestClient(app) as c:
        r = c.post("/api/demo/races", json=race_body())
        assert r.status_code == 503 and r.json()["error"] == "data_unavailable"


# ---------------------------------------------------------------- the headline behaviour: demo preempts competition

@pytest.mark.budget("6e10")
def test_demo_race_preempts_competition_job_which_resumes_from_checkpoint(client, env):
    c = client
    if True:
        sid = submit(c, config={**SMALL, "batch_size": 32}).json()["submission_id"]
        jid = c.get("/admin/queue", headers=H).json()["jobs"][0]["id"]
        wait_for(lambda: c.app.state.store.latest_checkpoint(jid), msg="first checkpoint")
        before = c.app.state.store.latest_checkpoint(jid)
        assert status(c, sid)["status"] == "running"
        rid = c.post("/api/demo/races", json=race_body(1.0)).json()["race_id"]
        # the competition job leaves the GPU right away and returns to the front of the queue
        wait_for(lambda: c.app.state.store.get_job(jid)["status"] == "queued", timeout=10, msg="preempted job requeued")
        job = c.app.state.store.get_job(jid)
        assert job["preemptions"] == 1 and job["samples_seen"] >= before["samples_seen"] - 1
        ev = read_sse(c, f"/api/demo/races/{rid}/stream")  # the race itself runs normally
        assert ev[-1][0] == "done" and ev[-1][1]["reason"] == "time"
        done = wait_for(lambda: (s := status(c, sid))["status"] == "done" and s, timeout=90, msg="resumed job done")
        assert done["stop_reason"] == "budget" and done["progress"]["preemptions"] == 1
        meta = json.loads((env / "models" / sid / "meta.json").read_text())
        assert meta["preemptions"] == 1
        # no double counting: samples == steps * batch, flops == exact accounting, only one step of overshoot allowed
        assert meta["samples_seen"] == meta["steps"] * 32
        fps = 6 * (773 * 32 + 32 * 32 + 32)  # 6 x matmul weights of the 2x32 net
        assert meta["flops_used"] == pytest.approx(fps * meta["samples_seen"], rel=1e-9)
        assert 6e10 <= meta["flops_used"] < 6e10 + fps * 32 + 1


def test_demo_mode_holds_competition_but_not_races(client, env):
    client.post("/admin/demo-mode", json={"enabled": True}, headers=H)
    assert client.get("/api/health").json()["demo_mode"] is True
    sid = submit(client).json()["submission_id"]
    rid = client.post("/api/demo/races", json=race_body(0.6)).json()["race_id"]
    assert read_sse(client, f"/api/demo/races/{rid}/stream")[-1][0] == "done"
    time.sleep(0.8)
    assert status(client, sid)["status"] == "queued"  # still held
    client.post("/admin/demo-mode", json={"enabled": False}, headers=H)
    wait_for(lambda: status(client, sid)["status"] == "done", msg="job after demo mode off")


def test_pause_resume(client):
    client.post("/admin/queue/pause", headers=H)
    h = client.get("/api/health").json()
    assert h["paused"] is True
    sid = submit(client).json()["submission_id"]
    time.sleep(0.8)
    assert status(client, sid)["status"] == "queued"
    client.post("/admin/queue/resume", headers=H)
    assert client.get("/api/health").json()["paused"] is False
    wait_for(lambda: status(client, sid)["status"] == "done", msg="job after resume")


# ---------------------------------------------------------------- admin

def test_admin_auth(client, monkeypatch):
    assert client.get("/admin/queue").status_code == 401
    assert client.get("/admin/queue", headers={"X-Admin-Token": "wrong"}).json()["error"] == "unauthorized"
    assert client.get("/admin/queue", headers=H).status_code == 200
    for method, path in (("post", "/admin/queue/pause"), ("get", "/admin/export"), ("get", "/admin/logs"), ("post", "/admin/reset")):
        assert getattr(client, method)(path).status_code in (401, 422), path


def test_admin_disabled_without_token(monkeypatch, env, server_data):
    monkeypatch.delenv("ADMIN_TOKEN")
    monkeypatch.setattr("app.config.load_env_file", lambda p: None)
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        r = c.get("/admin/queue", headers=H)
        assert r.status_code == 503 and r.json()["error"] == "admin_disabled"


def test_kill_remove_redo(client, env, monkeypatch):
    client.post("/admin/demo-mode", json={"enabled": True}, headers=H)
    a = submit(client, name="a").json()["submission_id"]
    b = submit(client, name="b").json()["submission_id"]
    jobs = {j["nickname"]: j["id"] for j in client.get("/admin/queue", headers=H).json()["jobs"]}
    # remove a queued job
    assert client.delete(f"/admin/jobs/{jobs['a']}", headers=H).json() == {"ok": True, "status": "removed"}
    assert status(client, a)["status"] == "removed"
    assert [j["nickname"] for j in client.get("/admin/queue", headers=H).json()["jobs"]] == ["b"]
    assert client.delete("/admin/jobs/9999", headers=H).status_code == 404
    # kill a queued job
    assert client.post(f"/admin/jobs/{jobs['b']}/kill", headers=H).json()["status"] == "killed"
    st = status(client, b)
    assert st["status"] == "killed" and st["stop_reason"] == "killed"
    assert client.post(f"/admin/jobs/{jobs['b']}/kill", headers=H).status_code == 409  # nothing left to kill
    # redo puts it back at the end, from scratch
    assert client.post(f"/admin/jobs/{jobs['b']}/redo", headers=H).json()["status"] == "queued"
    assert status(client, b)["status"] == "queued" and status(client, b)["stop_reason"] is None
    client.post("/admin/demo-mode", json={"enabled": False}, headers=H)
    wait_for(lambda: status(client, b)["status"] == "done", msg="redone job done")
    # redo a finished job: it trains again from zero
    assert client.post(f"/admin/jobs/{jobs['b']}/redo", headers=H).json()["status"] == "queued"
    done = wait_for(lambda: (s := status(client, b))["status"] == "done" and s, msg="second run")
    assert done["progress"]["preemptions"] == 0


@pytest.mark.budget("5e11")
def test_kill_and_redo_running_job(client, env):
    c = client
    if True:
        sid = submit(c, config={**SMALL, "batch_size": 32}).json()["submission_id"]
        jid = c.get("/admin/queue", headers=H).json()["jobs"][0]["id"]
        wait_for(lambda: status(c, sid)["status"] == "running" and c.app.state.store.latest_checkpoint(jid), msg="running with checkpoint")
        assert c.delete(f"/admin/jobs/{jid}", headers=H).status_code == 409  # running: must be killed first
        # redo while running: stops, wipes checkpoints/metrics, queues again from scratch
        sched = c.app.state.scheduler
        old = sched.current
        assert c.post(f"/admin/jobs/{jid}/redo", headers=H).json()["status"] == "redoing"
        wait_for(lambda: sched.current is not None and sched.current is not old, msg="job restarted from scratch")
        assert c.app.state.store.latest_checkpoint(jid) is None or c.app.state.store.latest_checkpoint(jid)["step"] < 300  # fresh run
        # kill
        assert c.post(f"/admin/jobs/{jid}/kill", headers=H).json()["status"] == "killing"
        st = wait_for(lambda: (s := status(c, sid))["status"] == "killed" and s, msg="killed")
        assert st["stop_reason"] == "killed"
        assert not (env / "models" / sid).exists()


def test_reset_snapshots_first_and_keeps_house_net(client, env):
    sid = submit(client).json()["submission_id"]
    wait_for(lambda: status(client, sid)["status"] == "done", msg="job done")
    (env / "models" / "house-net").mkdir(parents=True)
    (env / "models" / "house-net" / "config.json").write_text("{}")
    assert client.post("/admin/reset", json={"confirm": "nope"}, headers=H).status_code == 400
    r = client.post("/admin/reset", json={"confirm": "RESET"}, headers=H).json()
    snap = sqlite3.connect(r["snapshot"])
    assert snap.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 1  # the snapshot still has the run
    c = db.connect(env / "demo.db")
    assert all(c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0 for t in ("submissions", "jobs", "job_metrics", "races", "checkpoints"))
    assert c.execute("SELECT COUNT(*) FROM admin_events WHERE kind='reset'").fetchone()[0] == 1
    assert [p.name for p in (env / "models").iterdir()] == ["house-net"]
    assert client.get("/admin/queue", headers=H).json()["jobs"] == []
    assert submit(client).status_code == 200  # the system works after a reset


def test_export_excludes_contact_unless_asked(client):
    sid = submit(client, contact="alice@example.com").json()["submission_id"]
    wait_for(lambda: status(client, sid)["status"] == "done", msg="job done")
    j = client.get("/admin/export", headers=H).json()
    assert "contact" not in j["submissions"][0] and j["submissions"][0]["final_val_loss"] is not None
    assert j["submissions"][0]["config"]["layers"] == 2
    j2 = client.get("/admin/export?include_contact=true", headers=H).json()
    assert j2["submissions"][0]["contact"] == "alice@example.com"
    csv_text = client.get("/admin/export?format=csv", headers=H).text
    assert csv_text.splitlines()[0].startswith("id,") and "alice@example.com" not in csv_text and "alice" in csv_text


def test_logs_stream_and_presenter(client):
    submit(client)
    logs = client.get("/admin/logs?tail=50", headers=H).json()["lines"]
    assert isinstance(logs, list)
    with client.stream("GET", "/admin/stream?limit=1", headers=H) as r:
        lines = r.iter_lines()
        first, data = next(lines), next(lines)
        assert first == "event: status" and "gpu" in json.loads(data[len("data: "):])
    assert client.get("/admin/stream").status_code == 401
    assert client.get("/api/presenter").json()["stage_id"] == ""
    client.put("/api/presenter", json={"stage_id": "a1-spirals", "stage_index": 3, "title": "Spirals"})
    got = client.get("/api/presenter").json()
    assert got["stage_id"] == "a1-spirals" and got["updated_at"] > 0
    assert client.get("/api/leaderboard").json() == []
