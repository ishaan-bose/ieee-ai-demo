"""Part (b)/(c) of the training-speed work: several competition jobs at once, each in its own process (CPU, fake data, no GPU needed).

Every test here starts a real scheduler that spawns real child processes. What can NOT be shown on this machine (no GPU): that three processes really
fit and speed things up on an L40S, that `torch.cuda.mem_get_info()` agrees with the estimate, and a real CUDA out-of-memory error (it is injected)."""

import json
import logging
import os
import signal
import time

import pytest

from app.scheduler import vram
from app.store import Store
from tests.conftest import requires_torch
from tests.test_scheduler_api import H, SMALL, client, env, race_body, read_sse, server_data, status, submit, wait_for  # noqa: F401  (fixtures)

pytestmark = requires_torch
BATCH = {**SMALL, "batch_size": 32}


def jobs_of(c):
    return {j["nickname"]: j for j in c.get("/admin/queue", headers=H).json()["jobs"]}


def n_running(c):
    return sum(1 for j in c.get("/admin/queue", headers=H).json()["jobs"] if j["status"] == "running")


def started_order(store):
    return [j["id"] for j in sorted((j for j in store.list_jobs(True) if j["started_at"]), key=lambda j: j["started_at"])]


# ---------------------------------------------------------------- FCFS at every concurrency
@pytest.mark.parametrize("n", [1, 2, 3])
def test_fifo_order_and_the_concurrency_ceiling(request, monkeypatch, env, server_data, n):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("MAX_CONCURRENT_JOBS", str(n))
    monkeypatch.setenv("BUDGET_FLOPS", "2.5e10")
    with TestClient(app) as c:
        ids = []
        for k in range(4):
            sid = submit(c, name=f"p{k}", config=BATCH).json()["submission_id"]
            ids.append(sid)
        seen_max = 0
        t0 = time.time()
        while time.time() - t0 < 240:
            seen_max = max(seen_max, n_running(c))
            assert seen_max <= n  # never more than the ceiling
            if all(status(c, s)["status"] == "done" for s in ids):
                break
            time.sleep(0.05)
        else:
            raise AssertionError("jobs did not finish")
        assert seen_max == n  # and the ceiling is actually used when there is work
        store = c.app.state.store
        jobs = [store.submission_job(s) for s in ids]
        assert started_order(store) == [j["id"] for j in jobs]  # strict FCFS: jobs START in the order they were submitted
        for s in ids:
            meta = json.loads((env / "models" / s / "meta.json").read_text())
            assert meta["stop_reason"] == "budget" and meta["samples_seen"] == meta["steps"] * 32  # the participant's batch size and budget are untouched
            assert 2.5e10 <= meta["flops_used"] < 2.5e10 + 6 * (773 * 32 + 32 * 32 + 32) * 32 + 1
        # (c) per finished job: samples/s, concurrency it ran under, stop reason, visible in /admin
        wait_for(lambda: all(j["samples_per_s"] for j in jobs_of(c).values()), msg="stats recorded")  # (written by the parent right after it reaps the process)
        q = jobs_of(c)
        for j in q.values():
            assert j["stop_reason"] == "budget" and j["samples_per_s"] > 0 and 1 <= j["concurrency"] <= n
        assert max(j["concurrency"] for j in q.values()) == n


# ---------------------------------------------------------------- preemption with several jobs
@pytest.mark.jobs(3)
@pytest.mark.budget("2e10")
def test_a_demo_race_kills_all_running_jobs_which_resume_in_their_original_order(client, env, caplog):
    caplog.set_level(logging.INFO, logger="scheduler")
    c = client
    store = c.app.state.store
    sids = [submit(c, name=f"p{k}", config=BATCH).json()["submission_id"] for k in range(4)]  # 3 run, the 4th waits behind them
    jid = {s: store.submission_job(s)["id"] for s in sids}
    wait_for(lambda: all(store.latest_checkpoint(jid[s]) for s in sids[:3]) and n_running(c) == 3, msg="three running jobs with checkpoints")
    assert status(c, sids[3])["status"] == "queued"
    before = {s: store.latest_checkpoint(jid[s]) for s in sids[:3]}
    rid = c.post("/api/demo/races", json=race_body(1.0)).json()["race_id"]
    wait_for(lambda: all(store.get_job(jid[s])["status"] == "queued" for s in sids[:3]), timeout=20, msg="all three preempted")
    jobs = {s: store.get_job(jid[s]) for s in sids}
    assert all(jobs[s]["preemptions"] == 1 for s in sids[:3]) and jobs[sids[3]]["preemptions"] == 0
    orders = [jobs[s]["queue_order"] for s in sids]
    assert orders == sorted(orders) and len(set(orders)) == 4  # original order kept, and all three still ahead of the job that was waiting
    assert [j["id"] for j in store.list_jobs() if j["status"] == "queued" and j["kind"] == "competition"] == [jid[s] for s in sids]
    for s in sids[:3]:  # counters were restored from the checkpoint: nothing double counted
        assert jobs[s]["samples_seen"] == before[s]["samples_seen"] or jobs[s]["samples_seen"] >= before[s]["samples_seen"] - 1
    ev = read_sse(c, f"/api/demo/races/{rid}/stream")
    assert ev[-1][0] == "done"
    for s in sids:
        wait_for(lambda s=s: status(c, s)["status"] == "done", timeout=180, msg="job finished")
    starts = [int(r.getMessage().split()[1]) for r in caplog.records if "started in process" in r.getMessage()]
    assert starts[:3] == [jid[s] for s in sids[:3]] and starts[3:6] == [jid[s] for s in sids[:3]]  # initial start, then the resume: same order
    for s in sids[:3]:
        meta = json.loads((env / "models" / s / "meta.json").read_text())
        assert meta["preemptions"] == 1 and meta["samples_seen"] == meta["steps"] * 32  # no double-counted samples
        assert meta["flops_used"] == pytest.approx(6 * (773 * 32 + 32 * 32 + 32) * meta["samples_seen"], rel=1e-9)
        assert 2e10 <= meta["flops_used"] < 2e10 + 6 * (773 * 32 + 32 * 32 + 32) * 32 + 1


@pytest.mark.jobs(3)
def test_demo_mode_and_pause_hold_every_concurrent_job(client, env):
    c, store = client, client.app.state.store
    c.app.state.scheduler.s  # (settings are read at start-up)
    sids = [submit(c, name=f"p{k}", config=BATCH).json()["submission_id"] for k in range(3)]
    wait_for(lambda: n_running(c) >= 1, msg="running")
    c.post("/admin/queue/pause", headers=H)
    wait_for(lambda: n_running(c) == 0 and all(store.submission_job(s)["status"] in ("queued", "done") for s in sids), timeout=20, msg="paused")
    time.sleep(0.6)
    assert n_running(c) == 0
    c.post("/admin/queue/resume", headers=H)
    for s in sids:
        wait_for(lambda s=s: status(c, s)["status"] == "done", timeout=180, msg="done after resume")


# ---------------------------------------------------------------- out of memory
@pytest.mark.jobs(3)
def test_an_out_of_memory_error_requeues_the_job_and_lowers_the_concurrency(client, env, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="scheduler")
    c, store, sched = client, client.app.state.store, client.app.state.scheduler
    sched.oom_cooldown_s = 1.5
    monkeypatch.setenv("BYOAI_INJECT_OOM", "1")  # (inherited by the children spawned from now on) job 1 raises one CUDA OOM at its first start
    a = submit(c, name="a", config=BATCH).json()["submission_id"]
    b = submit(c, name="b", config=BATCH).json()["submission_id"]
    wait_for(lambda: any("ran out of GPU memory" in r.getMessage() for r in caplog.records), timeout=30, msg="the OOM was handled")
    assert sched._oom_penalty == 1 and sched.effective_limit() == 2  # one slot withheld
    assert (env / "oom_injected_1").exists()
    # the job was not lost and the server did not crash: both jobs still finish, a from the FRONT of the queue, with its counters intact
    for s in (a, b):
        wait_for(lambda s=s: status(c, s)["status"] == "done", timeout=120, msg="job finished after the OOM")
    assert store.submission_job(a)["preemptions"] == 0  # an OOM is not a preemption
    wait_for(lambda: sched.effective_limit() == 3, timeout=10, msg="concurrency climbs back after the cool-down")
    assert any("concurrency lowered to 2" in r.getMessage() for r in caplog.records)
    assert c.get("/api/health").status_code == 200


# ---------------------------------------------------------------- admission by VRAM
def test_vram_estimate_and_admission_rule():
    from app.chess_net.config import resolve_config

    small, _ = resolve_config({"layers": 2, "width": 32})
    big, _ = resolve_config({"layers": 7, "width": 2048, "batch_size": 4096})
    a, b = vram.estimate_job_vram(small, 1 << 30), vram.estimate_job_vram(big, 1 << 30)
    assert 1.5 * 2**30 < a < 4 * 2**30 and b > a  # a few GB for a small job (nvidia-smi showed ~3 GB), more for a 27M-parameter net
    GB = 2**30
    assert vram.admit(5 * GB, 40 * GB, 48 * GB, 10 * GB)
    assert not vram.admit(5 * GB, 4 * GB, 48 * GB, 0)  # not enough free right now
    assert not vram.admit(20 * GB, 45 * GB, 48 * GB, 30 * GB)  # free now, but the running jobs have not allocated what they were promised yet


@pytest.mark.jobs(3)
@pytest.mark.budget("6e10")
def test_a_job_that_does_not_fit_waits_and_fewer_jobs_run(client, env):
    c, store, sched = client, client.app.state.store, client.app.state.scheduler
    GB = 2**30
    sched.mem_info = lambda: (5 * GB - int(2.5 * GB) * len(sched.running), 48 * GB)  # 5 GB free, and every running job takes ~2.5 GB: a second one does not fit
    sids = [submit(c, name=f"p{k}", config=BATCH).json()["submission_id"] for k in range(3)]
    seen = 0
    t0 = time.time()
    while time.time() - t0 < 120 and not all(status(c, s)["status"] == "done" for s in sids):
        seen = max(seen, n_running(c))
        time.sleep(0.05)
    assert seen == 1 and started_order(store) == [store.submission_job(s)["id"] for s in sids]  # started fewer jobs, still strictly in order
    sched.mem_info = lambda: (40 * GB - int(2.5 * GB) * len(sched.running), 48 * GB)
    sids = [submit(c, name=f"q{k}", config=BATCH).json()["submission_id"] for k in range(3)]
    seen = 0
    t0 = time.time()
    while time.time() - t0 < 120 and not all(status(c, s)["status"] == "done" for s in sids):
        seen = max(seen, n_running(c))
        time.sleep(0.05)
    assert seen == 3  # plenty of memory: all three at once


# ---------------------------------------------------------------- admin actions with several running jobs
@pytest.mark.jobs(3)
@pytest.mark.budget("2e12")
def test_admin_kill_redo_remove_with_concurrent_jobs(client, env):
    c, store = client, client.app.state.store
    sids = {k: submit(c, name=k, config=BATCH).json()["submission_id"] for k in "abc"}
    jid = {k: store.submission_job(s)["id"] for k, s in sids.items()}
    wait_for(lambda: n_running(c) == 3 and all(store.latest_checkpoint(jid[k]) for k in "abc"), msg="three running")
    assert sorted(c.get("/admin/queue", headers=H).json()["running_job_ids"]) == sorted(jid.values())
    assert c.delete(f"/admin/jobs/{jid['a']}", headers=H).status_code == 409  # remove: not while running
    # kill b: only b stops
    assert c.post(f"/admin/jobs/{jid['b']}/kill", headers=H).json()["status"] == "killing"
    st = wait_for(lambda: (s := status(c, sids["b"]))["status"] == "killed" and s, msg="b killed")
    assert st["stop_reason"] == "killed" and not (env / "models" / sids["b"]).exists()
    assert status(c, sids["a"])["status"] == "running" and status(c, sids["c"])["status"] == "running"
    # redo c: it restarts from scratch (its checkpoints and metrics are wiped), a keeps going
    old_ck = store.latest_checkpoint(jid["c"])
    assert c.post(f"/admin/jobs/{jid['c']}/redo", headers=H).json()["status"] == "redoing"
    wait_for(lambda: (j := store.get_job(jid["c"]))["status"] == "running" and (store.latest_checkpoint(jid["c"]) is None or store.latest_checkpoint(jid["c"])["id"] != old_ck["id"]),
             timeout=30, msg="c restarted")
    assert store.get_job(jid["c"])["preemptions"] == 0 and status(c, sids["a"])["status"] == "running"
    # remove a queued job while the others run: d takes the slot b freed, e has to wait
    submit(c, name="d", config=BATCH)
    e = submit(c, name="e", config=BATCH).json()["submission_id"]
    wait_for(lambda: n_running(c) == 3, msg="d took the free slot")
    assert status(c, e)["status"] == "queued"
    assert c.delete(f"/admin/jobs/{store.submission_job(e)['id']}", headers=H).json()["status"] == "removed"
    assert n_running(c) == 3
    # pause: every job leaves the GPU; resume: they come back
    c.post("/admin/queue/pause", headers=H)
    wait_for(lambda: n_running(c) == 0, timeout=20, msg="paused")
    c.post("/admin/queue/resume", headers=H)
    wait_for(lambda: n_running(c) >= 1, timeout=20, msg="resumed")


@pytest.mark.jobs(3)
@pytest.mark.budget("2e12")
def test_nuclear_reset_with_three_running_jobs(client, env):
    c, store = client, client.app.state.store
    for k in "abc":
        submit(c, name=k, config=BATCH)
    wait_for(lambda: n_running(c) == 3, msg="three running")
    r = c.post("/admin/reset", json={"confirm": "RESET"}, headers=H)
    assert r.status_code == 200 and r.json()["ok"]
    assert c.app.state.scheduler.wait_idle(5) and store.list_jobs(True) == []
    time.sleep(0.5)
    assert n_running(c) == 0 and c.app.state.scheduler.running_jobs() == []


# ---------------------------------------------------------------- robustness
@pytest.mark.jobs(2)
@pytest.mark.budget("2e12")
def test_a_job_process_that_dies_is_restarted_from_its_checkpoint(client, env):
    c, store, sched = client, client.app.state.store, client.app.state.scheduler
    sid = submit(c, name="a", config=BATCH).json()["submission_id"]
    jid = store.submission_job(sid)["id"]
    wait_for(lambda: store.latest_checkpoint(jid), msg="checkpoint")
    first_pid = sched.running_jobs()[0].proc.pid
    os.kill(first_pid, signal.SIGKILL)  # a segfault / CUDA fault / OOM-killer
    wait_for(lambda: sched.running_jobs() and sched.running_jobs()[0].proc.pid != first_pid, timeout=30, msg="restarted in a new process")
    assert status(c, sid)["status"] == "running" and store.latest_checkpoint(jid)


@pytest.mark.jobs(2)
@pytest.mark.timecap("0.6")
@pytest.mark.budget("2e12")
def test_stop_reason_time_logs_a_concurrency_warning_and_shows_in_admin(client, env, caplog):
    caplog.set_level(logging.INFO, logger="scheduler")
    c = client
    sid = submit(c, name="slow", config=BATCH).json()["submission_id"]
    st = wait_for(lambda: (s := status(c, sid))["status"] == "done" and s, timeout=60, msg="finished by the time cap")
    assert st["stop_reason"] == "time"
    wait_for(lambda: any("may be too high" in r.getMessage() for r in caplog.records), msg="the warning")  # (logged by the parent right after it reaps the process)
    warn = [r for r in caplog.records if r.levelno == logging.WARNING and "may be too high" in r.getMessage()]
    assert warn and "MAX_CONCURRENT_JOBS" in warn[0].getMessage() and "stop reason" in warn[0].getMessage()
    wait_for(lambda: jobs_of(c)["slow"]["samples_per_s"], msg="stats")
    j = jobs_of(c)["slow"]
    assert j["stop_reason"] == "time" and j["samples_per_s"] > 0 and j["concurrency"] >= 1
    assert any("finished: stop reason time" in r.getMessage() and "samples/s" in r.getMessage() for r in caplog.records)


def test_recovery_after_power_off_keeps_the_original_order_of_several_running_jobs(tmp_path):
    from app import db

    db.init_db(tmp_path / "d.db")
    st = Store(tmp_path / "d.db")
    cfg = {"param_count": 1, "tier": "Light"}
    subs = [st.create_submission(nickname=n, model_name=n, contact=None, consent=True, config=cfg) for n in "abcde"]
    for k in (3, 0, 2):  # d, a, c were all training when the box lost power (marked running in an arbitrary order)
        st.mark_running(subs[k]["job_id"])
    rec = st.recover_after_restart()
    assert rec["requeued"] == [subs[0]["job_id"], subs[2]["job_id"], subs[3]["job_id"]]
    queued = [j["nickname"] for j in st.list_jobs() if j["status"] == "queued"]
    assert queued == ["a", "c", "d", "b", "e"]  # the interrupted ones first, in the order they had; then the rest
