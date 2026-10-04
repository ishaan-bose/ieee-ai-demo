import time

import numpy as np
import pytest
import torch

from app.training.race import DoodleData, Lane, resolve_lanes, run_race


def make_data(n=3000, device="cpu", resident=True, seed=0):
    """Learnable synthetic 'doodles': each class lights a different 7x7 block, plus noise."""
    rng = np.random.default_rng(seed)

    def gen(m):
        y = rng.integers(0, 10, m).astype(np.int8)
        x = rng.integers(0, 40, (m, 28, 28)).astype(np.uint8)
        for i, c in enumerate(y):
            r, cc = divmod(int(c), 4)
            x[i, r * 7:(r + 1) * 7, cc * 7:(cc + 1) * 7] += 150
        return x, y
    tx, ty = gen(n)
    vx, vy = gen(500)
    px, py = gen(16)
    return DoodleData(tx, ty, vx, vy, px, py, device, force_resident=resident)


def lanes_of(data, *cfgs):
    ls, e = resolve_lanes(list(cfgs))
    assert not e, e
    return [Lane(c, data, seed=0) for c in ls]


def run(lanes, secs, abort=lambda: False):
    ev = []
    reason = run_race(lanes, secs, lambda k, d: ev.append((k, d)), abort, tick_s=0.2, slice_s=0.05)
    return reason, ev


@pytest.mark.parametrize("resident", [True, False])
def test_three_lane_race_events_and_learning(resident):
    data = make_data(resident=resident)
    lanes = lanes_of(data, {"id": "ce", "loss": "ce"}, {"id": "mse", "loss": "mse", "lr": 0.05}, {"id": "focal", "loss": "focal", "gamma": 2})
    reason, ev = run(lanes, 1.5)
    assert reason == "time"
    ticks = [d for k, d in ev if k == "tick"]
    assert len(ticks) >= 4 and ev[-1][0] == "done" and ev[-1][1]["reason"] == "time"
    for t in ticks:
        assert set(t) == {"t", "lanes"} and [l["id"] for l in t["lanes"]] == ["ce", "mse", "focal"]
        for l in t["lanes"]:
            assert set(l) == {"id", "step", "samples_seen", "updates", "loss", "acc", "probe_preds"} and len(l["probe_preds"]) == 16
    assert [t["t"] for t in ticks] == sorted(t["t"] for t in ticks)
    final = ev[-1][1]["final"]
    assert final[0]["acc"] > 0.5  # cross-entropy clearly learned the signal
    assert all(f["updates"] > 0 and f["samples_seen"] == f["updates"] * 64 for f in final)


def test_equal_wall_clock_fairness():
    data = make_data()
    lanes = lanes_of(data, {"id": "a", "batch_size": 32}, {"id": "b", "batch_size": 32}, {"id": "c", "batch_size": 32})
    run(lanes, 1.2)
    u = [l.updates for l in lanes]
    assert max(u) / min(u) < 1.6, u  # identical lanes got near-identical time


def test_batch_one_and_full_batch_lanes():
    data = make_data(n=2500)
    lanes = lanes_of(data, {"id": "one", "batch_size": 1, "lr": 0.005}, {"id": "full", "batch_size": "full", "lr": 0.5})
    lanes[1].chunk = 700  # force several accumulation chunks per update
    run(lanes, 1.5)
    one, full = lanes
    assert one.samples_seen == one.updates and one.updates > 100
    assert full.updates >= 1 and full.samples_seen == full.updates * 2500


def test_full_batch_accumulation_equals_true_full_batch():
    data = make_data(n=1000)
    a = lanes_of(data, {"id": "a", "batch_size": "full", "lr": 0.1, "optimizer": "sgd", "max_updates": 3})[0]
    b = lanes_of(data, {"id": "b", "batch_size": "full", "lr": 0.1, "optimizer": "sgd", "max_updates": 3})[0]
    a.chunk, b.chunk = 128, 1000  # chunked vs one shot
    for l in (a, b):
        l.run_slice(time.perf_counter() + 30)
    for pa, pb in zip(a.model.parameters(), b.model.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5)


def test_complete_and_abort():
    data = make_data()
    reason, ev = run(lanes_of(data, {"id": "a", "max_updates": 5}, {"id": "b", "max_updates": 8}), 30)
    assert reason == "complete" and ev[-1][1]["final"][1]["updates"] == 8
    t0 = time.perf_counter()
    flag = {"v": False}
    import threading
    threading.Timer(0.4, lambda: flag.update(v=True)).start()
    reason, ev = run(lanes_of(data, {"id": "a"}), 30, abort=lambda: flag["v"])
    assert reason == "aborted" and time.perf_counter() - t0 < 3 and ev[-1][1]["reason"] == "aborted"


def test_all_losses_and_lane_validation():
    data = make_data()
    for loss in ("mse", "ce", "l1", "huber", "focal", "label_smooth"):
        lane = lanes_of(data, {"id": loss, "loss": loss})[0]
        lane.run_slice(time.perf_counter() + 0.1)
        assert lane.updates > 0 and np.isfinite(lane.snapshot()["loss"])
    bad = [({"loss": "nope"}, "loss"), ({"lr": 0}, "lr"), ({"batch_size": 0}, "batch_size"), ({"hidden": []}, "hidden"),
           ({"batch_size": "huge"}, "batch_size")]
    for raw, key in bad:
        lanes, errs = resolve_lanes([raw])
        assert lanes is None and any(key in e for e in errs), (raw, errs)
    assert resolve_lanes([{}] * 4)[0] is None
    assert resolve_lanes([{"id": "x"}, {"id": "x"}])[0] is None
