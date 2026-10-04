"""Part (a) of the training-speed work: the optimised loop trains EXACTLY like the original one, the FLOPs budget is untouched, and the
helpers that remove per-step GPU syncs behave. (CUDA-only effects, i.e. the actual speed-up, cannot be measured here: see FINAL_REPORT.)"""

import numpy as np
import pytest
import torch

from app.chess_net import encoder as E
from app.chess_net.config import flops_per_sample, resolve_config
from app.data.loaders import LichessData
from app.training.chess_data import ChessData
from app.training.hostio import HostStager
from app.training.trainer import CompetitionTrainer

BUDGET = 3e9


@pytest.fixture(scope="module")
def data(fake_data_dir):
    return ChessData(LichessData(fake_data_dir / "lichess"), "cpu", val_rows=200)


def cfg_of(**kw):
    c, e = resolve_config({"layers": 2, "width": 32, "batch_size": 32, **kw})
    assert not e, e
    return c


def train(data, cfg, optimized, budget=BUDGET, **kw):
    t = CompetitionTrainer(cfg, data, "cpu", budget_flops=budget, time_cap_s=600, ckpt_every_s=1e9, metrics_every_s=1e9, optimized=optimized, **kw)
    reason = t.run(lambda: False)
    return t, reason


@pytest.mark.parametrize("raw", [{}, {"color_flip_augmentation": True, "ema": {"enabled": True, "decay": 0.99}, "optimizer": "adam", "lr": 0.003},
                                 {"sampling": "weighted", "grad_clip": 1.0, "input_extras": {"attacks": True, "material": True, "en_passant": True}}])
def test_optimised_loop_trains_identically_to_the_original_loop(data, raw):
    cfg = cfg_of(**raw)
    a, ra = train(data, cfg, optimized=False)
    b, rb = train(data, cfg, optimized=True)
    assert ra == rb == "budget"
    assert (a.step, a.samples_seen, a.flops_used) == (b.step, b.samples_seen, b.flops_used)
    for (n, p), q in zip(a.model.state_dict().items(), b.model.state_dict().values()):
        assert torch.equal(p, q), n  # bit-identical weights: same seed, same data order, same arithmetic
    assert a.last_train_loss == b.last_train_loss
    assert a.evaluate() == b.evaluate()


def test_flops_budget_accounting_is_unchanged(data):
    cfg = cfg_of()
    t, reason = train(data, cfg, optimized=True)
    fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
    assert reason == "budget" and t.flops_used == fps * t.samples_seen == fps * t.step * 32
    assert BUDGET <= t.flops_used < BUDGET + fps * 32 + 1  # overshoots by at most one step, exactly as before
    assert t.samples_seen == t.step * cfg["batch_size"]  # the participant's batch size is never touched


def test_curves_are_identical_and_checkpoint_resume_stays_exact(data, tmp_path, monkeypatch):
    import app.training.trainer as T
    monkeypatch.setattr(T, "CHECK_MIN", 25); monkeypatch.setattr(T, "CHECK_MAX", 25)  # same sync points as the original loop (every 25 steps)
    cfg = cfg_of(color_flip_augmentation=True)
    kw = dict(budget_flops=BUDGET, time_cap_s=600, ckpt_every_s=1e9, metrics_every_s=0.0)  # a metrics record at every sync point
    runs = []
    for opt in (False, True):
        t = CompetitionTrainer(cfg, data, "cpu", optimized=opt, **kw)
        t.run(lambda: False)
        runs.append(t)
    key = lambda r: (r["step"], r["samples_seen"], r["flops"], r["train_loss"], r["val_loss"], r["val_mse"])
    assert len(runs[0].curves) > 3 and [key(r) for r in runs[0].curves] == [key(r) for r in runs[1].curves]  # the same training curves, value for value
    # resume: stop half way, save, continue in a fresh trainer == one uninterrupted run
    full = CompetitionTrainer(cfg, data, "cpu", **kw)
    full.run(lambda: False)
    first = CompetitionTrainer(cfg, data, "cpu", **kw)
    stop = {"n": 0}

    def abort():
        stop["n"] += 1
        return stop["n"] > 40
    assert first.run(abort) == "aborted"
    first.save_checkpoint(tmp_path / "c.pt")
    second = CompetitionTrainer(cfg, data, "cpu", **kw)
    second.load_checkpoint(tmp_path / "c.pt")
    assert second.run(lambda: False) == "budget" and second.step == full.step and second.samples_seen == full.samples_seen
    for p, q in zip(full.model.state_dict().values(), second.model.state_dict().values()):
        assert torch.equal(p, q)


def test_sync_interval_adapts_but_stays_in_bounds(data, monkeypatch):
    """The fast loop reads the GPU clock/loss at most every CHECK_MAX steps and at least every CHECK_MIN steps."""
    import app.training.trainer as T
    cfg = cfg_of()
    seen = []
    orig = CompetitionTrainer.evaluate
    t = CompetitionTrainer(cfg, data, "cpu", budget_flops=BUDGET, time_cap_s=600, ckpt_every_s=1e9, metrics_every_s=0.0, optimized=True)
    t.run(lambda: False, on_metrics=lambda rec: seen.append(rec["step"]))
    gaps = np.diff([0] + seen)
    assert gaps.min() >= T.CHECK_MIN and gaps.max() <= T.CHECK_MAX


def test_host_stager_ring_never_corrupts_data_and_reuses_buffers():
    st = HostStager("cpu", force_ring=True)
    outs = [st.upload(torch.arange(8, dtype=torch.int64) + 100 * k) for k in range(HostStager.RING * 3)]
    for k, o in enumerate(outs):  # every earlier upload is still intact after the ring wrapped around several times
        assert torch.equal(o, torch.arange(8, dtype=torch.int64) + 100 * k)
    assert len(st._rings) == 1 and len(next(iter(st._rings.values()))) == HostStager.RING
    st.upload(torch.zeros(3, dtype=torch.bool))
    assert len(st._rings) == 2  # one ring per (shape, dtype)
    assert HostStager("cpu").upload(torch.ones(2)).device.type == "cpu" and not HostStager("cpu").active


def test_attack_maps_avoid_boolean_mask_indexing():
    """`t[t >= 0]` on a CUDA tensor is a device->CPU sync. The valid-index tables are computed once and are plain index tensors."""
    tables = E._tables(torch.device("cpu"))
    rays, knight, king, pawn, v_rays, v_knight, v_king, v_pawn = tables
    for d in range(8):
        for k in range(7):
            v = v_rays[d][k]
            assert v.dtype == torch.long and v.dim() == 1 and bool((rays[d, k][v] >= 0).all()) and len(v) == int((rays[d, k] >= 0).sum())
    assert all(len(v) == int((knight[i] >= 0).sum()) for i, v in enumerate(v_knight))


def test_trace_of_a_training_step_has_no_item_or_nonzero_calls(data, monkeypatch):
    """Count the host-sync primitives a step triggers (Tensor.item / nonzero / tolist): the fast step needs none between sync points."""
    cfg = cfg_of(input_extras={"attacks": True}, color_flip_augmentation=True)
    t = CompetitionTrainer(cfg, data, "cpu", budget_flops=BUDGET, time_cap_s=600, ckpt_every_s=1e9, metrics_every_s=1e9)
    import sys
    calls = []

    def spy(name, orig):
        def f(self, *a, **k):
            # torch.optim reads Adam's step counter with .item(); that tensor lives on the CPU (no device sync), so it does not count
            if "/torch/optim/" not in sys._getframe(1).f_code.co_filename:
                calls.append(name)
            return orig(self, *a, **k)
        return f
    for name in ("item", "tolist", "nonzero"):
        monkeypatch.setattr(torch.Tensor, name, spy(name, getattr(torch.Tensor, name)))
    for _ in range(3):
        t._train_step()
    assert calls == [], calls
