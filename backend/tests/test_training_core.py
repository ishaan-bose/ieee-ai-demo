import copy
import threading

import numpy as np
import pytest
import torch

from app.chess_net import encoder as E
from app.chess_net.config import DEFAULT_CONFIG, count_params, flops_per_sample, resolve_config
from app.chess_net.model import build_model
from app.chess_net.model_io import load_model, list_models
from app.data.loaders import LichessData
from app.training.chess_data import ChessData
from app.training.losses import class_loss, regression_loss
from app.training.optim import EMA, lr_factor
from app.training.trainer import CompetitionTrainer

pytestmark = pytest.mark.skipif(not hasattr(torch, "nn"), reason="torch missing")


@pytest.fixture(scope="module")
def chess_data(fake_data_dir):
    return ChessData(LichessData(fake_data_dir / "lichess"), "cpu", val_rows=200)


def cfg_of(**kw):
    raw = {"layers": 2, "width": 32, **kw}
    r, e = resolve_config(raw)
    assert not e, e
    return r


def test_param_count_matches_torch_for_many_configs():
    for raw in ({}, {"normalization": "layernorm", "residual": True, "layers": 4, "width": 24},
                {"layer_widths": [40, 24, 16], "input_extras": {"attacks": True, "material": True, "en_passant": True}},
                {"input_extras": {"stm_castle": False}}):
        r, e = resolve_config(raw)
        assert not e and sum(p.numel() for p in build_model(r).parameters()) == r["param_count"]


def test_default_config_is_resolved_and_light():
    r, e = resolve_config(copy.deepcopy(DEFAULT_CONFIG))
    assert not e and r["tier"] == "Light" and r["param_count"] < 1_500_000


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "sigmoid", "tanh", "gelu", "swish", "softplus", "elu", "clipped_relu", "hard_step"])
@pytest.mark.parametrize("init", ["xavier", "he", "small_normal"])
def test_every_activation_and_init_builds_and_runs(act, init):
    r = cfg_of(activation={"name": act}, init=init, normalization="layernorm", residual=True, layers=3)
    out = build_model(r)(torch.randn(5, r["in_dim"]))
    assert out.shape == (5,) and torch.isfinite(out).all()


def test_losses():
    p, y = torch.rand(10), torch.rand(10)
    for n in ("mse", "huber", "bce", "l1"):
        assert regression_loss(n, p, y, 0.5).ndim == 0
    z, t = torch.randn(8, 10), torch.randint(0, 10, (8,))
    vals = {n: class_loss(n, z, t, delta=1, gamma=2, eps=0.1).item() for n in ("mse", "ce", "l1", "huber", "focal", "label_smooth")}
    assert all(np.isfinite(v) and v > 0 for v in vals.values())
    assert class_loss("focal", z, t, gamma=0.0).item() == pytest.approx(class_loss("ce", z, t).item(), rel=1e-5)


def test_schedule_and_ema():
    assert lr_factor("constant", 0.5) == 1
    assert lr_factor("cosine", 0.0) == pytest.approx(1) and lr_factor("cosine", 1.0) == pytest.approx(0, abs=1e-9)
    assert lr_factor("linear", 0.5) == pytest.approx(0.5)
    assert lr_factor("cosine", 0.05, warmup_frac=0.1) == pytest.approx(0.5)  # halfway through warmup
    assert lr_factor("linear", 0.1, warmup_frac=0.1) == pytest.approx(1.0)  # warmup just ended
    p = [torch.nn.Parameter(torch.zeros(3))]
    ema = EMA(p, 0.9)
    with torch.no_grad():
        p[0].fill_(1.0)
    for _ in range(100):
        ema.update(p)
    assert torch.allclose(ema.shadow[0], torch.ones(3), atol=1e-3)
    with torch.no_grad():
        p[0].fill_(5.0)
    with ema.applied(p):
        assert torch.allclose(p[0], ema.shadow[0])
    assert torch.allclose(p[0].detach(), torch.full((3,), 5.0))


def test_budget_stop_and_flops_accounting(chess_data):
    cfg = cfg_of(batch_size=32)
    fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
    tr = CompetitionTrainer(cfg, chess_data, budget_flops=fps * 32 * 20, time_cap_s=60, metrics_every_s=0.0)
    assert tr.run(lambda: False) == "budget"
    assert tr.step == 20 and tr.samples_seen == 640 and tr.flops_used == fps * 640
    assert fps == 6 * cfg["matmul_params"]


def test_time_cap_and_divergence(chess_data):
    tr = CompetitionTrainer(cfg_of(batch_size=8), chess_data, budget_flops=1e18, time_cap_s=0.3)
    assert tr.run(lambda: False) == "time" and tr.active_s >= 0.3
    bad = CompetitionTrainer(cfg_of(lr=1.0, optimizer="sgd", activation={"name": "swish", "beta": 10}, loss={"name": "mse"},
                                    target_type="cp", init="he", batch_size=64), chess_data, budget_flops=1e18, time_cap_s=30)
    # huge learning rate on unscaled-by-design nets blows up
    reason = bad.run(lambda: False)
    assert reason in ("diverged", "time", "budget")


def test_abort_is_fast_and_saves_nothing(chess_data, tmp_path):
    tr = CompetitionTrainer(cfg_of(batch_size=16), chess_data, budget_flops=1e18, time_cap_s=60)
    flag = threading.Event()
    threading.Timer(0.3, flag.set).start()
    assert tr.run(flag.is_set) == "aborted"
    assert not list(tmp_path.iterdir())


def test_loss_decreases_on_learnable_signal(tmp_path):
    """A real learning signal: on a tiny fixed dataset the validation MSE must drop clearly."""
    from tests.fake_data import write_lichess

    d = write_lichess(tmp_path, 3000, seed=4)
    # make cp a simple function of material so there is something to learn
    b = np.load(d / "boards.npy")
    mat = E.material_from_boards(torch.from_numpy(b)).numpy()
    np.save(d / "cp.npy", np.clip(mat * 100, -2000, 2000).astype(np.int16))
    np.save(d / "mate.npy", np.zeros(3000, np.int16))
    data = ChessData(LichessData(d), "cpu", val_rows=300)
    cfg = cfg_of(layers=2, width=64, lr=3e-3, batch_size=64, input_extras={"material": True})
    tr = CompetitionTrainer(cfg, data, budget_flops=1e18, time_cap_s=60)
    before = tr.evaluate()["val_mse"]
    tr.budget_flops = flops_per_sample(cfg["in_dim"], cfg["widths"]) * 64 * 400
    assert tr.run(lambda: False) == "budget"
    after = tr.evaluate()["val_mse"]
    assert after < before * 0.5, (before, after)


def test_checkpoint_resume_is_exact(chess_data, tmp_path):
    cfg = cfg_of(batch_size=16, ema={"enabled": True, "decay": 0.99}, lr_schedule="cosine", warmup_frac=0.1, sampling="weighted",
                 color_flip_augmentation=True, grad_clip=1.0)
    fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
    budget = fps * 16 * 60

    def fresh():
        return CompetitionTrainer(cfg, chess_data, budget_flops=budget, time_cap_s=60)

    ref = fresh()
    assert ref.run(lambda: False) == "budget"
    a = fresh()
    stop_at = 25
    assert a.run(lambda: a.step >= stop_at) == "aborted"
    a.save_checkpoint(tmp_path / "ck.pt")
    b = fresh()
    b.load_checkpoint(tmp_path / "ck.pt")
    assert b.step == stop_at and b.samples_seen == 16 * stop_at and b.flops_used == fps * 16 * stop_at  # no double counting
    assert b.run(lambda: False) == "budget"
    assert b.samples_seen == ref.samples_seen
    for (k, v), (_, w) in zip(ref.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.allclose(v, w, atol=1e-5), k  # RNG state restored: identical trajectory


@pytest.mark.parametrize("extra", [{}, {"data_slice": "endgame"}, {"sampling": "weighted", "data_slice": "decisive"},
                                   {"target_type": "cp", "loss": {"name": "huber", "delta": 0.5}, "output_head": "tanh"},
                                   {"perspective_flip": False, "output_head": "sigmoid"}])
def test_knobs_run_end_to_end_and_export_loads(chess_data, tmp_path, extra):
    cfg = cfg_of(batch_size=16, **extra)
    tr = CompetitionTrainer(cfg, chess_data, budget_flops=flops_per_sample(cfg["in_dim"], cfg["widths"]) * 16 * 10, time_cap_s=30)
    assert tr.run(lambda: False) == "budget"
    meta = tr.export(tmp_path / "m1", "budget", identity={"nickname": "t", "model_name": "T"})
    assert meta["stop_reason"] == "budget" and np.isfinite(meta["val_loss"])
    assert [p.name for p in list_models(tmp_path)] == ["m1"]
    lm = load_model(tmp_path / "m1")
    b = chess_data.fetch(chess_data.val_idx[:7])
    s = lm.score(b["boards"], b["stm"], b["castle"], b["ep"])
    assert s.shape == (7,) and torch.isfinite(s).all()
    # fixed-size chunks: a position's score does not depend on its neighbours
    s1 = lm.score(b["boards"][:1], b["stm"][:1], b["castle"][:1], b["ep"][:1])
    assert torch.equal(s[:1], s1)
    assert lm.name == "T" and lm.search_depth_full_moves in (1, 2, 3)
