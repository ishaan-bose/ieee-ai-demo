"""The server scripts, run in-process on tiny fake data in quick mode (what run_on_server.sh runs on the real box)."""

import json
import shutil

import numpy as np
import pytest
import torch

from tests.conftest import REPO_ROOT, requires_torch

pytestmark = requires_torch


@pytest.fixture(scope="module")
def sdata(tmp_path_factory, fake_data_dir):
    from tests.fake_data import make_fake_data
    from scripts import rasterize_quickdraw as rq
    import os

    root = tmp_path_factory.mktemp("scripts")
    make_fake_data(root / "data", 200, quickdraw=(300, 60, 230), speech=(700, 120, 120), lichess=3000)
    os.environ["LOG_DIR"] = str(root / "logs")
    rq.main(["--data-dir", str(root / "data"), "--workers", "1"])
    return root


@pytest.fixture
def senv(monkeypatch, sdata):
    monkeypatch.setenv("DATA_DIR", str(sdata / "data"))
    monkeypatch.setenv("STATE_DIR", str(sdata / "state"))
    monkeypatch.setenv("LOG_DIR", str(sdata / "logs"))
    monkeypatch.setenv("NUM_WORKERS", "1")
    return sdata


def test_export_format_matches_torch_and_golden_fixture():
    from app.export_format import export_sequential, numpy_forward

    g = json.loads((REPO_ROOT / "shared" / "golden" / "inference.json").read_text())
    for m in g["models"]:
        w = np.array(m["weights"], np.float32)
        for c in m["cases"]:
            x = np.array(c["x"], np.float32).reshape(m["input_shape"])
            np.testing.assert_allclose(numpy_forward(m["layers"], w, x), c["y"], atol=g["tolerance"])


def test_showcase_export_and_cached_json(senv):
    from scripts import export_weights, train_showcase

    fe = senv / "frontend"
    assert train_showcase.main(["--quick", "--frontend-dir", str(fe), "--doodle-epochs", "1"]) == 0
    for name in ("doodle", "raw_audio", "logmel_audio"):
        assert (senv / "state" / "showcase" / f"{name}.pt").is_file()
    trap = json.loads((fe / "public/cache/act3/trap.json").read_text())
    assert trap["always_no"]["accuracy"] > 0.95 and trap["always_no"]["recall"] == 0.0
    assert set(trap["naive"]["test"]) >= {"accuracy", "recall", "precision", "confusion"} and trap["weighted"]["epochs"]
    ov = json.loads((fe / "public/cache/act3/overfit.json").read_text())
    assert ov["runs"] and set(ov["runs"][0]["epochs"][0]) == {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    # idempotent: a second run skips everything
    before = (senv / "state" / "showcase" / "doodle.pt").stat().st_mtime_ns
    train_showcase.main(["--quick", "--frontend-dir", str(fe)])
    assert (senv / "state" / "showcase" / "doodle.pt").stat().st_mtime_ns == before
    # export, then verify the .bin against the torch checkpoint with the reference forward pass
    assert export_weights.main(["--frontend-dir", str(fe)]) == 0
    man = json.loads((fe / "public/models/manifest.json").read_text())
    assert set(man["models"]) == {"doodle", "raw_audio", "logmel_audio"}
    from app.export_format import numpy_forward
    from app.training import audio_models as am
    from app.training.classifier import build_mlp_classifier

    ck = torch.load(senv / "state/showcase/doodle.pt", weights_only=False)
    model = build_mlp_classifier(784, [256, 128], 10, "relu")
    model.load_state_dict(ck["state_dict"])
    ent = man["models"]["doodle"]
    w = np.fromfile(fe / "public" / ent["file"], dtype="<f4")
    x = np.random.default_rng(0).random(784).astype(np.float32)
    np.testing.assert_allclose(numpy_forward(ent["layers"], w, x), model(torch.from_numpy(x)[None]).detach().numpy()[0], atol=1e-4)
    assert ent["classes"][0] == "cat" and ent["input"]["shape"] == [784]
    assert man["models"]["logmel_audio"]["layers"][0]["type"] == "conv2d"


def test_record_races_grid_and_idempotency(senv):
    from app.training.race_grid import all_lane_configs
    from scripts import record_races

    fe = senv / "frontend2"
    assert record_races.main(["--quick", "--only", "lr", "--frontend-dir", str(fe)]) == 0
    files = sorted(p.name for p in (fe / "public/cache/races").glob("lr__*.json"))
    assert len(files) == 7
    rec = json.loads((fe / "public/cache/races" / files[0]).read_text())
    assert rec["ticks"] and rec["done"]["reason"] == "time" and len(rec["probe_labels"]) == 16 and len(rec["ticks"][-1]["probe_preds"]) == 16
    mt = (fe / "public/cache/races" / files[0]).stat().st_mtime_ns
    record_races.main(["--quick", "--only", "lr", "--frontend-dir", str(fe)])
    assert (fe / "public/cache/races" / files[0]).stat().st_mtime_ns == mt
    idx = json.loads((fe / "public/cache/races/index.json").read_text())
    assert len(idx["lanes"]) == 7 and len(all_lane_configs()) == 26


def test_house_net_script_trains_exports_and_is_idempotent(senv, capsys):
    from app.chess_net.model_io import load_model
    from scripts import train_house_net

    assert train_house_net.main(["--quick", "--trials", "0"]) == 0
    out = senv / "state" / "models" / "house-net"
    lm = load_model(out)
    assert lm.name == "House Net" and (out / "chosen_config.json").is_file()
    assert (senv / "state" / "baselines" / "default-config" / "weights.safetensors").is_file()
    assert "SUMMARY train_house_net" in capsys.readouterr().out
    mt = (out / "weights.safetensors").stat().st_mtime_ns
    train_house_net.main(["--quick"])
    assert (out / "weights.safetensors").stat().st_mtime_ns == mt  # skipped


def test_house_net_is_not_the_default_config():
    from app.chess_net.config import DEFAULT_CONFIG, resolve_config
    from scripts.train_house_net import HOUSE_START

    h, e = resolve_config(HOUSE_START)
    d, _ = resolve_config(DEFAULT_CONFIG)
    assert not e and h["param_count"] > d["param_count"] and h != d and HOUSE_START["ema"]["enabled"] and not DEFAULT_CONFIG["ema"]["enabled"]
