"""Phase 2: rasterizer + audio golden tests (Python side), loaders, rasterize_quickdraw.py."""

import json
import sys

import numpy as np
import pytest

from app.data import audio_features as af
from app.data.loaders import DATA_SLICES, LichessData, QuickDrawData, SpeechData
from app.data.rasterizer import rasterize
from tests.conftest import REPO_ROOT, SAMPLES_DIR, requires_torch

GOLDEN = REPO_ROOT / "shared" / "golden"


def test_rasterizer_matches_golden_exactly():
    g = json.loads((GOLDEN / "rasterizer.json").read_text())
    assert len(g["cases"]) >= 15
    for c in g["cases"]:
        assert rasterize(c["strokes"]).ravel().tolist() == c["pixels"], c["name"]


def test_rasterizer_properties():
    base = [[[0, 100, 100], [0, 0, 100]]]
    img = rasterize(base)
    assert img.shape == (28, 28) and img.dtype == np.uint8
    # scale / offset invariance (floats from a browser canvas)
    big = [[[x * 3.7 + 41 for x in base[0][0]], [y * 3.7 + 9 for y in base[0][1]]]]
    assert np.abs(rasterize(big).astype(int) - img.astype(int)).max() <= 1
    # aspect ratio preserved: a horizontal line occupies one row band and spans the width
    line = rasterize([[[0, 300], [0, 0]]])
    assert line[:, :].max() == 255 and np.flatnonzero(line.sum(axis=0))[[0, -1]].tolist() == [1, 26]
    assert np.flatnonzero(line.sum(axis=1)).max() <= 4  # top-aligned, thin
    assert rasterize([]).sum() == 0
    assert rasterize([[[5], [5]]]).max() > 200  # a dot


def test_rasterizer_real_samples_have_ink():
    for r in json.loads((SAMPLES_DIR / "quickdraw" / "samples.json").read_text()):
        img = rasterize(r["d"])
        assert 20 < (img > 0).sum() < 600


def test_audio_golden_python():
    g = json.loads((GOLDEN / "audio.json").read_text())
    d = SAMPLES_DIR / "speech"
    for c in g["cases"]:
        wav = af.read_wav(d / "samples" / c["file"])
        assert len(wav) == c["n_samples"]
        lm = af.log_mel(wav)
        assert lm.shape == (40, 98)
        np.testing.assert_allclose(lm, np.array(c["log_mel"], dtype=np.float32), atol=1e-4)
        np.testing.assert_allclose(af.raw_input(wav)[:64], c["raw_input_head"], atol=1e-5)
    fb = af.mel_filterbank()
    for m, row in g["fbank_rows"].items():
        np.testing.assert_allclose(fb[int(m)], row, atol=1e-5)


def test_audio_properties():
    lm = af.log_mel(np.zeros(16000, np.int16))
    assert np.isfinite(lm).all() and lm.shape == (40, 98)
    short = af.log_mel(np.ones(4000, np.float32) * 0.1)  # padded at the end
    assert np.isfinite(short).all()
    wav = af.read_wav(SAMPLES_DIR / "speech" / "samples" / "yes__022cd682_nohash_0.wav")
    lm = af.log_mel(wav)
    assert abs(lm.mean()) < 1e-4 and abs(lm.std() - 1) < 1e-3


@requires_torch
def test_audio_torch_twin_matches_numpy():
    import torch

    d = SAMPLES_DIR / "speech"
    X = np.stack([af.pad_or_trim(af.read_wav(d / "samples" / s["file"])) for s in json.loads((d / "samples.json").read_text())])
    a = np.stack([af.log_mel(x) for x in X])
    b = af.log_mel_batch_torch(torch.from_numpy(X)).numpy()
    assert np.abs(a - b).max() < 1e-3


# ---------------------------------------------------------------- loaders on fake data, samples, and rasterize script

def test_quickdraw_loader_on_samples_and_fake(fake_data_dir):
    s = QuickDrawData(SAMPLES_DIR / "quickdraw")
    x, y = s.rasterized("probe")
    assert x.shape == (16, 28, 28) and set(y.tolist()) == set(range(10))
    f = QuickDrawData(fake_data_dir / "quickdraw")
    assert sum(1 for _ in f.iter_jsonl("val")) >= 50
    assert len(f.json_records("duel")) == 200 and not f.has_tensors()


def test_rasterize_script_matches_direct_and_is_idempotent(tmp_path, fake_data_dir, monkeypatch, capsys):
    import shutil

    data = tmp_path / "data"
    shutil.copytree(fake_data_dir / "quickdraw", data / "quickdraw")
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from scripts import rasterize_quickdraw as rq

    monkeypatch.setattr(rq, "CHUNK_LINES", 7)  # many chunks, two workers: order must survive
    assert rq.main(["--data-dir", str(data), "--workers", "2"]) == 0
    qd = QuickDrawData(data / "quickdraw")
    assert qd.has_tensors()
    recs = list(qd.iter_jsonl("val"))
    x, y = qd.tensors("val")
    assert x.shape == (len(recs), 28, 28) and x.dtype == np.uint8 and y.dtype == np.int8
    for i in (0, 3, len(recs) - 1):
        assert np.array_equal(x[i], rasterize(recs[i]["d"])) and y[i] == recs[i]["c"]
    mtime = (data / "quickdraw/tensors/val_x.npy").stat().st_mtime_ns
    capsys.readouterr()
    rq.main(["--data-dir", str(data), "--workers", "2"])
    assert (data / "quickdraw/tensors/val_x.npy").stat().st_mtime_ns == mtime  # skipped
    assert "skipped" in capsys.readouterr().out
    assert not list((data / "quickdraw/tensors").glob(".*tmp*"))  # no half-written leftovers


def test_speech_loader(fake_data_dir):
    sp = SpeechData(fake_data_dir / "speech")
    assert sp.X("train").dtype == np.int16
    assert (sp.y("train")[sp.word_indices("train")] < 10).all()
    idx, lab = sp.marvin_subset("train", 0.02, seed=3)
    assert lab.sum() >= 1 and (sp.y("train")[idx] == 10).sum() == lab.sum()
    # all negatives are kept
    assert (lab == 0).sum() == (sp.y("train") != 10).sum()
    assert np.array_equal(idx, sp.marvin_subset("train", 0.02, seed=3)[0])  # fixed seed
    assert len(sp.overfit_subset(20)) == 20
    real = SpeechData(SAMPLES_DIR / "speech").samples()
    assert len(real) == 20 and real[0]["x"].shape == (16000,)


def test_marvin_subset_ratio_on_realistic_counts(tmp_path):
    """SPEC 4.2: ~2% positives, ~630 against ~30,800 negatives in train."""
    d = tmp_path / "speech"
    d.mkdir()
    y = np.concatenate([np.repeat(np.arange(10), 3080), np.full(1710, 10)]).astype(np.int8)
    np.save(d / "y_train.npy", y)
    (d / "classes.json").write_text(json.dumps(af.__dict__.get("X", None) or ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "marvin"]))
    idx, lab = SpeechData(d).marvin_subset("train")
    assert (lab == 0).sum() == 30800 and 600 <= lab.sum() <= 640
    assert abs(lab.mean() - 0.02) < 0.002


def test_lichess_loader_slices(fake_data_dir):
    li = LichessData(fake_data_dir / "lichess")
    n_val = len(li.val_indices())
    assert n_val >= 1
    assert len(li.train_indices("all")) == li.n - n_val
    cp, mate, npc = (np.asarray(li[k]) for k in ("cp", "mate", "npc"))
    tr = {s: li.train_indices(s) for s in DATA_SLICES}
    assert (npc[tr["endgame"]] <= 10).all()
    assert ((mate[tr["balanced"]] == 0) & (np.abs(cp[tr["balanced"]]) <= 100)).all()
    assert ((mate[tr["decisive"]] != 0) | (np.abs(cp[tr["decisive"]]) >= 300)).all()
    assert all(not np.isin(tr[s], li.val_indices()).any() for s in DATA_SLICES)
    g = li.gather(tr["all"][:5])
    assert g["boards"].shape == (5, 64)
    with pytest.raises(ValueError):
        li.train_indices("nope")


def test_lichess_loader_on_real_samples():
    li = LichessData(SAMPLES_DIR / "lichess")
    assert li.n == 2000 and len(li.train_indices("all")) + len(li.val_indices()) == 2000


def test_shared_config_defaults_file_is_current():
    """frontend imports shared/config_defaults.json: it must match the backend (regenerate with scripts/make_golden.py)."""
    from app.chess_net.config import DEFAULT_CONFIG, tier_table

    d = json.loads((REPO_ROOT / "shared" / "config_defaults.json").read_text())
    assert d["defaults"] == DEFAULT_CONFIG and d["tiers"] == tier_table()
