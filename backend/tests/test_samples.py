"""Golden checks of the real files committed in data/samples/ against SPEC section 4."""

import json
import wave

import numpy as np

from app.data import contract
from tests.conftest import SAMPLES_DIR


def test_samples_pass_contract():
    assert contract.check_samples_dir(SAMPLES_DIR) == {"quickdraw": [], "speech": [], "lichess": []}


def test_quickdraw_samples_facts():
    d = SAMPLES_DIR / "quickdraw"
    assert json.loads((d / "classes.json").read_text()) == contract.QUICKDRAW_CLASSES
    duel = json.loads((d / "duel.json").read_text())
    probe = json.loads((d / "probe.json").read_text())
    samples = json.loads((d / "samples.json").read_text())
    assert (len(duel), len(probe), len(samples)) == (200, 16, 50)
    # split provenance: duel/probe from pool, samples from val (md5 of key_id)
    assert {contract.quickdraw_split_of(r["k"]) for r in duel + probe} == {"pool"}
    assert {contract.quickdraw_split_of(r["k"]) for r in samples} == {"val"}


def test_quickdraw_coordinates_verify_item():
    """SPEC 4.1 [verify]: 'aligned to the top-left, larger side 255'.

    Result on the real samples: every coordinate is an int in 0..255, but the
    normalization is only approximate (off by up to 2 px in ~24% of drawings).
    The rasterizer therefore must re-normalize (it does so anyway, per SPEC).
    """
    recs = []
    for name in ("duel", "probe", "samples"):
        recs += json.loads((SAMPLES_DIR / "quickdraw" / f"{name}.json").read_text())
    exact = 0
    for r in recs:
        xs = [x for s in r["d"] for x in s[0]]
        ys = [y for s in r["d"] for y in s[1]]
        assert all(0 <= v <= 255 for v in xs + ys)
        assert min(xs) <= 2 and min(ys) <= 2
        assert max(max(xs) - min(xs), max(ys) - min(ys)) >= 253
        exact += min(xs) == 0 and min(ys) == 0 and max(max(xs), max(ys)) == 255
    assert 0.6 < exact / len(recs) < 1.0  # mostly exact, but not always


def test_speech_samples_facts():
    d = SAMPLES_DIR / "speech"
    samples = json.loads((d / "samples.json").read_text())
    assert len(samples) == 20
    assert sorted(s["label"] for s in samples) == sorted(list(range(10)) * 2)
    for s in samples:
        with wave.open(str(d / "samples" / s["file"])) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert 0 < len(a) <= 16000
        x = a.astype(np.float32) / 32768.0  # SPEC 4.2 conversion
        assert -1.0 <= x.min() and x.max() < 1.0 and np.abs(x).max() > 0.01  # not silent
    assert "Attribution 4.0" in (d / "LICENSE").read_text()


def test_lichess_samples_facts():
    d = SAMPLES_DIR / "lichess"
    meta = json.loads((d / "meta.json").read_text())
    assert meta["sample"] == {"rows": 2000, "stride": 6000, "note": meta["sample"]["note"]}
    assert meta["kept"] == contract.LICHESS_REAL_N
    a = {k: np.load(d / f"{k}.npy") for k in contract.LICHESS_ARRAYS}
    for k, (dtype, tail) in contract.LICHESS_ARRAYS.items():
        assert a[k].dtype == np.dtype(dtype) and a[k].shape == (2000, *tail), k
    assert set(np.unique(a["shard_of"])) == {0, 1}  # both shards represented
    assert 0.05 < (a["mate"] != 0).mean() < 0.2  # about 12% mate scores
    assert 0.005 < a["is_val"].mean() < 0.04  # about 2% validation
    # SPEC 4.3: positive mate = White mates. Material should lean the same way.
    assert a["mat"][a["mate"] > 0].mean() > a["mat"][a["mate"] < 0].mean()
