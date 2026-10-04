"""fake_data.py must produce exactly the SPEC section 4 layout, at any size."""

import json

import numpy as np
import pytest

from app.data import contract
from tests.fake_data import make_fake_data, write_lichess, write_quickdraw


@pytest.mark.parametrize("n", [1, 60, 400])
def test_fake_data_passes_contract(tmp_path, n):
    root = make_fake_data(tmp_path, n=n, seed=n)
    results = contract.check_data_dir(root)
    assert results == {"quickdraw": [], "speech": [], "lichess": []}


def test_same_file_names_as_spec(fake_data_dir):
    qd = {p.name for p in (fake_data_dir / "quickdraw" / "processed").iterdir()}
    assert qd == {"train.jsonl", "val.jsonl", "pool.jsonl", "classes.json", "duel.json", "probe.json", "samples.json"}
    sp = {p.name for p in (fake_data_dir / "speech" / "processed").iterdir()}
    expected = {"classes.json", "samples.json", "samples", "LICENSE", "README.md"}
    for s in contract.SPEECH_SPLITS:
        expected |= {f"X_{s}.npy", f"y_{s}.npy", f"files_{s}.json", f"speakers_{s}.json"}
    assert sp == expected
    li = {p.name for p in (fake_data_dir / "lichess").iterdir()}
    assert li == {f"{k}.npy" for k in contract.LICHESS_ARRAYS} | {"meta.json"}


def test_explicit_sizes(tmp_path):
    root = make_fake_data(tmp_path, quickdraw=(30, 60, 230), speech=(40, 15, 25), lichess=123)
    qd = root / "quickdraw" / "processed"
    assert [sum(1 for _ in open(qd / f"{s}.jsonl")) for s in ("train", "val", "pool")] == [30, 60, 230]
    sp = root / "speech" / "processed"
    assert [np.load(sp / f"X_{s}.npy").shape[0] for s in ("train", "val", "test")] == [40, 15, 25]
    assert np.load(root / "lichess" / "boards.npy").shape == (123, 64)


def test_deterministic(tmp_path):
    a = make_fake_data(tmp_path / "a", n=30, seed=5)
    b = make_fake_data(tmp_path / "b", n=30, seed=5)
    for rel in ("quickdraw/processed/train.jsonl", "quickdraw/processed/duel.json",
                "speech/processed/files_train.json", "lichess/boards.npy", "lichess/cp.npy"):
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def test_too_small_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_quickdraw(tmp_path, 10, 10, 10)
    with pytest.raises(ValueError):
        write_lichess(tmp_path, 1)


def test_conventions(fake_data_dir):
    sp = fake_data_dir / "speech" / "processed"
    y = np.load(sp / "y_train.npy")
    assert (y == contract.SPEECH_MARVIN).any()  # marvin rows exist (excluded later by the loader)
    X = np.load(sp / "X_train.npy")
    assert (X[:, -1] == 0).any()  # some clips are zero-padded at the end
    li = fake_data_dir / "lichess"
    boards, castle = np.load(li / "boards.npy"), np.load(li / "castle.npy")
    start = np.array([10, 8, 9, 11, 12, 9, 8, 10] + [7] * 8 + [0] * 32 + [1] * 8 + [4, 2, 3, 5, 6, 3, 2, 4], np.uint8)
    rows = np.flatnonzero((boards == start).all(axis=1))
    assert rows.size > 0  # the start position appears, oriented a8..h1
    assert (castle[rows] == 15).all()
    mate, cp = np.load(li / "mate.npy"), np.load(li / "cp.npy")
    assert (mate != 0).any() and (cp[mate != 0] == 0).all()


# --- the checker must actually catch broken files (not pass vacuously) ---

def test_checker_catches_bad_lichess(tmp_path):
    d = write_lichess(tmp_path, 50)
    np.save(d / "cp.npy", np.load(d / "cp.npy").astype(np.int32))
    assert any("cp" in p for p in contract.check_lichess(d))
    d = write_lichess(tmp_path / "b", 50)
    b = np.load(d / "boards.npy")
    b[3, b[3] == 6] = 0  # remove the white king
    np.save(d / "boards.npy", b)
    probs = contract.check_lichess(d)
    assert any("white king" in p for p in probs) and any("npc" in p for p in probs)
    assert not any("mat" in p.split(":")[0] for p in probs)
    d = write_lichess(tmp_path / "c", 50)
    st = {}
    mat = np.load(d / "mat.npy")
    mat[5] += 1  # a mat/board mismatch is a statistic, not a failure
    np.save(d / "mat.npy", mat)
    assert contract.check_lichess(d, stats=st) == [] and st["mat_mismatch_rows"] == 1
    m, cp = np.load(d / "mate.npy"), np.load(d / "cp.npy")
    i = int(np.flatnonzero(m == 0)[0])
    m[i] = 3
    cp[i] = 50
    np.save(d / "mate.npy", m)
    np.save(d / "cp.npy", cp)
    assert any("mate != 0 but cp != 0" in p for p in contract.check_lichess(d))


def test_checker_catches_bad_quickdraw(tmp_path):
    d = write_quickdraw(tmp_path, 20, 50, 220)
    duel = json.loads((d / "duel.json").read_text())
    duel[0]["d"][0][0].append(5)  # xs longer than ys
    (d / "duel.json").write_text(json.dumps(duel))
    assert any("unequal" in p for p in contract.check_quickdraw_processed(d))
    st: dict = {}  # a record spanning only 100 is a statistic, not a failure (accepted imperfection)
    assert not contract.check_quickdraw_record({"c": 3, "k": "1", "d": [[[0, 100], [0, 50]]]}, stats=st)
    assert st["larger_side_not_255"] == 1 and st["larger_side_spans_less_than_252"] == 1
    assert contract.check_quickdraw_record({"c": 10, "k": "1", "d": [[[0, 255], [0, 3]]]})  # bad class
    assert not contract.check_quickdraw_record({"c": 9, "k": "1", "d": [[[1, 255], [0, 3]]]})


def test_checker_catches_bad_speech(tmp_path):
    root = make_fake_data(tmp_path, n=30)
    sp = root / "speech" / "processed"
    np.save(sp / "y_val.npy", np.load(sp / "y_val.npy").astype(np.int64))
    spk_train = json.loads((sp / "speakers_train.json").read_text())
    spk_test = json.loads((sp / "speakers_test.json").read_text())
    spk_test[0] = spk_train[0]  # leak a speaker across splits
    (sp / "speakers_test.json").write_text(json.dumps(spk_test))
    probs = contract.check_speech_processed(sp)
    assert any("y_val" in p for p in probs)
    assert any("speakers appear in both" in p for p in probs)
