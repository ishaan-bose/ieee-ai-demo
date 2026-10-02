"""Executable version of the SPEC section 4 data contract.

One checker for all three sources of data, so the same rules apply everywhere:
- the fake data written by tests/fake_data.py,
- the real samples committed in data/samples/ (layout of section 4.5),
- the real data under DATA_DIR on the server (`python -m app.data.contract`).

Every check_* function returns a list of human-readable problems; empty = pass.
Large arrays are opened with mmap and checked in chunks, so this is safe on the
16 GB CPU plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- constants

QUICKDRAW_CLASSES = ["cat", "bicycle", "house", "pizza", "lightning", "star", "fish", "tree", "umbrella", "sword"]
QUICKDRAW_SPLITS = ("train", "val", "pool")
# Exact sizes of the derived JSON arrays (SPEC 4.1). Fake data uses the same sizes.
QUICKDRAW_DUEL_PER_CLASS = 20  # duel.json: 200 records from pool
QUICKDRAW_PROBE_N = 16  # probe.json: one per class + 6 extra, from pool
QUICKDRAW_SAMPLES_PER_CLASS = 5  # samples.json: 50 records from val
QUICKDRAW_REAL_COUNTS = {"train": 1_090_830, "val": 25_106, "pool": 2_579}
# Google's "simplified" coordinates are only approximately normalized: in the
# real samples ~24% of drawings start at 1-2 instead of 0, or span 253-254
# instead of 255 (see NOTES.md). Allowed slack, in pixels:
QUICKDRAW_NORM_TOL = 3

SPEECH_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "marvin"]
SPEECH_SPLITS = ("train", "val", "test")
SPEECH_RATE = 16_000
SPEECH_LEN = 16_000
SPEECH_MARVIN = 10
SPEECH_SAMPLES_PER_WORD = 2
SPEECH_REAL_COUNTS = {"train": 32_479, "val": 3_898, "test": 4_269}

# name -> (dtype, shape after the leading N)
LICHESS_ARRAYS: dict[str, tuple[str, tuple[int, ...]]] = {
    "boards": ("uint8", (64,)),
    "stm": ("uint8", ()),
    "castle": ("uint8", ()),
    "ep": ("int8", ()),
    "cp": ("int16", ()),
    "mate": ("int16", ()),
    "depth": ("int16", ()),
    "mat": ("int8", ()),
    "npc": ("uint8", ()),
    "is_val": ("uint8", ()),
    "shard_of": ("uint8", ()),
}
LICHESS_REAL_N = 12_000_000
LICHESS_REAL_VAL = 239_546
LICHESS_CP_CLIP = 10_000
# piece code -> material, codes 0 empty, 1-6 white PNBRQK, 7-12 black pnbrqk
LICHESS_MATERIAL = np.array([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0], dtype=np.int64)
LICHESS_WHITE_KING = 6
LICHESS_BLACK_KING = 12
LICHESS_WHITE_PAWN = 1
LICHESS_BLACK_PAWN = 7

CHUNK_ROWS = 1_000_000


def quickdraw_bucket(key: str) -> int:
    """Deterministic split bucket used by the owner's prep (md5 of key_id)."""
    return int(hashlib.md5(key.encode()).hexdigest(), 16) % 1000


def quickdraw_split_of(key: str) -> str:
    b = quickdraw_bucket(key)
    return "val" if b < 20 else "pool" if b < 22 else "train"


def _load_json(path: Path, problems: list[str]):
    if not path.is_file():
        problems.append(f"missing file {path}")
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        problems.append(f"{path}: invalid JSON ({e})")
        return None


# ---------------------------------------------------------------- Quick, Draw!

def check_quickdraw_record(r, where: str = "") -> list[str]:
    p: list[str] = []
    if not isinstance(r, dict) or set(r) != {"c", "k", "d"}:
        return [f"{where}: record must be an object with exactly keys c, k, d"]
    c, k, d = r["c"], r["k"], r["d"]
    if type(c) is not int or not 0 <= c <= 9:
        p.append(f"{where}: c must be an int 0-9, got {c!r}")
    if not isinstance(k, str) or not k.isdigit():
        p.append(f"{where}: k must be a digit string, got {k!r}")
    if not isinstance(d, list) or not d:
        return p + [f"{where}: d must be a non-empty list of strokes"]
    xs_all: list[int] = []
    ys_all: list[int] = []
    for si, s in enumerate(d):
        if not (isinstance(s, list) and len(s) == 2 and isinstance(s[0], list) and isinstance(s[1], list)):
            p.append(f"{where}: stroke {si} must be [xs, ys]")
            continue
        xs, ys = s
        if len(xs) != len(ys) or not xs:
            p.append(f"{where}: stroke {si} has unequal or empty xs/ys ({len(xs)}, {len(ys)})")
            continue
        if not all(type(v) is int and 0 <= v <= 255 for v in xs + ys):
            p.append(f"{where}: stroke {si} has a coordinate that is not an int in 0..255")
            continue
        xs_all += xs
        ys_all += ys
    if xs_all and not p:
        tol = QUICKDRAW_NORM_TOL
        if min(xs_all) > tol or min(ys_all) > tol:
            p.append(f"{where}: not aligned to top-left (min x {min(xs_all)}, min y {min(ys_all)})")
        extent = max(max(xs_all) - min(xs_all), max(ys_all) - min(ys_all))
        if extent < 255 - tol:
            p.append(f"{where}: larger side spans {extent}, expected ~255")
    return p


def _check_records(records, where: str, max_problems: int = 20) -> list[str]:
    p: list[str] = []
    for i, r in enumerate(records):
        p += check_quickdraw_record(r, f"{where}[{i}]")
        if len(p) >= max_problems:
            p.append(f"{where}: stopping after {max_problems} problems")
            break
    return p


def check_quickdraw_json_arrays(dir_: Path, strict_split: bool = True) -> list[str]:
    """classes.json, duel.json, probe.json, samples.json (present in both layouts)."""
    p: list[str] = []
    classes = _load_json(dir_ / "classes.json", p)
    if classes is not None and classes != QUICKDRAW_CLASSES:
        p.append(f"classes.json is {classes}, expected {QUICKDRAW_CLASSES}")

    def counts(recs):
        out = [0] * 10
        for r in recs:
            if isinstance(r, dict) and type(r.get("c")) is int and 0 <= r["c"] <= 9:
                out[r["c"]] += 1
        return out

    duel = _load_json(dir_ / "duel.json", p)
    if duel is not None:
        p += _check_records(duel, "duel.json")
        if counts(duel) != [QUICKDRAW_DUEL_PER_CLASS] * 10:
            p.append(f"duel.json class counts {counts(duel)}, expected {QUICKDRAW_DUEL_PER_CLASS} each")
    probe = _load_json(dir_ / "probe.json", p)
    if probe is not None:
        p += _check_records(probe, "probe.json")
        if len(probe) != QUICKDRAW_PROBE_N or min(counts(probe)) < 1:
            p.append(f"probe.json must have {QUICKDRAW_PROBE_N} records covering every class, got {counts(probe)}")
    samples = _load_json(dir_ / "samples.json", p)
    if samples is not None:
        p += _check_records(samples, "samples.json")
        if counts(samples) != [QUICKDRAW_SAMPLES_PER_CLASS] * 10:
            p.append(f"samples.json class counts {counts(samples)}, expected {QUICKDRAW_SAMPLES_PER_CLASS} each")

    if duel is not None and probe is not None and samples is not None and not p:
        keys = [r["k"] for r in duel + probe + samples]
        if len(set(keys)) != len(keys):
            p.append("duel/probe/samples share key_ids (they must be distinct drawings)")
        if strict_split:
            for name, recs, split in (("duel", duel, "pool"), ("probe", probe, "pool"), ("samples", samples, "val")):
                wrong = [r["k"] for r in recs if quickdraw_split_of(r["k"]) != split]
                if wrong:
                    p.append(f"{name}.json: {len(wrong)} keys do not hash into the {split} split, e.g. {wrong[0]}")
    return p


def check_quickdraw_jsonl(path: Path, split: str, max_records: int | None = None,
                          expected_lines: int | None = None) -> list[str]:
    """Validate a JSONL split. max_records limits full parsing (lines are still counted)."""
    p: list[str] = []
    if not path.is_file():
        return [f"missing file {path}"]
    n = 0
    per_class = [0] * 10
    with path.open("rb") as f:
        for line in f:
            if max_records is None or n < max_records:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    p.append(f"{path.name} line {n + 1}: invalid JSON")
                    r = None
                if r is not None:
                    rp = check_quickdraw_record(r, f"{path.name} line {n + 1}")
                    if not rp and quickdraw_split_of(r["k"]) != split:
                        rp.append(f"{path.name} line {n + 1}: key {r['k']} hashes into split "
                                  f"{quickdraw_split_of(r['k'])}, not {split}")
                    p += rp
                    if not rp:
                        per_class[r["c"]] += 1
                if len(p) >= 20:
                    p.append(f"{path.name}: stopping after 20 problems")
                    return p
            n += 1
    if n == 0:
        p.append(f"{path.name} is empty")
    if expected_lines is not None and n != expected_lines:
        p.append(f"{path.name} has {n} lines, expected {expected_lines}")
    if max_records is None and n and min(per_class) == 0:
        p.append(f"{path.name}: some class has no records ({per_class})")
    return p


def check_quickdraw_processed(dir_: Path, real: bool = False, max_records: int | None = None) -> list[str]:
    """The DATA_DIR/quickdraw/processed layout (fake or real)."""
    p = check_quickdraw_json_arrays(dir_)
    for split in QUICKDRAW_SPLITS:
        expected = QUICKDRAW_REAL_COUNTS[split] if real else None
        p += check_quickdraw_jsonl(dir_ / f"{split}.jsonl", split, max_records, expected)
    return p


# ---------------------------------------------------------------- Speech Commands

def _read_wav(path: Path) -> tuple[np.ndarray, list[str]]:
    p: list[str] = []
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1:
            p.append(f"{path.name}: {w.getnchannels()} channels, expected mono")
        if w.getsampwidth() != 2:
            p.append(f"{path.name}: sample width {w.getsampwidth()} bytes, expected 2 (int16)")
        if w.getframerate() != SPEECH_RATE:
            p.append(f"{path.name}: {w.getframerate()} Hz, expected {SPEECH_RATE}")
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if not 0 < len(a) <= SPEECH_LEN:
        p.append(f"{path.name}: {len(a)} samples, expected 1..{SPEECH_LEN}")
    return a, p


def check_speech_samples(dir_: Path) -> list[str]:
    """classes.json, samples.json, samples/*.wav, LICENSE (present in both layouts)."""
    p: list[str] = []
    classes = _load_json(dir_ / "classes.json", p)
    if classes is not None and classes != SPEECH_CLASSES:
        p.append(f"classes.json is {classes}, expected {SPEECH_CLASSES}")
    if not (dir_ / "LICENSE").is_file():
        p.append(f"missing file {dir_ / 'LICENSE'}")
    samples = _load_json(dir_ / "samples.json", p)
    if samples is None:
        return p
    per = [0] * 10
    for i, s in enumerate(samples):
        if not isinstance(s, dict) or set(s) != {"file", "label", "word"}:
            p.append(f"samples.json[{i}] must have exactly keys file, label, word")
            continue
        label, word, fname = s["label"], s["word"], s["file"]
        if type(label) is not int or not 0 <= label <= 9:
            p.append(f"samples.json[{i}]: label must be 0-9 (no marvin), got {label!r}")
            continue
        per[label] += 1
        if word != SPEECH_CLASSES[label]:
            p.append(f"samples.json[{i}]: word {word!r} does not match label {label}")
        if not fname.startswith(f"{word}__") or not fname.endswith(".wav"):
            p.append(f"samples.json[{i}]: file name {fname!r} should look like '{word}__<id>.wav'")
        wav = dir_ / "samples" / fname
        if not wav.is_file():
            p.append(f"missing file {wav}")
            continue
        _, wp = _read_wav(wav)
        p += wp
    if per != [SPEECH_SAMPLES_PER_WORD] * 10:
        p.append(f"samples.json per-word counts {per}, expected {SPEECH_SAMPLES_PER_WORD} each")
    return p


def check_speech_processed(dir_: Path, real: bool = False) -> list[str]:
    """The DATA_DIR/speech/processed layout (fake or real)."""
    p = check_speech_samples(dir_)
    if not (dir_ / "README.md").is_file():
        p.append(f"missing file {dir_ / 'README.md'}")
    speakers_by_split: dict[str, set[str]] = {}
    test_x = test_files = None
    for split in SPEECH_SPLITS:
        xp, yp = dir_ / f"X_{split}.npy", dir_ / f"y_{split}.npy"
        if not xp.is_file() or not yp.is_file():
            p.append(f"missing X_{split}.npy or y_{split}.npy")
            continue
        X = np.load(xp, mmap_mode="r")
        y = np.load(yp, mmap_mode="r")
        if X.dtype != np.int16 or X.ndim != 2 or X.shape[1] != SPEECH_LEN:
            p.append(f"X_{split}: {X.dtype} {X.shape}, expected int16 (N, {SPEECH_LEN})")
        if y.dtype != np.int8 or y.shape != (X.shape[0],):
            p.append(f"y_{split}: {y.dtype} {y.shape}, expected int8 ({X.shape[0]},)")
        n = X.shape[0]
        if real and n != SPEECH_REAL_COUNTS[split]:
            p.append(f"X_{split} has {n} rows, expected {SPEECH_REAL_COUNTS[split]}")
        if n and (y.min() < 0 or y.max() > SPEECH_MARVIN):
            p.append(f"y_{split}: labels outside 0..{SPEECH_MARVIN}")
        for kind in ("files", "speakers"):
            lst = _load_json(dir_ / f"{kind}_{split}.json", p)
            if lst is None:
                continue
            if not isinstance(lst, list) or len(lst) != n or not all(isinstance(v, str) for v in lst):
                p.append(f"{kind}_{split}.json must be a list of {n} strings")
                continue
            if kind == "files":
                test_files = lst if split == "test" else test_files
                for i in range(n):
                    word = lst[i].split("/")[0]
                    if word not in SPEECH_CLASSES or SPEECH_CLASSES.index(word) != int(y[i]):
                        p.append(f"files_{split}.json[{i}] = {lst[i]!r} does not match label {int(y[i])}")
                        break
            else:
                speakers_by_split[split] = set(lst)
        if split == "test":
            test_x = X
    splits = list(speakers_by_split)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = speakers_by_split[a] & speakers_by_split[b]
            if shared:
                p.append(f"speakers appear in both {a} and {b}: {sorted(shared)[:3]}")

    # samples/ are copies of test clips: each must equal its test row (zero-padded to 1 s).
    samples = _load_json(dir_ / "samples.json", [])
    if samples and test_x is not None and test_files is not None:
        index = {f.replace("/", "__"): i for i, f in enumerate(test_files)}
        for s in samples:
            wav = dir_ / "samples" / s["file"]
            if s["file"] not in index:
                p.append(f"sample {s['file']} is not in files_test.json")
                continue
            if wav.is_file():
                a, _ = _read_wav(wav)
                row = np.asarray(test_x[index[s["file"]]])
                if not (np.array_equal(row[: len(a)], a) and not row[len(a):].any()):
                    p.append(f"sample {s['file']} does not match its X_test row")
    return p


# ---------------------------------------------------------------- Lichess

def check_lichess(dir_: Path, real: bool = False, expected_n: int | None = None) -> list[str]:
    """The DATA_DIR/lichess layout (fake, data/samples/lichess, or real)."""
    p: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for name, (dtype, tail) in LICHESS_ARRAYS.items():
        path = dir_ / f"{name}.npy"
        if not path.is_file():
            p.append(f"missing file {path}")
            continue
        a = np.load(path, mmap_mode="r")
        if a.dtype != np.dtype(dtype) or a.shape[1:] != tail:
            p.append(f"{name}: {a.dtype} {a.shape}, expected {dtype} (N{', ' + str(tail[0]) if tail else ''})")
            continue
        arrays[name] = a
    meta = _load_json(dir_ / "meta.json", p)
    if p:
        return p

    ns = {name: a.shape[0] for name, a in arrays.items()}
    n = ns["boards"]
    if len(set(ns.values())) != 1:
        return [f"arrays have different lengths: {ns}"]
    if real:
        expected_n = LICHESS_REAL_N
    if expected_n is not None and n != expected_n:
        p.append(f"N = {n}, expected {expected_n}")
    if n == 0:
        return p + ["arrays are empty"]
    if isinstance(meta, dict) and "convention" not in meta:
        p.append("meta.json has no 'convention' entry")

    def fail(msg: str, bad: np.ndarray, start: int):
        idx = np.flatnonzero(bad)
        if idx.size:
            p.append(f"{msg}: {idx.size} rows in chunk at {start}, first row {start + int(idx[0])}")

    n_val = 0
    for start in range(0, n, CHUNK_ROWS):
        sl = slice(start, min(n, start + CHUNK_ROWS))
        b = np.asarray(arrays["boards"][sl])
        stm, castle, ep = (np.asarray(arrays[k][sl]) for k in ("stm", "castle", "ep"))
        cp, mate, depth = (np.asarray(arrays[k][sl]) for k in ("cp", "mate", "depth"))
        mat, npc, is_val, shard = (np.asarray(arrays[k][sl]) for k in ("mat", "npc", "is_val", "shard_of"))
        fail("board codes outside 0..12", (b > 12).any(axis=1), start)
        fail("not exactly one white king", (b == LICHESS_WHITE_KING).sum(axis=1) != 1, start)
        fail("not exactly one black king", (b == LICHESS_BLACK_KING).sum(axis=1) != 1, start)
        back = np.concatenate([b[:, :8], b[:, 56:]], axis=1)
        fail("pawn on rank 1 or 8", ((back == LICHESS_WHITE_PAWN) | (back == LICHESS_BLACK_PAWN)).any(axis=1), start)
        fail("npc != number of pieces on the board", (b > 0).sum(axis=1) != npc, start)
        fail("mat != white minus black material", LICHESS_MATERIAL[np.minimum(b, 12)].sum(axis=1) != mat, start)
        fail("stm not in {0,1}", stm > 1, start)
        fail("castle not in 0..15", castle > 15, start)
        fail("ep not in -1..7", (ep < -1) | (ep > 7), start)
        fail(f"|cp| > {LICHESS_CP_CLIP}", np.abs(cp.astype(np.int32)) > LICHESS_CP_CLIP, start)
        fail("mate != 0 but cp != 0", (mate != 0) & (cp != 0), start)
        fail("depth < 1", depth < 1, start)
        fail("is_val not in {0,1}", is_val > 1, start)
        fail("shard_of not in {0,1}", shard > 1, start)
        n_val += int(is_val.sum())
        if len(p) >= 20:
            p.append("stopping after 20 problems")
            break
    if real and n_val != LICHESS_REAL_VAL:
        p.append(f"{n_val} validation rows, expected {LICHESS_REAL_VAL}")
    if n >= 100 and not 0.005 <= n_val / n <= 0.05:
        p.append(f"validation share {n_val / n:.4f}, expected about 0.02")
    return p


# ---------------------------------------------------------------- whole DATA_DIR

def check_data_dir(data_dir: Path, real: bool = False, max_records: int | None = None) -> dict[str, list[str]]:
    """Check the full DATA_DIR layout. Returns {dataset: problems}."""
    data_dir = Path(data_dir)
    return {
        "quickdraw": check_quickdraw_processed(data_dir / "quickdraw" / "processed", real, max_records),
        "speech": check_speech_processed(data_dir / "speech" / "processed", real),
        "lichess": check_lichess(data_dir / "lichess", real),
    }


def check_samples_dir(samples_dir: Path) -> dict[str, list[str]]:
    """Check the committed data/samples/ layout (SPEC 4.5)."""
    samples_dir = Path(samples_dir)
    lichess_dir = samples_dir / "lichess"
    meta = _load_json(lichess_dir / "meta.json", [])
    rows = (meta or {}).get("sample", {}).get("rows")
    return {
        "quickdraw": check_quickdraw_json_arrays(samples_dir / "quickdraw"),
        "speech": check_speech_samples(samples_dir / "speech"),
        "lichess": check_lichess(lichess_dir, expected_n=rows),
    }


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    ap = argparse.ArgumentParser(description="Check DATA_DIR against SPEC section 4.")
    ap.add_argument("--data-dir", type=Path, default=None, help="default: $DATA_DIR (~/demo/data)")
    ap.add_argument("--real", action="store_true", help="also check the exact real counts from SPEC 4")
    ap.add_argument("--max-records", type=int, default=None,
                    help="fully parse only the first N lines of each quickdraw JSONL (lines are still counted)")
    args = ap.parse_args(argv)
    data_dir = args.data_dir or get_settings().data_dir
    print(f"Checking {data_dir} (real counts: {args.real})")
    results = check_data_dir(data_dir, args.real, args.max_records)
    ok = True
    for name, problems in results.items():
        print(f"{'PASS' if not problems else 'FAIL'} {name}")
        for prob in problems:
            print(f"    - {prob}")
        ok = ok and not problems
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
