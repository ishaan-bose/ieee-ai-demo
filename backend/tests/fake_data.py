"""Tiny synthetic datasets in exactly the SPEC section 4 formats.

Same directory layout, file names, dtypes, shapes and conventions as the real
data under DATA_DIR, so the same loaders run on fake data, on data/samples/ and
on the real files. Contents are random but obey every invariant the owner's
audit checked (one king per side, npc/mat consistent with the board, mate rows
have cp == 0, speaker-separated splits, md5-based quickdraw splits, ...).

Usage:
    python -m tests.fake_data --out /tmp/fake_data --n 200     (from backend/)
    DATA_DIR=/tmp/fake_data uvicorn ...                         (run the app on it)

`n` scales every dataset; individual sizes can be overridden. A few derived
files have fixed sizes by contract (duel 200, probe 16, samples 50, speech
samples 20), so some splits have a minimum size.
"""

from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path

import numpy as np

from app.data.contract import (
    LICHESS_MATERIAL,
    QUICKDRAW_CLASSES,
    QUICKDRAW_DUEL_PER_CLASS,
    QUICKDRAW_PROBE_N,
    QUICKDRAW_SAMPLES_PER_CLASS,
    SPEECH_CLASSES,
    SPEECH_LEN,
    SPEECH_MARVIN,
    SPEECH_RATE,
    SPEECH_SAMPLES_PER_WORD,
    quickdraw_split_of,
)

# Smallest splits that can still supply the fixed-size derived files.
MIN_QD_POOL = 10 * (QUICKDRAW_DUEL_PER_CLASS + 2)  # 20 per class for duel + probe picks
MIN_QD_VAL = 10 * QUICKDRAW_SAMPLES_PER_CLASS
MIN_SPEECH_TEST = 11 * SPEECH_SAMPLES_PER_WORD  # labels cycle over 11 classes
MIN_SPEECH_SPLIT = 11


# ---------------------------------------------------------------- Quick, Draw!

def _fake_drawing(rng: random.Random) -> list:
    strokes = []
    for _ in range(rng.randint(1, 6)):
        x, y = rng.uniform(0, 100), rng.uniform(0, 100)
        xs, ys = [], []
        for _ in range(rng.randint(2, 20)):
            x += rng.uniform(-15, 15)
            y += rng.uniform(-15, 15)
            xs.append(x)
            ys.append(y)
        strokes.append((xs, ys))
    min_x = min(min(s[0]) for s in strokes)
    min_y = min(min(s[1]) for s in strokes)
    extent = max(max(max(s[0]) for s in strokes) - min_x, max(max(s[1]) for s in strokes) - min_y) or 1.0
    # Like Google's "simplified" data: aligned top-left, larger side 255. Real data
    # is off by 1-2 px in ~24% of drawings, so imitate that too.
    offset_x, offset_y, side = 0, 0, 255
    if rng.random() < 0.25:
        offset_x, offset_y, side = rng.choice([(0, 1, 255), (1, 0, 255), (0, 2, 255), (2, 0, 255),
                                               (0, 0, 254), (0, 0, 253)])
    out = []
    for xs, ys in strokes:
        out.append([
            [min(255, int(round((v - min_x) / extent * (side - offset_x))) + offset_x) for v in xs],
            [min(255, int(round((v - min_y) / extent * (side - offset_y))) + offset_y) for v in ys],
        ])
    return out


def _key_for_split(rng: random.Random, split: str, used: set[str]) -> str:
    while True:
        k = str(rng.randrange(4_500_000_000_000_000, 6_800_000_000_000_000))
        if k not in used and quickdraw_split_of(k) == split:
            used.add(k)
            return k


def write_quickdraw(out: Path, n_train: int, n_val: int, n_pool: int, seed: int = 0) -> Path:
    if n_val < MIN_QD_VAL or n_pool < MIN_QD_POOL or n_train < 10:
        raise ValueError(f"quickdraw needs train >= 10, val >= {MIN_QD_VAL}, pool >= {MIN_QD_POOL}")
    rng = random.Random(seed)
    d = out / "quickdraw" / "processed"
    d.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    splits: dict[str, list[dict]] = {}
    for split, n in (("train", n_train), ("val", n_val), ("pool", n_pool)):
        recs = [{"c": i % 10, "k": _key_for_split(rng, split, used), "d": _fake_drawing(rng)} for i in range(n)]
        rng.shuffle(recs)
        splits[split] = recs
        with (d / f"{split}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    pool = splits["pool"]
    by_class = {c: [r for r in pool if r["c"] == c] for c in range(10)}
    duel = [r for c in range(10) for r in rng.sample(by_class[c], QUICKDRAW_DUEL_PER_CLASS)]
    rng.shuffle(duel)
    used_keys = {r["k"] for r in duel}
    rest = [r for r in pool if r["k"] not in used_keys]
    probe = [rng.choice([r for r in rest if r["c"] == c]) for c in range(10)]
    probe += rng.sample([r for r in rest if r not in probe], QUICKDRAW_PROBE_N - 10)
    rng.shuffle(probe)
    seen = [0] * 10
    samples = []
    for r in splits["val"]:
        if seen[r["c"]] < QUICKDRAW_SAMPLES_PER_CLASS:
            samples.append(r)
            seen[r["c"]] += 1

    (d / "classes.json").write_text(json.dumps(QUICKDRAW_CLASSES))
    for name, recs in (("duel", duel), ("probe", probe), ("samples", samples)):
        (d / f"{name}.json").write_text(json.dumps(recs, separators=(",", ":")))
    return d


# ---------------------------------------------------------------- Speech Commands

def _fake_clip(rng: np.random.Generator, label: int) -> tuple[np.ndarray, int]:
    """One int16 clip of 16000 samples and its true length (shorter clips are zero-padded at the end)."""
    length = SPEECH_LEN if rng.random() > 0.12 else int(rng.integers(SPEECH_LEN // 2, SPEECH_LEN))
    t = np.arange(length) / SPEECH_RATE
    freq = 200 + 90 * label
    sig = 0.3 * np.sin(2 * np.pi * freq * t) * np.hanning(length) + 0.02 * rng.standard_normal(length)
    clip = np.zeros(SPEECH_LEN, dtype=np.int16)
    clip[:length] = np.clip(sig * 32768, -32768, 32767).astype(np.int16)
    clip[:length][clip[:length] == 0] = 1  # keep the real part distinguishable from padding
    return clip, length


def write_speech(out: Path, n_train: int, n_val: int, n_test: int, seed: int = 0) -> Path:
    if n_test < MIN_SPEECH_TEST or min(n_train, n_val) < MIN_SPEECH_SPLIT:
        raise ValueError(f"speech needs test >= {MIN_SPEECH_TEST}, train/val >= {MIN_SPEECH_SPLIT}")
    rng = np.random.default_rng(seed)
    d = out / "speech" / "processed"
    (d / "samples").mkdir(parents=True, exist_ok=True)
    taken: set[str] = set()
    test_lengths: list[int] = []
    test_files: list[str] = []
    test_x = None
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        # Speaker-separated: every split draws from its own pool of speaker ids.
        speakers_pool = []
        while len(speakers_pool) < max(2, n // 4):
            s = f"{int(rng.integers(0, 2**32)):08x}"
            if s not in taken:
                taken.add(s)
                speakers_pool.append(s)
        X = np.zeros((n, SPEECH_LEN), dtype=np.int16)
        y = np.zeros(n, dtype=np.int8)
        files, speakers, counter = [], [], {}
        for i in range(n):
            label = i % len(SPEECH_CLASSES)  # includes label 10 = marvin
            X[i], length = _fake_clip(rng, label)
            y[i] = label
            spk = speakers_pool[int(rng.integers(0, len(speakers_pool)))]
            k = counter.get((label, spk), 0)
            counter[(label, spk)] = k + 1
            files.append(f"{SPEECH_CLASSES[label]}/{spk}_nohash_{k}.wav")
            speakers.append(spk)
            if split == "test":
                test_lengths.append(length)
        # Keep the real data's ordering: sorted by word, then by file name.
        order = sorted(range(n), key=lambda i: (int(y[i]), files[i].split("/")[1]))
        X, y = X[order], y[order]
        files = [files[i] for i in order]
        speakers = [speakers[i] for i in order]
        if split == "test":
            test_lengths = [test_lengths[i] for i in order]
            test_files, test_x = files, X
        np.save(d / f"X_{split}.npy", X)
        np.save(d / f"y_{split}.npy", y)
        (d / f"files_{split}.json").write_text(json.dumps(files))
        (d / f"speakers_{split}.json").write_text(json.dumps(speakers))

    samples = []
    for label in range(SPEECH_MARVIN):  # labels 0-9 only, no marvin
        picked = [i for i, f in enumerate(test_files) if f.startswith(SPEECH_CLASSES[label] + "/")]
        for i in picked[:SPEECH_SAMPLES_PER_WORD]:
            name = test_files[i].replace("/", "__")
            with wave.open(str(d / "samples" / name), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SPEECH_RATE)
                w.writeframes(test_x[i, : test_lengths[i]].astype("<i2").tobytes())
            samples.append({"file": name, "label": label, "word": SPEECH_CLASSES[label]})
    (d / "samples.json").write_text(json.dumps(samples))
    (d / "classes.json").write_text(json.dumps(SPEECH_CLASSES))
    (d / "LICENSE").write_text("FAKE DATA for tests. The real dataset is CC BY 4.0 (Speech Commands v0.02).\n")
    (d / "README.md").write_text("Synthetic tones generated by backend/tests/fake_data.py. Not real speech.\n")
    return d


# ---------------------------------------------------------------- Lichess

def _fake_position(rng: random.Random) -> tuple[list[int], int, int, int]:
    """Returns (board, stm, castle, ep) obeying the SPEC 4.3 conventions."""
    board = [0] * 64
    if rng.random() < 0.05:  # the start position, with all castling rights
        board = [10, 8, 9, 11, 12, 9, 8, 10] + [7] * 8 + [0] * 32 + [1] * 8 + [4, 2, 3, 5, 6, 3, 2, 4]
        return board, 1, 15, -1
    empty = list(range(64))
    rng.shuffle(empty)
    board[empty.pop()] = 6  # white king
    board[empty.pop()] = 12  # black king
    for _ in range(rng.randint(0, 24)):
        code = rng.choice([1, 1, 1, 2, 3, 4, 5, 7, 7, 7, 8, 9, 10, 11])
        spots = [s for s in empty if code not in (1, 7) or 8 <= s < 56]  # no pawns on rank 1/8
        if not spots:
            continue
        s = rng.choice(spots)
        empty.remove(s)
        board[s] = code
    stm = rng.randint(0, 1)
    # Castling bits only where king and rook stand on their home squares.
    castle = 0
    for bit, king_sq, king, rook_sq, rook in ((1, 60, 6, 63, 4), (2, 60, 6, 56, 4), (4, 4, 12, 7, 10), (8, 4, 12, 0, 10)):
        if board[king_sq] == king and board[rook_sq] == rook and rng.random() < 0.7:
            castle |= bit
    # En passant only behind a pawn that could just have moved two squares.
    row, pawn = (3, 7) if stm == 1 else (4, 1)
    files = [f for f in range(8) if board[row * 8 + f] == pawn]
    ep = rng.choice(files) if files and rng.random() < 0.3 else -1
    return board, stm, castle, ep


def write_lichess(out: Path, n: int, seed: int = 0) -> Path:
    if n < 2:
        raise ValueError("lichess needs n >= 2")
    rng = random.Random(seed)
    d = out / "lichess"
    d.mkdir(parents=True, exist_ok=True)
    boards = np.zeros((n, 64), np.uint8)
    stm = np.zeros(n, np.uint8)
    castle = np.zeros(n, np.uint8)
    ep = np.full(n, -1, np.int8)
    cp = np.zeros(n, np.int16)
    mate = np.zeros(n, np.int16)
    depth = np.zeros(n, np.int16)
    for i in range(n):
        b, stm[i], castle[i], ep[i] = _fake_position(rng)
        boards[i] = b
        if rng.random() < 0.12:
            mate[i] = rng.choice([-1, 1]) * rng.randint(1, 40)  # positive = White mates; cp stays 0
        else:
            cp[i] = max(-10_000, min(10_000, int(rng.choice([-1, 1]) * rng.expovariate(1 / 150))))
        depth[i] = rng.randint(1, 60)
    mat = np.clip(LICHESS_MATERIAL[boards].sum(axis=1), -127, 127).astype(np.int8)
    npc = (boards > 0).sum(axis=1).astype(np.uint8)
    is_val = np.array([rng.random() < 0.02 for _ in range(n)], np.uint8)
    is_val[0], is_val[-1] = 0, 1  # always at least one row of each
    shard_of = (np.arange(n) >= n // 2).astype(np.uint8)
    for name, arr in (("boards", boards), ("stm", stm), ("castle", castle), ("ep", ep), ("cp", cp),
                      ("mate", mate), ("depth", depth), ("mat", mat), ("npc", npc), ("is_val", is_val),
                      ("shard_of", shard_of)):
        np.save(d / f"{name}.npy", arr)
    (d / "meta.json").write_text(json.dumps({
        "fake": True,
        "kept": n,
        "seed": seed,
        "convention": "cp and mate are White's view; boards index=row*8+col, row0=rank8; pieces 1-6 PNBRQK white, 7-12 black",
        "selection": "synthetic positions from backend/tests/fake_data.py",
    }, indent=2))
    return d


# ---------------------------------------------------------------- everything

def make_fake_data(root: Path | str, n: int = 100, seed: int = 0, *,
                   quickdraw: tuple[int, int, int] | None = None,
                   speech: tuple[int, int, int] | None = None,
                   lichess: int | None = None) -> Path:
    """Write all three datasets under `root` (the DATA_DIR layout). Returns root.

    n scales everything; pass explicit (train, val, pool/test) tuples to override.
    """
    root = Path(root)
    qd = quickdraw or (max(10, n), max(MIN_QD_VAL, n // 5), max(MIN_QD_POOL, n // 20))
    sp = speech or (max(MIN_SPEECH_SPLIT, n), max(MIN_SPEECH_SPLIT, n // 8), max(MIN_SPEECH_TEST, n // 8))
    write_quickdraw(root, *qd, seed=seed)
    write_speech(root, *sp, seed=seed)
    write_lichess(root, lichess if lichess is not None else max(2, 10 * n), seed=seed)
    return root


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Write tiny fake datasets in the SPEC section 4 formats.")
    ap.add_argument("--out", type=Path, required=True, help="directory to use as DATA_DIR")
    ap.add_argument("--n", type=int, default=100, help="size scale (default 100)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    make_fake_data(args.out, args.n, args.seed)
    print(f"fake data written to {args.out}")


if __name__ == "__main__":
    main()
