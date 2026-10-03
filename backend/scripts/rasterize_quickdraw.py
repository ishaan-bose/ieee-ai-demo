#!/usr/bin/env python3
"""Stroke JSONL -> image tensors (SPEC 4.1). Idempotent and streaming.

Reads quickdraw/processed/{train,val,pool}.jsonl and writes quickdraw/tensors/{split}_x.npy (uint8 N x 28 x 28)
and {split}_y.npy (int8 N). Only chunks of lines are in memory (a few MB per worker), and output goes
straight into a memory-mapped .npy, so 1.09M drawings fit comfortably on a 16 GB box. Work is spread over
NUM_WORKERS processes (env var, default 4; never os.cpu_count()).

Rasterizer spec: shared/RASTERIZER.md. Re-running skips splits whose outputs are complete and newer than the input.

    cd ~/ieee-ai-demo/backend && python3 scripts/rasterize_quickdraw.py [--force] [--limit N] [--splits train val pool]
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.data.rasterizer import SIZE, rasterize  # noqa: E402
from scripts._common import setup_logging  # noqa: E402

CHUNK_LINES = 5000


def _work(lines: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    x = np.zeros((len(lines), SIZE, SIZE), np.uint8)
    y = np.zeros(len(lines), np.int8)
    for i, line in enumerate(lines):
        r = json.loads(line)
        x[i] = rasterize(r["d"])
        y[i] = r["c"]
    return x, y


def _chunks(path: Path, limit: int | None):
    buf: list[bytes] = []
    n = 0
    with path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            buf.append(line)
            n += 1
            if len(buf) == CHUNK_LINES:
                yield buf
                buf = []
            if limit is not None and n >= limit:
                break
    if buf:
        yield buf


def count_lines(path: Path, limit: int | None) -> int:
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
                if limit is not None and n >= limit:
                    break
    return n


def is_complete(src: Path, x_path: Path, y_path: Path, n: int) -> bool:
    if not (x_path.is_file() and y_path.is_file()):
        return False
    if min(x_path.stat().st_mtime, y_path.stat().st_mtime) < src.stat().st_mtime:
        return False
    try:
        x = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")
    except Exception:
        return False
    return x.shape == (n, SIZE, SIZE) and x.dtype == np.uint8 and y.shape == (n,) and y.dtype == np.int8


def rasterize_split(src: Path, out_dir: Path, split: str, workers: int, limit: int | None, force: bool, log) -> dict:
    x_path, y_path = out_dir / f"{split}_x.npy", out_dir / f"{split}_y.npy"
    n = count_lines(src, limit)
    if not force and is_complete(src, x_path, y_path, n):
        log.info("%s: up to date (%d drawings), skipping", split, n)
        return {"split": split, "n": n, "skipped": True}
    t0 = time.time()
    tmp_x, tmp_y = out_dir / f".{split}_x.tmp.npy", out_dir / f".{split}_y.tmp.npy"
    X = np.lib.format.open_memmap(tmp_x, mode="w+", dtype=np.uint8, shape=(n, SIZE, SIZE))
    Y = np.lib.format.open_memmap(tmp_y, mode="w+", dtype=np.int8, shape=(n,))
    done = 0
    pool = mp.Pool(workers) if workers > 1 else None

    def ordered_results():
        """Results in input order with at most 2 x workers chunks in flight (bounded memory)."""
        if pool is None:
            yield from map(_work, _chunks(src, limit))
            return
        pending = collections.deque()
        for chunk in _chunks(src, limit):
            pending.append(pool.apply_async(_work, (chunk,)))
            if len(pending) >= 2 * workers:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()

    try:
        for x, y in ordered_results():
            X[done:done + len(y)] = x
            Y[done:done + len(y)] = y
            done += len(y)
            if (done // CHUNK_LINES) % 20 == 0:
                log.info("%s: %d / %d (%.0fs)", split, done, n, time.time() - t0)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    assert done == n, f"{split}: rasterized {done} of {n}"
    X.flush(); Y.flush()
    del X, Y
    os.replace(tmp_x, x_path)  # atomic: a half-written file never carries the final name
    os.replace(tmp_y, y_path)
    ink = float(np.load(x_path, mmap_mode="r")[:: max(1, n // 2000)].mean())
    per_class = np.bincount(np.load(y_path), minlength=10).tolist()
    log.info("%s: wrote %d drawings in %.0fs, mean pixel %.1f, per class %s", split, n, time.time() - t0, ink, per_class)
    return {"split": split, "n": n, "seconds": round(time.time() - t0, 1), "mean_pixel": round(ink, 2), "per_class": per_class}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "pool"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N drawings of each split (testing)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="default: NUM_WORKERS env var (4)")
    args = ap.parse_args(argv)
    settings = get_settings()
    log = setup_logging("rasterize_quickdraw", settings)
    data_dir = args.data_dir or settings.data_dir
    workers = args.workers or settings.num_workers
    processed, out = data_dir / "quickdraw" / "processed", data_dir / "quickdraw" / "tensors"
    out.mkdir(parents=True, exist_ok=True)
    log.info("rasterizing %s -> %s with %d workers", processed, out, workers)
    results = [rasterize_split(processed / f"{s}.jsonl", out, s, workers, args.limit, args.force, log) for s in args.splits]
    print("SUMMARY rasterize_quickdraw:", json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
