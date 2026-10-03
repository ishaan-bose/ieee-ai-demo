"""Stroke JSON -> 28x28 uint8 image. Spec: shared/RASTERIZER.md (TypeScript twin: frontend/src/lib/rasterizer.ts)."""

from __future__ import annotations

import numpy as np

SIZE = 28
MARGIN = 2
LINE_WIDTH = 2.0
NORM = 255.0

_ys, _xs = np.mgrid[0:SIZE, 0:SIZE]
_PX = _xs.ravel() + 0.5  # pixel centers, x and y (row-major, 784 each)
_PY = _ys.ravel() + 0.5


def normalize_strokes(strokes) -> list[np.ndarray]:
    """Step 1+2 of the spec: returns one (n_points, 2) float array per stroke, in canvas coordinates."""
    pts = [np.stack([np.asarray(s[0], dtype=np.float64), np.asarray(s[1], dtype=np.float64)], axis=1) for s in strokes]
    pts = [p for p in pts if len(p)]
    if not pts:
        return []
    allp = np.concatenate(pts)
    lo = allp.min(axis=0)
    extent = float((allp.max(axis=0) - lo).max())
    if extent == 0:
        extent = 1.0
    scale = (SIZE - 2 * MARGIN) / NORM
    return [(p - lo) * (NORM / extent) * scale + MARGIN for p in pts]


def _segments(norm: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a, b = [], []
    for p in norm:
        if len(p) == 1:
            a.append(p[0:1]); b.append(p[0:1])
        else:
            a.append(p[:-1]); b.append(p[1:])
    return np.concatenate(a), np.concatenate(b)


def rasterize(strokes) -> np.ndarray:
    """One drawing -> uint8 (28, 28)."""
    norm = normalize_strokes(strokes)
    if not norm:
        return np.zeros((SIZE, SIZE), np.uint8)
    a, b = _segments(norm)  # (S, 2) each
    ax, ay = a[:, 0:1], a[:, 1:2]
    abx, aby = b[:, 0:1] - ax, b[:, 1:2] - ay
    denom = np.maximum(abx * abx + aby * aby, 1e-12)  # (S, 1)
    apx, apy = _PX[None, :] - ax, _PY[None, :] - ay  # (S, 784)
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    dx, dy = apx - t * abx, apy - t * aby
    # coverage is monotone decreasing in distance, so the union (max coverage) is
    # the coverage of the smallest distance: take the min first, one sqrt per pixel.
    dist = np.sqrt((dx * dx + dy * dy).min(axis=0))
    cov = np.clip(LINE_WIDTH / 2 + 0.5 - dist, 0.0, 1.0)
    return np.floor(255.0 * cov + 0.5).astype(np.uint8).reshape(SIZE, SIZE)


def rasterize_many(records) -> tuple[np.ndarray, np.ndarray]:
    """Records `{"c":..,"d":..}` -> (uint8 (N,28,28), int8 (N,))."""
    x = np.zeros((len(records), SIZE, SIZE), np.uint8)
    y = np.zeros(len(records), np.int8)
    for i, r in enumerate(records):
        x[i] = rasterize(r["d"])
        y[i] = r["c"]
    return x, y
