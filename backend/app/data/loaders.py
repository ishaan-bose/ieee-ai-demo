"""Loaders for the SPEC section 4 formats.

The same code reads fake data (tests/fake_data.py), the committed samples (data/samples/)
and the real files under DATA_DIR. Big arrays are memory-mapped (`mmap_mode="r"`); nothing here
loads a whole dataset into RAM unless asked (`to_device`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from app.data import audio_features as af
from app.data.contract import QUICKDRAW_CLASSES, SPEECH_CLASSES, SPEECH_MARVIN
from app.data.rasterizer import rasterize_many

# ---------------------------------------------------------------- Quick, Draw!


class QuickDrawData:
    """`root` = DATA_DIR/quickdraw (has processed/ and tensors/) or data/samples/quickdraw (flat)."""

    def __init__(self, root: Path | str):
        root = Path(root)
        self.processed = root / "processed" if (root / "processed").is_dir() else root
        self.tensors_dir = root / "tensors"
        self.classes: list[str] = json.loads((self.processed / "classes.json").read_text())
        assert self.classes == QUICKDRAW_CLASSES

    def iter_jsonl(self, split: str) -> Iterator[dict]:
        with (self.processed / f"{split}.jsonl").open("rb") as f:
            for line in f:
                yield json.loads(line)

    def json_records(self, name: str) -> list[dict]:
        """duel / probe / samples."""
        return json.loads((self.processed / f"{name}.json").read_text())

    def has_tensors(self, splits=("train", "val", "pool")) -> bool:
        return all((self.tensors_dir / f"{s}_{k}.npy").is_file() for s in splits for k in ("x", "y"))

    def tensors(self, split: str, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(uint8 (N,28,28), int8 (N,)) from quickdraw/tensors (built by scripts/rasterize_quickdraw.py)."""
        mode = "r" if mmap else None
        return (np.load(self.tensors_dir / f"{split}_x.npy", mmap_mode=mode),
                np.load(self.tensors_dir / f"{split}_y.npy", mmap_mode=mode))

    def rasterized(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Rasterize duel / probe / samples on the fly (they are tiny)."""
        return rasterize_many(self.json_records(name))


# ---------------------------------------------------------------- Speech Commands


class SpeechData:
    """`root` = DATA_DIR/speech/processed, or data/samples/speech (only samples/classes available)."""

    def __init__(self, root: Path | str):
        root = Path(root)
        self.root = root / "processed" if (root / "processed").is_dir() else root
        self.classes: list[str] = json.loads((self.root / "classes.json").read_text())
        assert self.classes == SPEECH_CLASSES

    def X(self, split: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.root / f"X_{split}.npy", mmap_mode="r" if mmap else None)

    def y(self, split: str) -> np.ndarray:
        return np.load(self.root / f"y_{split}.npy")

    def files(self, split: str) -> list[str]:
        return json.loads((self.root / f"files_{split}.json").read_text())

    def speakers(self, split: str) -> list[str]:
        return json.loads((self.root / f"speakers_{split}.json").read_text())

    def word_indices(self, split: str) -> np.ndarray:
        """Row indices of the ten command words (label < 10): `marvin` is excluded from the 10-word classifier."""
        return np.flatnonzero(self.y(split) < SPEECH_MARVIN)

    def marvin_subset(self, split: str, pos_fraction: float = 0.02, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Metrics-trap subset (SPEC 4.2): keep ALL negatives, subsample `marvin` positives to ~pos_fraction.

        Returns (row indices sorted, binary labels 1 = marvin).
        """
        y = self.y(split)
        neg = np.flatnonzero(y != SPEECH_MARVIN)
        pos = np.flatnonzero(y == SPEECH_MARVIN)
        n_pos = min(len(pos), max(1, int(round(len(neg) * pos_fraction / (1 - pos_fraction)))))
        pos = np.random.default_rng(seed).choice(pos, n_pos, replace=False)
        idx = np.sort(np.concatenate([neg, pos]))
        return idx, (y[idx] == SPEECH_MARVIN).astype(np.int8)

    def overfit_subset(self, n: int = 300, seed: int = 0) -> np.ndarray:
        """Tiny fixed-seed subset of train (10 words only) for the overfitting demo."""
        idx = self.word_indices("train")
        return np.sort(np.random.default_rng(seed).choice(idx, min(n, len(idx)), replace=False))

    def samples(self) -> list[dict]:
        """The 20 committed .wav clips: [{file,label,word,x (int16, 16000)}]."""
        out = []
        for s in json.loads((self.root / "samples.json").read_text()):
            x = af.pad_or_trim(af.read_wav(self.root / "samples" / s["file"]))
            out.append({**s, "x": x})
        return out

    @staticmethod
    def to_float(x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) / 32768.0


# ---------------------------------------------------------------- Lichess

LICHESS_NAMES = ("boards", "stm", "castle", "ep", "cp", "mate", "depth", "mat", "npc", "is_val", "shard_of")
DATA_SLICES = ("all", "endgame", "balanced", "decisive")


class LichessData:
    """`root` = DATA_DIR/lichess or data/samples/lichess. All arrays memory-mapped."""

    def __init__(self, root: Path | str, mmap: bool = True):
        self.root = Path(root)
        self.a = {n: np.load(self.root / f"{n}.npy", mmap_mode="r" if mmap else None) for n in LICHESS_NAMES}
        self.meta = json.loads((self.root / "meta.json").read_text())
        self.n = int(self.a["boards"].shape[0])
        self._cache: dict[str, np.ndarray] = {}

    def __getitem__(self, name: str) -> np.ndarray:
        return self.a[name]

    def _chunks(self, rows: int = 1_000_000):
        for s in range(0, self.n, rows):
            yield s, slice(s, min(self.n, s + rows))

    def val_indices(self) -> np.ndarray:
        if "val" not in self._cache:
            self._cache["val"] = np.flatnonzero(np.asarray(self.a["is_val"]) == 1)
        return self._cache["val"]

    def train_indices(self, data_slice: str = "all") -> np.ndarray:
        """Training row indices (is_val == 0) restricted to a data slice (SPEC 8.3).

        endgame: npc <= 10; balanced: no mate and |cp| <= 100; decisive: mate or |cp| >= 300.
        Uses only cp/mate/npc (never the stored `mat`).
        """
        if data_slice not in DATA_SLICES:
            raise ValueError(f"data_slice must be one of {DATA_SLICES}")
        key = f"train:{data_slice}"
        if key not in self._cache:
            parts = []
            for s, sl in self._chunks():
                keep = np.asarray(self.a["is_val"][sl]) == 0
                if data_slice == "endgame":
                    keep &= np.asarray(self.a["npc"][sl]) <= 10
                elif data_slice in ("balanced", "decisive"):
                    cp = np.asarray(self.a["cp"][sl]).astype(np.int32)
                    mate = np.asarray(self.a["mate"][sl])
                    if data_slice == "balanced":
                        keep &= (mate == 0) & (np.abs(cp) <= 100)
                    else:
                        keep &= (mate != 0) | (np.abs(cp) >= 300)
                parts.append(np.flatnonzero(keep) + s)
            self._cache[key] = np.concatenate(parts) if parts else np.zeros(0, np.int64)
        return self._cache[key]

    def gather(self, idx: np.ndarray, names=("boards", "stm", "castle", "ep", "cp", "mate")) -> dict[str, np.ndarray]:
        idx = np.asarray(idx)
        return {n: np.asarray(self.a[n][idx]) for n in names}

    def nbytes(self, names=("boards", "stm", "castle", "ep", "cp", "mate")) -> int:
        return sum(int(self.a[n].nbytes) for n in names)
