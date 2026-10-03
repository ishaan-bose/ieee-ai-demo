"""The planned grid of race configurations (SPEC 9). Shared by record_races.py, benchmark.py and (mirrored in TypeScript)
frontend/src/lib/raceGrid.ts. Each entry is ONE lane; the frontend composes up to 3 recorded lanes into a race."""

from __future__ import annotations

BASE = {"loss": "ce", "lr": 0.03, "batch_size": 64, "optimizer": "momentum", "hidden": [128, 64], "activation": "relu"}
BASE_BATCH_RACE_LR = 0.02  # batch races use a fixed, moderate learning rate

LOSS_OPTIONS = [
    ("mse", {"loss": "mse"}), ("ce", {"loss": "ce"}), ("l1", {"loss": "l1"}),
    ("huber_d0.5", {"loss": "huber", "delta": 0.5}), ("huber_d1", {"loss": "huber", "delta": 1.0}), ("huber_d2", {"loss": "huber", "delta": 2.0}),
    ("focal_g0.5", {"loss": "focal", "gamma": 0.5}), ("focal_g1", {"loss": "focal", "gamma": 1.0}),
    ("focal_g2", {"loss": "focal", "gamma": 2.0}), ("focal_g5", {"loss": "focal", "gamma": 5.0}),
    ("ls_e0.05", {"loss": "label_smooth", "eps": 0.05}), ("ls_e0.1", {"loss": "label_smooth", "eps": 0.1}),
    ("ls_e0.3", {"loss": "label_smooth", "eps": 0.3}),
]
LRS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BATCHES = [1, 8, 32, 128, 512, "full"]


def lane_configs(kind: str) -> list[tuple[str, dict]]:
    """[(cache key, full lane config)] for kind in loss | lr | batch."""
    if kind == "loss":
        return [(f"loss__{k}", {**BASE, **cfg}) for k, cfg in LOSS_OPTIONS]
    if kind == "lr":
        return [(f"lr__{lr:g}", {**BASE, "lr": lr}) for lr in LRS]
    if kind == "batch":
        return [(f"batch__{b}", {**BASE, "lr": BASE_BATCH_RACE_LR, "batch_size": b}) for b in BATCHES]
    raise ValueError(kind)


def all_lane_configs() -> list[tuple[str, str, dict]]:
    return [(kind, key, cfg) for kind in ("loss", "lr", "batch") for key, cfg in lane_configs(kind)]
