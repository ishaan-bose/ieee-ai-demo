"""Competition model config (SPEC 8.3): schema, ranges, defaults, parameter counting, tiers.

`resolve_config(raw)` validates a user config server-side (every knob range-checked), fills defaults for
anything missing, ignores unknown/removed knobs silently, and returns (resolved dict, errors). The resolved dict is
plain JSON and is what is stored in the DB and in models/<id>/config.json.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.activations import ACTIVATION_NAMES
from app.chess_net.encoder import Extras, feature_dim

MAX_PARAMS = 30_000_000
MAX_LAYERS = 16
MIN_WIDTH, MAX_WIDTH = 8, 8192

ActivationName = Literal[ACTIVATION_NAMES]  # type: ignore[valid-type]


class ActivationCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: ActivationName = "relu"
    alpha: float = Field(0.01, ge=0.0, le=1.0)  # leaky_relu slope, elu alpha
    beta: float = Field(1.0, ge=0.1, le=10.0)  # swish, softplus
    cap: float = Field(6.0, ge=0.5, le=20.0)  # clipped_relu


class LossCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: Literal["mse", "huber", "bce", "l1"] = "mse"
    delta: float = Field(1.0, ge=0.01, le=10.0)


class ExtrasCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stm_castle: bool = True
    en_passant: bool = False
    material: bool = False
    attacks: bool = False


class EmaCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = False
    decay: float = Field(0.999, ge=0.9, le=0.99999)


class CompetitionConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # core knobs
    layers: int = Field(3, ge=1, le=MAX_LAYERS)
    width: int = Field(256, ge=MIN_WIDTH, le=MAX_WIDTH)
    layer_widths: list[int] | None = None  # overrides layers and width
    activation: ActivationCfg = ActivationCfg()
    loss: LossCfg = LossCfg()
    lr: float = Field(1e-3, ge=1e-6, le=1.0)
    batch_size: int = Field(256, ge=1, le=16384)
    optimizer: Literal["sgd", "momentum", "adam"] = "adam"
    # advanced knobs
    target_type: Literal["winprob", "cp"] = "winprob"
    eval_squash_scale: float = Field(400.0, ge=50.0, le=2000.0)
    mate_clip: float = Field(2000.0, ge=300.0, le=10000.0)
    input_extras: ExtrasCfg = ExtrasCfg()
    perspective_flip: bool = True
    color_flip_augmentation: bool = False
    data_slice: Literal["all", "endgame", "balanced", "decisive"] = "all"
    sampling: Literal["uniform", "weighted"] = "uniform"
    init: Literal["xavier", "he", "small_normal"] = "xavier"
    lr_schedule: Literal["constant", "cosine", "linear"] = "constant"
    warmup_frac: float = Field(0.0, ge=0.0, le=0.5)
    grad_clip: float = Field(0.0, ge=0.0, le=100.0)  # 0 = off
    ema: EmaCfg = EmaCfg()
    normalization: Literal["none", "layernorm"] = "none"
    residual: bool = False
    output_head: Literal["linear", "tanh", "sigmoid"] = "linear"
    seed: int = Field(0, ge=0, le=2**31 - 1)


DEFAULT_CONFIG: dict = CompetitionConfig().model_dump()  # sensible but unremarkable: a working, mediocre bot


# ---------------------------------------------------------------- parameter counting

def count_params(in_dim: int, widths: list[int], normalization: str = "none") -> int:
    """Total parameters (weights + biases + norm affine) of the MLP with a single output unit."""
    total, prev = 0, in_dim
    for w in widths:
        total += prev * w + w
        if normalization == "layernorm":
            total += 2 * w
        prev = w
    return total + prev + 1


def matmul_params(in_dim: int, widths: list[int]) -> int:
    """Weights that take part in matrix multiplications (biases/norms excluded), for FLOPs = 6 * this * samples."""
    total, prev = 0, in_dim
    for w in widths:
        total += prev * w
        prev = w
    return total + prev


def flops_per_sample(in_dim: int, widths: list[int]) -> int:
    return 6 * matmul_params(in_dim, widths)


# ---------------------------------------------------------------- tiers

@lru_cache(maxsize=1)
def _tiers_file() -> dict:
    path = Path(os.environ.get("TIERS_FILE") or Path(__file__).resolve().parents[3] / "shared" / "tiers.json")
    return json.loads(path.read_text())


def tier_table() -> list[dict]:
    return _tiers_file()["tiers"]


def tier_for(param_count: int) -> dict:
    for t in tier_table():
        if param_count <= t["max_params"]:
            return t
    return tier_table()[-1]


# ---------------------------------------------------------------- resolve

def resolve_config(raw: dict | None) -> tuple[dict | None, list[str]]:
    """Validate and fill defaults. Returns (resolved, errors). `resolved` is None when invalid."""
    errors: list[str] = []
    try:
        cfg = CompetitionConfig.model_validate(raw or {})
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return None, errors
    d = cfg.model_dump()
    if cfg.layer_widths is not None:
        if not 1 <= len(cfg.layer_widths) <= MAX_LAYERS:
            errors.append(f"layer_widths: need between 1 and {MAX_LAYERS} entries")
        if any(not MIN_WIDTH <= w <= MAX_WIDTH for w in cfg.layer_widths):
            errors.append(f"layer_widths: each width must be between {MIN_WIDTH} and {MAX_WIDTH}")
        widths = list(cfg.layer_widths)
    else:
        widths = [cfg.width] * cfg.layers
    if errors:
        return None, errors
    in_dim = feature_dim(Extras(**d["input_extras"]))
    n = count_params(in_dim, widths, cfg.normalization)
    if n > MAX_PARAMS:
        errors.append(f"too many parameters: {n:,} (the limit is {MAX_PARAMS:,})")
    if cfg.loss.name == "bce" and cfg.target_type != "winprob":
        errors.append("loss 'bce' needs target_type 'winprob'")
    if errors:
        return None, errors
    tier = tier_for(n)
    d.update({"widths": widths, "layers": len(widths), "in_dim": in_dim, "param_count": n,
              "matmul_params": matmul_params(in_dim, widths), "tier": tier["name"],
              "search_depth_full_moves": tier["search_depth_full_moves"]})
    return d, []
