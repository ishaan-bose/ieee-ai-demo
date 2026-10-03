"""Optimizers, LR schedules (defined over the FLOPs budget) and EMA (SPEC 6.2, 8.3)."""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch


def build_optimizer(params, name: str, lr: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def lr_factor(schedule: str, progress: float, warmup_frac: float = 0.0) -> float:
    """Multiplier on the base LR. `progress` = flops_used / budget in [0, 1], so every config gets a complete schedule."""
    p = min(max(progress, 0.0), 1.0)
    if warmup_frac > 0 and p < warmup_frac:
        return max(p / warmup_frac, 1e-3)
    rest = (p - warmup_frac) / (1.0 - warmup_frac) if warmup_frac < 1 else 1.0
    if schedule == "cosine":
        return 0.5 * (1 + math.cos(math.pi * rest))
    if schedule == "linear":
        return max(1.0 - rest, 0.0)
    return 1.0


class EMA:
    """Exponential moving average of the weights (with the usual warm-up of the decay)."""

    def __init__(self, params, decay: float):
        self.decay = decay
        self.n = 0
        self.shadow = [p.detach().clone() for p in params]

    @torch.no_grad()
    def update(self, params) -> None:
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        torch._foreach_mul_(self.shadow, d)
        torch._foreach_add_(self.shadow, [p.detach() for p in params], alpha=1 - d)

    @contextmanager
    def applied(self, params):
        """Temporarily swap the EMA weights into the model (for evaluation / saving)."""
        params = list(params)
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, s in zip(params, self.shadow):
                p.copy_(s)
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.copy_(b)

    def state(self) -> dict:
        return {"shadow": self.shadow, "n": self.n}

    def load(self, st: dict) -> None:
        self.shadow = [t.clone() for t in st["shadow"]]
        self.n = st["n"]
