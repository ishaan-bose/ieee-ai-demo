"""Losses. Regression losses for the chess evaluator (SPEC 8.3) and classification losses for the doodle races (SPEC 5.3)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

REGRESSION_LOSSES = ("mse", "huber", "bce", "l1")
CLASS_LOSSES = ("mse", "ce", "l1", "huber", "focal", "label_smooth")


def regression_loss(name: str, pred: torch.Tensor, y: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """pred, y in the same target space (see encoder.make_target)."""
    if name == "mse":
        return F.mse_loss(pred, y)
    if name == "l1":
        return F.l1_loss(pred, y)
    if name == "huber":
        return F.huber_loss(pred, y, delta=delta)
    if name == "bce":
        return F.binary_cross_entropy(pred.clamp(1e-6, 1 - 1e-6), y)
    raise ValueError(f"unknown loss {name!r}")


def class_loss(name: str, logits: torch.Tensor, y: torch.Tensor, *, delta: float = 1.0, gamma: float = 2.0,
               eps: float = 0.1) -> torch.Tensor:
    """Doodle-classification losses (y = int class labels). MSE / L1 / Huber act on output probabilities."""
    if name == "ce":
        return F.cross_entropy(logits, y)
    if name == "label_smooth":
        return F.cross_entropy(logits, y, label_smoothing=eps)
    if name == "focal":
        logp = F.log_softmax(logits, dim=1).gather(1, y[:, None])[:, 0]
        return (-((1 - logp.exp()) ** gamma) * logp).mean()
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(y, logits.shape[1]).to(probs.dtype)
    if name == "mse":
        return F.mse_loss(probs, onehot)
    if name == "l1":
        return F.l1_loss(probs, onehot)
    if name == "huber":
        return F.huber_loss(probs, onehot, delta=delta)
    raise ValueError(f"unknown class loss {name!r}")
