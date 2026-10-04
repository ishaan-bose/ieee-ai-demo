"""Small MLP classifiers (doodles, audio) shared by the race runner, the showcase trainer and the benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.activations import Activation


def build_mlp_classifier(in_dim: int, hidden: list[int], n_out: int, activation: dict | str = "relu", seed: int = 0) -> nn.Sequential:
    """Plain MLP (Linear + activation per hidden layer). Same seed -> identical initial weights (races rely on this)."""
    act = {"name": activation} if isinstance(activation, str) else activation
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        lin = nn.Linear(prev, h)
        nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5)
        layers += [lin, Activation(act.get("name", "relu"), act.get("alpha", 0.01), act.get("beta", 1.0), act.get("cap", 6.0))]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, chunk: int = 8192) -> float:
    was = model.training
    model.eval()
    correct = 0
    for s in range(0, x.shape[0], chunk):
        correct += (model(x[s:s + chunk]).argmax(1) == y[s:s + chunk]).sum().item()
    model.train(was)
    return correct / max(1, x.shape[0])


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> list[int]:
    was = model.training
    model.eval()
    out = model(x).argmax(1).tolist()
    model.train(was)
    return out
