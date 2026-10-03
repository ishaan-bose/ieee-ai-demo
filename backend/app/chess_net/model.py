"""The competition network: an MLP evaluator (SPEC 8.1, 8.3). Position features in, one raw score out."""

from __future__ import annotations

import torch
import torch.nn as nn

from app.activations import Activation


class EvalMLP(nn.Module):
    def __init__(self, in_dim: int, widths: list[int], *, activation: dict | None = None, normalization: str = "none",
                 residual: bool = False, init: str = "xavier"):
        super().__init__()
        act = activation or {}
        self.residual = residual
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList() if normalization == "layernorm" else None
        self.act = Activation(act.get("name", "relu"), act.get("alpha", 0.01), act.get("beta", 1.0), act.get("cap", 6.0))
        prev = in_dim
        for w in widths:
            self.linears.append(nn.Linear(prev, w))
            if self.norms is not None:
                self.norms.append(nn.LayerNorm(w))
            prev = w
        self.out = nn.Linear(prev, 1)
        self._init(init, act.get("name", "relu"))

    def _init(self, init: str, act_name: str) -> None:
        gain_name = "relu" if act_name in ("relu", "clipped_relu", "hard_step") else "leaky_relu" if act_name == "leaky_relu" else "tanh"
        for m in [*self.linears, self.out]:
            if init == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu" if gain_name == "relu" else "leaky_relu")
            elif init == "small_normal":
                nn.init.normal_(m.weight, std=0.02)
            else:
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(gain_name if gain_name != "leaky_relu" else "relu"))
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.linears):
            h = lin(x)
            if self.norms is not None:
                h = self.norms[i](h)
            h = self.act(h)
            x = x + h if (self.residual and i > 0 and x.shape[-1] == h.shape[-1]) else h
        return self.out(x).squeeze(-1)


def build_model(cfg: dict) -> EvalMLP:
    """From a RESOLVED config (see config.resolve_config)."""
    return EvalMLP(cfg["in_dim"], cfg["widths"], activation=cfg["activation"], normalization=cfg["normalization"],
                   residual=cfg["residual"], init=cfg["init"])
