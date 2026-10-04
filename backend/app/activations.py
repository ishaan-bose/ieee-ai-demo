"""Activation registry shared by the competition nets and the doodle race nets (mirrors the demo gallery, SPEC 5.3).

Data-driven: adding a function is a one-entry change in ACTIVATIONS. The browser twin is frontend/src/lib/activations.ts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# name -> (callable(x, alpha, beta, cap), has_param description)
ACTIVATIONS = {
    "relu": lambda x, a, b, c: F.relu(x),
    "leaky_relu": lambda x, a, b, c: F.leaky_relu(x, a),
    "sigmoid": lambda x, a, b, c: torch.sigmoid(x),
    "tanh": lambda x, a, b, c: torch.tanh(x),
    "gelu": lambda x, a, b, c: F.gelu(x),
    "swish": lambda x, a, b, c: x * torch.sigmoid(b * x),
    "softplus": lambda x, a, b, c: F.softplus(x, beta=b),
    "elu": lambda x, a, b, c: F.elu(x, alpha=a),
    "clipped_relu": lambda x, a, b, c: x.clamp(0.0, c),
    "hard_step": lambda x, a, b, c: (x > 0).to(x.dtype),  # slope is zero everywhere: cannot learn
}
ACTIVATION_NAMES = tuple(ACTIVATIONS)


class Activation(torch.nn.Module):
    def __init__(self, name: str = "relu", alpha: float = 0.01, beta: float = 1.0, cap: float = 6.0):
        super().__init__()
        if name not in ACTIVATIONS:
            raise ValueError(f"unknown activation {name!r}; choose from {ACTIVATION_NAMES}")
        self.name, self.alpha, self.beta, self.cap = name, alpha, beta, cap
        self._fn = ACTIVATIONS[name]

    def forward(self, x):
        return self._fn(x, self.alpha, self.beta, self.cap)

    def extra_repr(self) -> str:
        return f"{self.name}(alpha={self.alpha}, beta={self.beta}, cap={self.cap})"
