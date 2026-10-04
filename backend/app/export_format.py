"""Browser-loadable model format (SPEC 9): ONE float32 .bin per model + a manifest entry describing the layers.

Supported layers (everything the showcase models use): dense, conv2d (stride 1, square kernel, same padding), relu, maxpool
(non-overlapping, k x k), flatten. The TypeScript twin is frontend/src/lib/inference.ts. `numpy_forward` here is the reference
used to prove the manifest means what we think (tests) and to produce golden outputs for the TS tests.
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn

from app.activations import Activation


def export_sequential(model: nn.Sequential) -> tuple[list[dict], np.ndarray]:
    """nn.Sequential -> (layer descriptors, flat float32 weights). Offsets are in FLOAT ELEMENTS into the .bin."""
    chunks: list[np.ndarray] = []
    pos = 0
    layers: list[dict] = []

    def put(t) -> dict:
        nonlocal pos
        a = t.detach().cpu().numpy().astype(np.float32).ravel()
        chunks.append(a)
        d = {"offset": pos, "shape": list(t.shape)}
        pos += a.size
        return d

    for m in model:
        if isinstance(m, nn.Linear):
            layers.append({"type": "dense", "in": m.in_features, "out": m.out_features, "w": put(m.weight), "b": put(m.bias)})
        elif isinstance(m, nn.Conv2d):
            assert m.stride == (1, 1) and m.kernel_size[0] == m.kernel_size[1] and m.padding[0] == m.kernel_size[0] // 2, "unsupported conv"
            layers.append({"type": "conv2d", "in_ch": m.in_channels, "out_ch": m.out_channels, "k": m.kernel_size[0],
                           "w": put(m.weight), "b": put(m.bias)})
        elif isinstance(m, nn.ReLU):
            layers.append({"type": "relu"})
        elif isinstance(m, Activation) and m.name == "relu":
            layers.append({"type": "relu"})
        elif isinstance(m, nn.MaxPool2d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            assert (m.stride in (None, k)) or m.stride == k
            layers.append({"type": "maxpool", "k": k})
        elif isinstance(m, nn.Flatten):
            layers.append({"type": "flatten"})
        else:
            raise ValueError(f"cannot export layer {type(m).__name__}")
    return layers, np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


def numpy_forward(layers: list[dict], weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Reference forward pass for ONE example. `x` is flat (dense models) or (C, H, W) (conv models)."""
    def get(d):
        return weights[d["offset"]:d["offset"] + int(np.prod(d["shape"]))].reshape(d["shape"])

    for L in layers:
        t = L["type"]
        if t == "dense":
            x = get(L["w"]) @ x.reshape(-1) + get(L["b"])
        elif t == "relu":
            x = np.maximum(x, 0)
        elif t == "flatten":
            x = x.reshape(-1)
        elif t == "maxpool":
            k = L["k"]
            c, h, w = x.shape
            x = x[:, : h // k * k, : w // k * k].reshape(c, h // k, k, w // k, k).max(axis=(2, 4))
        elif t == "conv2d":
            w, b, k = get(L["w"]), get(L["b"]), L["k"]
            c, h, wd = x.shape
            p = k // 2
            xp = np.pad(x, ((0, 0), (p, p), (p, p)))
            out = np.zeros((L["out_ch"], h, wd), np.float32)
            for dy in range(k):
                for dx in range(k):
                    out += np.einsum("oc,chw->ohw", w[:, :, dy, dx], xp[:, dy:dy + h, dx:dx + wd])
            x = out + b[:, None, None]
    return x
