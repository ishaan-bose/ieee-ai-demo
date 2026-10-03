"""Audio showcase models (SPEC 5.3 Act 3): a raw-waveform MLP (expected to flail) and a small log-mel CNN (expected to work).
Both are small enough for a plain TypeScript forward pass in the browser (frontend/src/lib/inference.ts)."""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.data import audio_features as af


def build_raw_model(n_out: int = 10, seed: int = 0) -> nn.Sequential:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(af.RAW_LEN, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))


def build_logmel_model(n_out: int = 10, seed: int = 0) -> nn.Sequential:
    """Input (B, 1, 40, 98). conv3x3(8)+ReLU+pool2 -> conv3x3(16)+ReLU+pool2 -> flatten(16*10*24) -> 64 -> n_out."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(16 * 10 * 24, 64), nn.ReLU(), nn.Linear(64, n_out))


def raw_features(x_int16: torch.Tensor) -> torch.Tensor:
    """(N, 16000) int16 -> (N, 4000) float32: x/32768 average-pooled by 4 (AUDIO_FEATURES.md)."""
    return (x_int16.float() / 32768.0).reshape(x_int16.shape[0], af.RAW_LEN, af.RAW_POOL).mean(dim=2)


def logmel_features(x_int16: torch.Tensor) -> torch.Tensor:
    """(N, 16000) int16 -> (N, 1, 40, 98) float16 (stored compactly; cast to float per batch)."""
    return af.log_mel_batch_torch(x_int16).unsqueeze(1).half()


def make_features(kind: str, X: np.ndarray, device="cpu", chunk: int = 4096) -> torch.Tensor:
    fn = raw_features if kind == "raw" else logmel_features
    return torch.cat([fn(torch.from_numpy(np.ascontiguousarray(X[s:s + chunk])).to(device)).cpu() for s in range(0, len(X), chunk)])


def train_audio(model: nn.Module, xtr, ytr, xva, yva, *, epochs: int, device="cpu", batch: int = 128, lr: float = 1e-3,
                seed: int = 0, on_epoch=None, weight=None, max_seconds: float | None = None) -> list[dict]:
    """Plain Adam training. Returns a per-epoch history of train/val loss and accuracy."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    ytr_t, yva_t = torch.as_tensor(ytr).long(), torch.as_tensor(yva).long()
    w = None if weight is None else torch.as_tensor(weight, dtype=torch.float32, device=device)
    hist, t0 = [], time.time()

    def evaluate(x, y):
        model.eval()
        loss = correct = 0.0
        with torch.no_grad():
            for s in range(0, len(y), 4096):
                out = model(x[s:s + 4096].to(device).float())
                loss += F.cross_entropy(out, y[s:s + 4096].to(device), weight=w, reduction="sum").item()
                correct += (out.argmax(1).cpu() == y[s:s + 4096]).sum().item()
        model.train()
        return loss / len(y), correct / len(y)

    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(ytr_t), generator=g)
        for s in range(0, len(perm), batch):
            i = perm[s:s + batch]
            loss = F.cross_entropy(model(xtr[i].to(device).float()), ytr_t[i].to(device), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        tl, ta = evaluate(xtr, ytr_t)
        vl, va = evaluate(xva, yva_t)
        rec = {"epoch": ep, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va, "seconds": round(time.time() - t0, 1)}
        hist.append(rec)
        if on_epoch:
            on_epoch(rec)
        if max_seconds is not None and time.time() - t0 > max_seconds:
            break
    return hist
