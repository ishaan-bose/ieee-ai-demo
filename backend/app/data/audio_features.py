"""Log-mel features. Spec: shared/AUDIO_FEATURES.md (TypeScript twin: frontend/src/lib/audio.ts)."""

from __future__ import annotations

import numpy as np

SR = 16_000
N_SAMPLES = 16_000
WIN = 400
HOP = 160
NFFT = 512
N_MELS = 40
FMIN = 20.0
FMAX = 7600.0
LOG_FLOOR = 1e-6
NORM_EPS = 1e-5
N_FRAMES = 1 + (N_SAMPLES - WIN) // HOP  # 98
RAW_POOL = 4  # raw-waveform model input: average-pool by 4 -> 4000 values
RAW_LEN = N_SAMPLES // RAW_POOL


def _mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _inv_mel(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def hann_window() -> np.ndarray:
    n = np.arange(WIN)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / WIN)


def mel_filterbank() -> np.ndarray:
    """(N_MELS, NFFT//2 + 1) float64, triangular, HTK mel, no area normalization."""
    edges = _inv_mel(np.linspace(_mel(FMIN), _mel(FMAX), N_MELS + 2))
    freqs = np.arange(NFFT // 2 + 1) * SR / NFFT
    fb = np.zeros((N_MELS, NFFT // 2 + 1))
    for m in range(N_MELS):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / (ce - lo)
        down = (hi - freqs) / (hi - ce)
        fb[m] = np.where((freqs >= lo) & (freqs <= ce), up, np.where((freqs > ce) & (freqs <= hi), down, 0.0))
    return fb


_WINDOW = hann_window()
_FB = mel_filterbank()


def pad_or_trim(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if len(x) >= N_SAMPLES:
        return x[:N_SAMPLES]
    return np.concatenate([x, np.zeros(N_SAMPLES - len(x), x.dtype)])


def to_float(x_int16: np.ndarray) -> np.ndarray:
    return np.asarray(x_int16, dtype=np.float32) / 32768.0


def log_mel(x: np.ndarray) -> np.ndarray:
    """One clip (float in [-1,1], or int16) -> float32 (40, 98), normalized per clip."""
    x = np.asarray(x)
    if x.dtype.kind == "i":
        x = to_float(x)
    x = pad_or_trim(x.astype(np.float64))
    idx = np.arange(N_FRAMES)[:, None] * HOP + np.arange(WIN)[None, :]
    frames = x[idx] * _WINDOW  # (98, 400)
    power = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2  # (98, 257)
    mel = _FB @ power.T  # (40, 98)
    logm = np.log(mel + LOG_FLOOR)
    return ((logm - logm.mean()) / (logm.std() + NORM_EPS)).astype(np.float32)


def log_mel_batch_torch(x, chunk: int = 2048):
    """Batched torch twin. x: (N, 16000) int16 or float tensor -> float32 (N, 40, 98). Same math as `log_mel`."""
    import torch

    dev = x.device
    win = torch.as_tensor(_WINDOW, dtype=torch.float32, device=dev)
    fb = torch.as_tensor(_FB, dtype=torch.float32, device=dev)
    out = []
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]
        xb = xb.float() / 32768.0 if xb.dtype in (torch.int16, torch.int32, torch.int8) else xb.float()
        frames = xb.unfold(1, WIN, HOP)[:, :N_FRAMES] * win  # (B, 98, 400)
        power = torch.fft.rfft(frames, n=NFFT, dim=2).abs() ** 2  # (B, 98, 257)
        mel = torch.einsum("mk,btk->bmt", fb, power)  # (B, 40, 98)
        logm = torch.log(mel + LOG_FLOOR)
        mean = logm.mean(dim=(1, 2), keepdim=True)
        std = logm.std(dim=(1, 2), keepdim=True, unbiased=False)
        out.append((logm - mean) / (std + NORM_EPS))
    return torch.cat(out)


def raw_input(x_int16: np.ndarray) -> np.ndarray:
    """Raw-waveform model input: float, average-pooled by 4 -> (4000,)."""
    x = pad_or_trim(to_float(x_int16))
    return x.reshape(RAW_LEN, RAW_POOL).mean(axis=1).astype(np.float32)


def read_wav(path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
