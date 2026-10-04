"""Will one more competition job fit on the GPU? (MAX_CONCURRENT_JOBS is a ceiling, not a promise.)

`estimate_job_vram` is a deliberately generous estimate of one job's peak memory from its resolved config; `admit` compares it with what the GPU has
free right now AND with what the jobs that were just started have not allocated yet. Nothing here touches CUDA: the caller passes the numbers
(`torch.cuda.mem_get_info()`), so it is testable anywhere.
"""

from __future__ import annotations

CUDA_CONTEXT_BYTES = 800 * 2**20  # CUDA context + cuBLAS workspaces + allocator slack of one process
SAFETY = 1.3  # headroom on top of the estimate
EVAL_CHUNK = 8192  # validation batch (CompetitionTrainer.evaluate)
PER_SAMPLE_INPUT_BYTES = 64 * 13 * 8 + 64 * 8 + 4096  # one_hot (int64) + the long copy of the board + the float planes / extras of the encoder


def estimate_job_vram(cfg: dict, data_resident_bytes: int = 0) -> int:
    """Peak bytes for one training process: weights, grads, optimizer state, EMA (+ its swap-in copy), activations of a training step or an evaluation chunk
    (whichever is bigger), the resident training data when it is loaded onto the GPU, and the per-process CUDA overhead."""
    params = int(cfg["param_count"])
    opt_state = {"sgd": 0, "momentum": 1, "adam": 2}.get(cfg.get("optimizer", "adam"), 2)
    copies = 2 + opt_state + (2 if cfg.get("ema", {}).get("enabled") else 0)  # weights + grads + optimizer state + EMA shadow + evaluate()'s backup
    state = params * 4 * copies
    widths = [int(w) for w in cfg["widths"]]
    per_sample = PER_SAMPLE_INPUT_BYTES + 4 * int(cfg["in_dim"]) * 3 + 4 * sum(widths) * (4 if cfg.get("residual") or cfg.get("normalization") not in (None, "none") else 3)
    batch = max(int(cfg["batch_size"]), EVAL_CHUNK)
    # colour-flip augmentation builds a second copy of the batch
    act = batch * per_sample * (2 if cfg.get("color_flip_augmentation") else 1)
    return int(SAFETY * (state + act + data_resident_bytes) + CUDA_CONTEXT_BYTES)


def admit(est: int, free_bytes: int, total_bytes: int, committed: int, headroom: float = 0.05) -> bool:
    """True when a job estimated at `est` bytes may start now.
    `free_bytes`: free VRAM right now. `committed`: the estimates of the jobs already running (a job that has just been started has not allocated its
    memory yet, so `free_bytes` alone would let a third job in on the strength of memory the first two are about to take)."""
    usable = total_bytes * (1 - headroom)
    return est <= free_bytes - total_bytes * headroom and committed + est <= usable
