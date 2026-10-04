"""Chess training data access (SPEC 4.3): device residency, slices, sampling, fixed validation set.

On a GPU the arrays are loaded as uint8/int16 tensors (about 1 GB for 12M positions) when they fit in a fraction of the
FREE VRAM (never a hardcoded size); otherwise, and on the CPU box, they stay memory-mapped and batches are gathered with
numpy. Index sampling always happens on a CPU generator so runs are reproducible and resumable across devices.
"""

from __future__ import annotations

import numpy as np
import torch

from app.chess_net import encoder as E
from app.data.loaders import LichessData

FIELDS = ("boards", "stm", "castle", "ep", "cp", "mate")


class Pool:
    def __init__(self, idx: torch.Tensor, cdf: torch.Tensor | None):
        self.idx, self.cdf = idx, cdf

    def __len__(self):
        return len(self.idx)


class ChessData:
    def __init__(self, lichess: LichessData, device: str | torch.device = "cpu", val_rows: int = 50_000,
                 max_resident_fraction: float = 0.4, force_resident: bool | None = None):
        self.li = lichess
        self.device = torch.device(device)
        self.val_rows = val_rows
        self._pools: dict[tuple, Pool] = {}
        self.gpu: dict[str, torch.Tensor] | None = None
        want = force_resident
        if want is None and self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            want = lichess.nbytes(FIELDS) < free * max_resident_fraction
        if want:
            self.gpu = {k: torch.from_numpy(np.array(lichess[k])).to(self.device) for k in FIELDS}
        v = lichess.val_indices()
        stride = max(1, len(v) // max(1, val_rows))
        self.val_idx = torch.from_numpy(v[::stride][:val_rows].astype(np.int64))

    @property
    def resident(self) -> bool:
        return self.gpu is not None

    def pool(self, data_slice: str, sampling: str, mate_clip: float = 2000.0, K: float = 400.0) -> Pool:
        key = (data_slice, sampling, mate_clip if sampling == "weighted" else 0, K if sampling == "weighted" else 0)
        if key not in self._pools:
            idx = torch.from_numpy(self.li.train_indices(data_slice).astype(np.int64))
            if len(idx) == 0:
                raise ValueError(f"data_slice {data_slice!r} has no training rows")
            cdf = None
            if sampling == "weighted":  # weight grows with |eval|: decisive positions are sampled more often
                cp = torch.from_numpy(np.asarray(self.li["cp"])[idx.numpy()].astype(np.float32))
                mate = torch.from_numpy(np.asarray(self.li["mate"])[idx.numpy()])
                w = 0.25 + E.effective_cp(cp, mate, mate_clip).abs() / K
                cdf = torch.cumsum(w.double(), dim=0)
            self._pools[key] = Pool(idx, cdf)
        return self._pools[key]

    def sample_indices(self, pool: Pool, batch: int, gen: torch.Generator) -> torch.Tensor:
        if pool.cdf is None:
            pos = torch.randint(len(pool), (batch,), generator=gen)
        else:
            u = torch.rand(batch, generator=gen, dtype=torch.float64) * pool.cdf[-1]
            pos = torch.searchsorted(pool.cdf, u).clamp_(max=len(pool) - 1)
        return pool.idx[pos]

    def fetch(self, idx: torch.Tensor, stager=None) -> dict[str, torch.Tensor]:
        """Row indices (cpu long) -> dict of tensors on the training device. `stager` (training.hostio.HostStager) makes the uploads non-blocking."""
        up = stager.upload if stager is not None else (lambda t: t.to(self.device))
        if self.gpu is not None:
            i = up(idx)
            return {k: self.gpu[k][i] for k in FIELDS}
        order = np.sort(idx.numpy())  # batch order is irrelevant; sorted reads are faster on mmap
        return {k: up(torch.from_numpy(np.asarray(self.li[k][order]))) for k in FIELDS}

    def val_batches(self, chunk: int = 8192, stager=None):
        for s in range(0, len(self.val_idx), chunk):
            yield self.fetch(self.val_idx[s:s + chunk], stager)
