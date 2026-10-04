"""Host -> device uploads that do not stall the GPU pipeline.

`tensor.to("cuda")` from ordinary (pageable) memory makes PyTorch wait for the whole CUDA stream before returning. In the training loop that
turned every step's tiny index upload into a full pipeline drain: the CPU could never queue step N+1 while the GPU was still busy with step N, so
the GPU sat mostly idle waiting for Python and kernel launches. `HostStager` copies the small tensor into a PINNED buffer of a small ring and
issues a non-blocking copy; an event per buffer guarantees a buffer is only overwritten after its previous copy has finished (waiting for that
one old copy, never for the whole stream). On the CPU it is a plain `.to()` (nothing to hide), so results are identical everywhere.
"""

from __future__ import annotations

import torch


class HostStager:
    RING = 4

    def __init__(self, device: str | torch.device, force_ring: bool = False):
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.active = self.cuda or force_ring  # force_ring: exercise the ring bookkeeping on the CPU (tests)
        self._rings: dict[tuple, list[list]] = {}
        self._next: dict[tuple, int] = {}

    def upload(self, t: torch.Tensor) -> torch.Tensor:
        if not self.active:
            return t.to(self.device)
        key = (tuple(t.shape), t.dtype)
        ring = self._rings.get(key)
        if ring is None:
            ring = self._rings[key] = [[torch.empty(t.shape, dtype=t.dtype, pin_memory=self.cuda), None] for _ in range(self.RING)]
            self._next[key] = 0
        slot = ring[self._next[key] % self.RING]
        self._next[key] += 1
        buf, ev = slot
        if ev is not None:
            ev.synchronize()  # only this buffer's previous copy, not the stream
        buf.copy_(t)
        if not self.cuda:
            return buf.clone()
        out = buf.to(self.device, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        slot[1] = ev
        return out
