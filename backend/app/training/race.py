"""Demo races (SPEC 5.3 Act 2, 6.1): up to 3 lanes train the same network on the same data, interleaved in short
slices so wall-clock fairness is automatic. Full-batch lanes use gradient accumulation in chunks sized from free VRAM
(math identical to a true full batch). Emits `tick` / `done` events exactly as the SSE contract (SPEC 7)."""

from __future__ import annotations

import time
from typing import Callable, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.data.loaders import QuickDrawData
from app.training.classifier import accuracy, build_mlp_classifier, predict
from app.training.losses import CLASS_LOSSES, class_loss

SLICE_S = 0.15
TICK_S = 0.5
VAL_N = 2000
RACE_KINDS = ("loss", "lr", "batch")


class LaneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(min_length=1, max_length=40)
    loss: Literal["mse", "ce", "l1", "huber", "focal", "label_smooth"] = "ce"
    delta: float = Field(1.0, ge=0.01, le=10.0)
    gamma: float = Field(2.0, ge=0.0, le=10.0)
    eps: float = Field(0.1, ge=0.0, le=0.9)
    lr: float = Field(0.03, ge=1e-5, le=10.0)
    batch_size: int | Literal["full"] = 64
    optimizer: Literal["sgd", "momentum", "adam"] = "momentum"
    hidden: list[int] = [128, 64]
    activation: Literal["relu", "leaky_relu", "sigmoid", "tanh", "gelu", "swish", "softplus", "elu", "clipped_relu"] = "relu"
    max_updates: int | None = Field(None, ge=1)  # a lane that reaches it stops; when all have, the race is "complete"


def resolve_lanes(raw_lanes: list[dict]) -> tuple[list[LaneConfig] | None, list[str]]:
    errors: list[str] = []
    if not 1 <= len(raw_lanes) <= 3:
        return None, ["a race needs between 1 and 3 lanes"]
    lanes = []
    for i, raw in enumerate(raw_lanes):
        cfg = {"id": f"lane{i}", **(raw.get("config") or {}), **{k: v for k, v in raw.items() if k != "config"}}
        try:
            lane = LaneConfig.model_validate(cfg)
        except ValidationError as e:
            errors += [f"lanes[{i}].{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
            continue
        if isinstance(lane.batch_size, int) and not 1 <= lane.batch_size <= 1_000_000:
            errors.append(f"lanes[{i}].batch_size: must be 1..1000000 or 'full'")
        if not lane.hidden or len(lane.hidden) > 4 or any(not 2 <= h <= 1024 for h in lane.hidden):
            errors.append(f"lanes[{i}].hidden: 1-4 layers of 2..1024 units")
        lanes.append(lane)
    ids = [l.id for l in lanes]
    if len(set(ids)) != len(ids):
        errors.append("lane ids must be unique")
    return (None, errors) if errors else (lanes, [])


class DoodleData:
    """Doodle tensors (SPEC 4.1): train resident on the GPU when they fit in a fraction of free VRAM, else mmap."""

    def __init__(self, train_x, train_y, val_x, val_y, probe_x, probe_y, device="cpu", max_resident_fraction: float = 0.4,
                 force_resident: bool | None = None):
        self.device = torch.device(device)
        self.n = int(train_x.shape[0])
        want = force_resident
        if want is None and self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            want = train_x.nbytes < free * max_resident_fraction
        self.resident = bool(want)
        if self.resident:
            self.train_x = torch.from_numpy(np.array(train_x)).to(self.device)
            self.train_y = torch.from_numpy(np.asarray(train_y).astype(np.int64)).to(self.device)
        else:
            self.train_x, self.train_y = train_x, np.asarray(train_y)  # numpy / mmap
        stride = max(1, len(val_y) // VAL_N)
        self.val_x = self._f(val_x[::stride][:VAL_N])
        self.val_y = torch.from_numpy(np.asarray(val_y[::stride][:VAL_N]).astype(np.int64)).to(self.device)
        self.probe_x = self._f(probe_x)
        self.probe_y = list(map(int, probe_y))

    def _f(self, x_u8) -> torch.Tensor:
        return torch.from_numpy(np.array(x_u8)).to(self.device).reshape(len(x_u8), -1).float() / 255.0

    @classmethod
    def from_dir(cls, qd: QuickDrawData, device="cpu", **kw) -> "DoodleData":
        if not qd.has_tensors(("train", "val")):
            raise FileNotFoundError("quickdraw/tensors not built: run backend/scripts/rasterize_quickdraw.py")
        tx, ty = qd.tensors("train")
        vx, vy = qd.tensors("val")
        px, py = qd.rasterized("probe")
        return cls(tx, ty, vx, vy, px, py, device, **kw)

    def batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.resident:
            i = idx.to(self.device)
            return self.train_x[i].reshape(len(idx), -1).float() / 255.0, self.train_y[i]
        order = np.sort(idx.numpy())
        x = torch.from_numpy(np.asarray(self.train_x[order])).to(self.device).reshape(len(order), -1).float() / 255.0
        return x, torch.from_numpy(self.train_y[order].astype(np.int64)).to(self.device)

    def slice(self, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Contiguous rows [start, stop) for full-batch accumulation."""
        if self.resident:
            return self.train_x[start:stop].reshape(stop - start, -1).float() / 255.0, self.train_y[start:stop]
        x = torch.from_numpy(np.asarray(self.train_x[start:stop])).to(self.device).reshape(stop - start, -1).float() / 255.0
        return x, torch.from_numpy(self.train_y[start:stop].astype(np.int64)).to(self.device)


class Lane:
    def __init__(self, cfg: LaneConfig, data: DoodleData, seed: int = 0):
        self.cfg, self.data, self.device = cfg, data, data.device
        self.model = build_mlp_classifier(784, cfg.hidden, 10, cfg.activation, seed).to(self.device)
        params = self.model.parameters()
        self.opt = (torch.optim.SGD(params, lr=cfg.lr) if cfg.optimizer == "sgd" else
                    torch.optim.SGD(params, lr=cfg.lr, momentum=0.9) if cfg.optimizer == "momentum" else
                    torch.optim.Adam(params, lr=cfg.lr))
        self.gen = torch.Generator().manual_seed(seed + 1)
        self.updates = 0
        self.samples_seen = 0
        self.finished = False
        self.full = cfg.batch_size == "full"
        self.bs = data.n if self.full else int(cfg.batch_size)
        self.chunk = self._initial_chunk()
        self.pos = 0  # full-batch accumulation pointer
        self._loss_sum = torch.zeros((), device=self.device)
        self._loss_n = 0
        self.last_loss = float("nan")

    def _initial_chunk(self) -> int:
        if not self.full:
            return self.bs
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            per_sample = (784 + sum(self.cfg.hidden) * 3 + 10) * 4 * 3  # activations + grads, rough
            return int(max(1024, min(self.data.n, free * 0.3 / per_sample)))
        return 16384

    def _loss(self, x, y):
        c = self.cfg
        return class_loss(c.loss, self.model(x), y, delta=c.delta, gamma=c.gamma, eps=c.eps)

    def run_slice(self, deadline: float) -> None:
        """Train until `deadline` (perf_counter time) or finished."""
        self.model.train()
        while not self.finished and time.perf_counter() < deadline:
            if self.full:
                self._full_chunk()
            else:
                idx = torch.randint(self.data.n, (self.bs,), generator=self.gen)
                x, y = self.data.batch(idx)
                loss = self._loss(x, y)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()
                self._loss_sum += loss.detach()
                self._loss_n += 1
                self.updates += 1
                self.samples_seen += self.bs
            if self.cfg.max_updates is not None and self.updates >= self.cfg.max_updates:
                self.finished = True
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _full_chunk(self) -> None:
        """One accumulation chunk of a true full-batch update (shrinks the chunk on OOM, never crashes)."""
        n = self.data.n
        stop = min(self.pos + self.chunk, n)
        try:
            x, y = self.data.slice(self.pos, stop)
            if self.pos == 0:
                self.opt.zero_grad(set_to_none=True)
                self._acc = torch.zeros((), device=self.device)
            loss = self._loss(x, y) * ((stop - self.pos) / n)  # weights so the chunks sum to the full-batch mean
            loss.backward()
            self._acc += loss.detach()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self.chunk = max(256, self.chunk // 2)
            self.opt.zero_grad(set_to_none=True)
            self.pos = 0
            return
        self.pos = stop
        if self.pos >= n:
            self.opt.step()
            self.pos = 0
            self.updates += 1
            self.samples_seen += n
            self._loss_sum += self._acc
            self._loss_n += 1

    def snapshot(self) -> dict:
        if self._loss_n:
            self.last_loss = (self._loss_sum / self._loss_n).item()
            self._loss_sum.zero_()
            self._loss_n = 0
        return {"id": self.cfg.id, "step": self.updates, "samples_seen": self.samples_seen, "updates": self.updates,
                "loss": None if np.isnan(self.last_loss) else self.last_loss,
                "acc": accuracy(self.model, self.data.val_x, self.data.val_y),
                "probe_preds": predict(self.model, self.data.probe_x)}


def run_race(lanes: list[Lane], max_seconds: float, emit: Callable[[str, dict], None], should_abort: Callable[[], bool],
             tick_s: float = TICK_S, slice_s: float = SLICE_S) -> str:
    """Interleave `lanes` in short slices. Emits tick/done events; returns the done reason."""
    t0 = time.perf_counter()
    last_tick = -1e9
    reason = "time"
    while True:
        now = time.perf_counter() - t0
        if should_abort():
            reason = "aborted"
            break
        if now >= max_seconds:
            reason = "time"
            break
        if all(l.finished for l in lanes):
            reason = "complete"
            break
        for lane in lanes:
            lane.run_slice(min(time.perf_counter() + slice_s, t0 + max_seconds))
        if time.perf_counter() - t0 - last_tick >= tick_s:
            last_tick = time.perf_counter() - t0
            emit("tick", {"t": round(last_tick, 3), "lanes": [l.snapshot() for l in lanes]})
    final = [l.snapshot() for l in lanes]
    emit("tick", {"t": round(time.perf_counter() - t0, 3), "lanes": final})
    emit("done", {"reason": reason, "final": final})
    return reason
