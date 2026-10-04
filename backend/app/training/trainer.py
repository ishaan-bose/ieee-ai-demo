"""Competition trainer (SPEC 6.1, 6.2, 8): trains one chess evaluator on a FLOPs budget.

- Stops at the FLOPs budget or the active-GPU-time cap, whichever comes first (`run` returns "budget" / "time"),
  on NaN/inf ("diverged"), or when asked to abort ("aborted"; the caller decides kill vs preempt). No saving on abort.
- FLOPs = 6 x matmul-weight-count x samples_seen (exact, logged). LR schedule + warmup are over FLOPs progress.
- Checkpoints (model, optimizer, EMA, RNG, counters) every ~ckpt_every_s of ACTIVE time, so a resumed run never double counts.
- Handles CUDA OOM by splitting the batch into micro-batches; the worker never crashes on it.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable

import torch

from app.chess_net import encoder as E
from app.chess_net.config import flops_per_sample
from app.chess_net.model import build_model
from app.chess_net.model_io import save_model
from app.training.chess_data import ChessData
from app.training.hostio import HostStager
from app.training.losses import regression_loss
from app.training.optim import EMA, build_optimizer, lr_factor

CHECK_EVERY = 25  # steps between sync points in the legacy (optimized=False) loop
CHECK_MIN, CHECK_MAX, CHECK_TARGET_S = 10, 400, 0.25  # optimized loop: the interval between sync points adapts to ~0.25 s of training


class CompetitionTrainer:
    def __init__(self, cfg: dict, data: ChessData, device: str | torch.device = "cpu", *, budget_flops: float,
                 time_cap_s: float, ckpt_every_s: float = 10.0, metrics_every_s: float = 2.0, optimized: bool = True):
        """optimized=False keeps the original loop (blocking uploads, a sync every 25 steps) so tests can prove the fast path trains identically."""
        self.cfg, self.data, self.device = cfg, data, torch.device(device)
        self.optimized = optimized
        self.stager = HostStager(self.device) if optimized else None
        self.budget_flops, self.time_cap_s = float(budget_flops), float(time_cap_s)
        self.ckpt_every_s, self.metrics_every_s = ckpt_every_s, metrics_every_s
        torch.manual_seed(cfg["seed"])
        self.model = build_model(cfg).to(self.device)
        self.opt = build_optimizer(self.model.parameters(), cfg["optimizer"], cfg["lr"])
        self.ema = EMA(self.model.parameters(), cfg["ema"]["decay"]) if cfg["ema"]["enabled"] else None
        self.gen = torch.Generator().manual_seed(cfg["seed"] + 12345)
        self.pool = data.pool(cfg["data_slice"], cfg["sampling"], cfg["mate_clip"], cfg["eval_squash_scale"])
        self.extras = E.Extras.from_dict(cfg["input_extras"])
        self.fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
        self.step = self.samples_seen = 0
        self.flops_used = 0.0
        self.active_s = 0.0
        self.micro = 1  # micro-batch split factor, grows on OOM
        self.curves: list[dict] = []
        self.last_train_loss = float("nan")

    # ------------------------------------------------------------ one step
    def _upload(self, t: torch.Tensor) -> torch.Tensor:
        return self.stager.upload(t) if self.stager is not None else t.to(self.device)

    def _prepare(self, b: dict) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        boards, stm, castle, ep, cp, mate = (b[k] for k in ("boards", "stm", "castle", "ep", "cp", "mate"))
        if c["color_flip_augmentation"]:
            m = self._upload(torch.rand(boards.shape[0], generator=self.gen) < 0.5)
            fb, fs, fc, fe = E.flip_position(boards, stm, castle, ep)
            boards, stm, castle = (torch.where(m[:, None] if x.dim() > 1 else m, f, x) for x, f in ((boards, fb), (stm, fs), (castle, fc)))
            cp, mate = (torch.where(m, -x, x) for x in (cp, mate))  # the White-view eval negates under the flip
        x = E.encode(boards, stm, castle, ep, self.extras)
        y = E.make_target(cp, mate, stm, target_type=c["target_type"], K=c["eval_squash_scale"], mate_clip=c["mate_clip"],
                          perspective_flip=c["perspective_flip"])
        return x, y

    def _loss(self, x, y):
        c = self.cfg
        z = self.model(x)
        pred = E.head_to_target_space(z, head=c["output_head"], target_type=c["target_type"], K=c["eval_squash_scale"],
                                      mate_clip=c["mate_clip"])
        return regression_loss(c["loss"]["name"], pred, y, c["loss"]["delta"])

    def _train_step(self) -> torch.Tensor:
        c = self.cfg
        idx = self.data.sample_indices(self.pool, c["batch_size"], self.gen)
        x, y = self._prepare(self.data.fetch(idx, self.stager))
        for g in self.opt.param_groups:
            g["lr"] = c["lr"] * lr_factor(c["lr_schedule"], self.flops_used / self.budget_flops, c["warmup_frac"])
        while True:
            try:
                self.opt.zero_grad(set_to_none=True)
                if self.micro == 1 and self.optimized:  # the common case: no zeros(), no x1.0 scaling, no chunking (bit-identical results)
                    loss = self._loss(x, y)
                    loss.backward()
                    total = loss.detach()
                else:
                    total = torch.zeros((), device=self.device)
                    n = x.shape[0]
                    for xs, ys in zip(x.chunk(self.micro), y.chunk(self.micro)):
                        loss = self._loss(xs, ys) * (xs.shape[0] / n)
                        loss.backward()
                        total += loss.detach()
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.micro >= x.shape[0]:
                    raise
                self.micro *= 2
        if c["grad_clip"] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["grad_clip"])
        self.opt.step()
        if self.ema is not None:
            self.ema.update(self.model.parameters())
        self.step += 1
        self.samples_seen += c["batch_size"]
        self.flops_used = float(self.fps) * self.samples_seen
        return total

    # ------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Validation (is_val == 1 subset), fp32. `val_loss` uses the configured loss; `val_mse` is the comparable
        win-probability MSE at the canonical scale (same number for every config)."""
        c = self.cfg
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        ctx = self.ema.applied(self.model.parameters()) if self.ema is not None else _null()
        self.model.eval()
        sums = {"loss": torch.zeros((), dtype=torch.float64, device=self.device), "mse": torch.zeros((), dtype=torch.float64, device=self.device)}
        n = 0  # (accumulated in float64 on the device, read back once: the same arithmetic as per-batch .item() but without a sync per batch)
        try:
            with ctx:
                for b in self.data.val_batches(stager=self.stager):
                    x = E.encode(b["boards"], b["stm"], b["castle"], b["ep"], self.extras)
                    y = E.make_target(b["cp"], b["mate"], b["stm"], target_type=c["target_type"], K=c["eval_squash_scale"],
                                      mate_clip=c["mate_clip"], perspective_flip=c["perspective_flip"])
                    z = self.model(x)
                    pred = E.head_to_target_space(z, head=c["output_head"], target_type=c["target_type"],
                                                  K=c["eval_squash_scale"], mate_clip=c["mate_clip"])
                    m = x.shape[0]
                    sums["loss"] += regression_loss(c["loss"]["name"], pred, y, c["loss"]["delta"]).double() * m
                    score = E.score_cp_for_stm(pred, b["stm"], target_type=c["target_type"], K=c["eval_squash_scale"],
                                               mate_clip=c["mate_clip"], perspective_flip=c["perspective_flip"])
                    ref = torch.sigmoid(E.effective_cp(b["cp"], b["mate"], 2000.0) * torch.where(b["stm"] == 1, 1.0, -1.0) / E.EVAL_K)
                    sums["mse"] += ((torch.sigmoid(score / E.EVAL_K) - ref) ** 2).mean().double() * m
                    n += m
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
            self.model.train()
        return {"val_loss": sums["loss"].item() / max(n, 1), "val_mse": sums["mse"].item() / max(n, 1)}

    # ------------------------------------------------------------ main loop
    def run(self, should_abort: Callable[[], bool], on_metrics: Callable[[dict], None] | None = None,
            on_checkpoint: Callable[[], None] | None = None) -> str:
        """Train until budget / time / divergence / abort. Returns the stop reason."""
        self.model.train()
        base_active = self.active_s
        t0 = time.perf_counter()
        last_metrics = last_ckpt = base_active
        loss_acc = torch.zeros((), device=self.device)
        loss_n = 0

        def sync_active() -> float:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.active_s = base_active + (time.perf_counter() - t0)
            return self.active_s

        check_every = CHECK_MIN if self.optimized else CHECK_EVERY
        since = 0  # steps since the last sync point
        prev_a = base_active
        reason = None
        while reason is None:
            if should_abort():
                reason = "aborted"
                break
            if self.flops_used >= self.budget_flops:
                reason = "budget"
                break
            loss_acc += self._train_step().detach()
            loss_n += 1
            since += 1
            if since >= check_every:
                # The ONLY GPU sync in the loop: read the clock and the loss every `check_every` steps (adaptive, ~0.25 s of work), not every step.
                a = sync_active()
                if self.optimized:
                    per_step = max((a - prev_a) / since, 1e-6)
                    check_every = int(min(CHECK_MAX, max(CHECK_MIN, CHECK_TARGET_S / per_step)))
                prev_a, since = a, 0
                self.last_train_loss = (loss_acc / loss_n).item()
                if not math.isfinite(self.last_train_loss):
                    reason = "diverged"
                    break
                if a >= self.time_cap_s:
                    reason = "time"
                    break
                if a - last_metrics >= self.metrics_every_s:
                    last_metrics = a
                    rec = {"t": round(a, 3), "step": self.step, "samples_seen": self.samples_seen, "flops": self.flops_used,
                           "train_loss": self.last_train_loss, **self.evaluate()}
                    self.curves.append(rec)
                    loss_acc.zero_(); loss_n = 0
                    if on_metrics:
                        on_metrics(rec)
                if a - last_ckpt >= self.ckpt_every_s and on_checkpoint:
                    last_ckpt = a
                    on_checkpoint()
        sync_active()
        if reason == "aborted":
            return reason
        if loss_n:
            self.last_train_loss = (loss_acc / loss_n).item()
        if not _finite_weights(self.model):
            reason = "diverged"
        return reason

    # ------------------------------------------------------------ persistence
    def checkpoint_state(self) -> dict:
        return {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "ema": self.ema.state() if self.ema else None,
                "gen": self.gen.get_state(), "step": self.step, "samples_seen": self.samples_seen, "flops_used": self.flops_used,
                "active_s": self.active_s, "micro": self.micro, "curves": self.curves}

    def save_checkpoint(self, path: Path | str) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(self.checkpoint_state(), tmp)
        tmp.replace(path)
        return {"path": str(path), "step": self.step, "samples_seen": self.samples_seen, "flops_used": self.flops_used,
                "active_gpu_seconds": self.active_s}

    def load_checkpoint(self, path: Path | str) -> None:
        st = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        if self.ema is not None and st["ema"] is not None:
            self.ema.load(st["ema"])
        self.gen.set_state(st["gen"].cpu())
        self.step, self.samples_seen, self.flops_used = st["step"], st["samples_seen"], st["flops_used"]
        self.active_s, self.micro, self.curves = st["active_s"], st.get("micro", 1), st["curves"]

    def final_state_dict(self) -> dict:
        """Weights to ship: EMA weights when EMA is on."""
        if self.ema is None:
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        with self.ema.applied(self.model.parameters()):
            return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def export(self, out_dir: Path | str, stop_reason: str, *, preemptions: int = 0, identity: dict | None = None,
               started_at: float | None = None) -> dict:
        final = self.evaluate()
        self.curves.append({"t": round(self.active_s, 3), "step": self.step, "samples_seen": self.samples_seen,
                            "flops": self.flops_used, "train_loss": self.last_train_loss, **final})
        meta = {"stop_reason": stop_reason, "flops_used": self.flops_used, "flops_budget": self.budget_flops,
                "active_gpu_seconds": self.active_s, "samples_seen": self.samples_seen, "steps": self.step,
                "preemptions": preemptions, "val_loss": final["val_loss"], "val_mse": final["val_mse"],
                "train_loss": self.last_train_loss, "started_at": started_at, "finished_at": time.time(),
                "device": str(self.device)}
        save_model(out_dir, self.final_state_dict(), self.cfg, meta, self.curves, identity)
        return meta


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _finite_weights(model) -> bool:
    return all(torch.isfinite(p).all().item() for p in model.parameters())
