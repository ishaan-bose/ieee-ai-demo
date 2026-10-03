"""Saved model format (SPEC 8.4) and the ONE loader used by the backend and the tournament.

models/<submission_id>/
  weights.safetensors   fp32 state dict
  config.json           {"config": <resolved config>, "param_count", "tier", "search_depth_full_moves", "nickname", "model_name"}
  meta.json             stop_reason, flops_used, active_gpu_seconds, samples_seen, preemptions, val_loss, timestamps
  curves.json           train/val loss over time
See shared/MODEL_FORMAT.md.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from app.chess_net import encoder as E
from app.chess_net.model import build_model

EVAL_CHUNK = 256  # fixed batch size for leaf evaluations (tournament determinism)


def save_model(out_dir: Path | str, state_dict: dict, config: dict, meta: dict, curves, identity: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.detach().to("cpu", torch.float32).contiguous().clone() for k, v in state_dict.items()}
    tmp = out / "weights.safetensors.tmp"
    save_file(tensors, str(tmp))
    tmp.replace(out / "weights.safetensors")
    cfg_doc = {"config": config, "param_count": config["param_count"], "tier": config["tier"],
               "search_depth_full_moves": config["search_depth_full_moves"], **(identity or {})}
    (out / "config.json").write_text(json.dumps(cfg_doc, indent=2))
    (out / "meta.json").write_text(json.dumps({**meta, "saved_at": time.time()}, indent=2))
    (out / "curves.json").write_text(json.dumps(curves))
    return out


class LoadedModel:
    """A trained evaluator. `score` returns centipawn-like scores for the SIDE TO MOVE."""

    def __init__(self, model, cfg_doc: dict, device, meta: dict | None = None):
        self.model, self.doc, self.cfg, self.device, self.meta = model, cfg_doc, cfg_doc["config"], torch.device(device), meta or {}
        self.extras = E.Extras.from_dict(self.cfg["input_extras"])
        self.name = cfg_doc.get("model_name") or cfg_doc.get("nickname") or "model"
        self.search_depth_full_moves = cfg_doc["search_depth_full_moves"]

    @torch.no_grad()
    def score(self, boards, stm, castle, ep) -> torch.Tensor:
        """Inputs: numpy arrays or tensors of N positions. Returns float32 cpu tensor (N,).

        Runs in FIXED-size fp32 chunks (padded), so a position's score never depends on its neighbours.
        """
        t = [torch.as_tensor(a) for a in (boards, stm, castle, ep)]
        n = t[0].shape[0]
        out = []
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # fp32 for evaluation (SPEC 6.4)
        try:
            for s in range(0, n, EVAL_CHUNK):
                chunk = [x[s:s + EVAL_CHUNK] for x in t]
                m = chunk[0].shape[0]
                if m < EVAL_CHUNK:  # pad by repeating the first row
                    chunk = [torch.cat([x, x[:1].expand(EVAL_CHUNK - m, *x.shape[1:])]) for x in chunk]
                chunk = [x.to(self.device) for x in chunk]
                z = self.model(E.encode(*chunk, self.extras))
                pred = E.head_to_target_space(z, head=self.cfg["output_head"], target_type=self.cfg["target_type"],
                                              K=self.cfg["eval_squash_scale"], mate_clip=self.cfg["mate_clip"])
                sc = E.score_cp_for_stm(pred, chunk[1], target_type=self.cfg["target_type"], K=self.cfg["eval_squash_scale"],
                                        mate_clip=self.cfg["mate_clip"], perspective_flip=self.cfg["perspective_flip"])
                out.append(sc[:m].float().cpu())
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        return torch.cat(out) if out else torch.zeros(0)


def load_model(model_dir: Path | str, device="cpu") -> LoadedModel:
    d = Path(model_dir)
    cfg_doc = json.loads((d / "config.json").read_text())
    model = build_model(cfg_doc["config"])
    model.load_state_dict(load_file(str(d / "weights.safetensors")))
    model.to(device).eval()
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").is_file() else {}
    return LoadedModel(model, cfg_doc, device, meta)


def list_models(models_dir: Path | str) -> list[Path]:
    p = Path(models_dir)
    return sorted(d for d in p.iterdir() if (d / "weights.safetensors").is_file() and (d / "config.json").is_file()) if p.is_dir() else []
