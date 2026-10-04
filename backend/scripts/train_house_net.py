#!/usr/bin/env python3
"""Train the House Net (SPEC 8.6): the strongest config the owner can find WITHIN THE RULES, same pipeline, same budget and cap.

1. (optional) a small random search of configs at a fraction of the budget, ranked by comparable validation MSE;
2. the best config is trained with the full BUDGET_FLOPS / time cap -> STATE_DIR/models/house-net/ (entered as "House Net");
3. the DEFAULT_CONFIG is trained with the same budget -> STATE_DIR/baselines/default-config/ so the gap can be checked
   ("default is meh, House Net is clearly better"). The House Net is NOT the default.

Idempotent: skips when models/house-net/meta.json exists (--force redoes); the search result is cached in STATE_DIR/house_search.json.
Use `--config path.json` to train a config you chose yourself instead of searching.

    cd ~/ieee-ai-demo/backend && python3 scripts/train_house_net.py [--trials 12] [--quick] [--force]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chess_net.config import DEFAULT_CONFIG, resolve_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.data.loaders import LichessData  # noqa: E402
from app.device import detect_device  # noqa: E402
from app.training.chess_data import ChessData  # noqa: E402
from app.training.trainer import CompetitionTrainer  # noqa: E402
from scripts._common import setup_logging, write_json  # noqa: E402

# A hand-picked starting point (educated guess, to be improved by the search): deep-ish residual layernorm net, cosine schedule,
# EMA, richer inputs, symmetry augmentation, perspective flip. Everything here is a normal participant knob.
HOUSE_START = {
    "layers": 4, "width": 1024, "activation": {"name": "gelu"}, "loss": {"name": "bce"}, "lr": 1e-3, "batch_size": 1024,
    "optimizer": "adam", "target_type": "winprob", "eval_squash_scale": 400.0, "mate_clip": 2000.0,
    "input_extras": {"stm_castle": True, "en_passant": True, "material": True, "attacks": True},
    "perspective_flip": True, "color_flip_augmentation": True, "data_slice": "all", "sampling": "uniform", "init": "he",
    "lr_schedule": "cosine", "warmup_frac": 0.02, "grad_clip": 1.0, "ema": {"enabled": True, "decay": 0.999},
    "normalization": "layernorm", "residual": True, "output_head": "linear", "seed": 7,
}


def random_config(rng: random.Random) -> dict:
    c = copy.deepcopy(HOUSE_START)
    c["layers"] = rng.choice([3, 4, 5, 6])
    c["width"] = rng.choice([512, 768, 1024, 1536])
    c["activation"] = {"name": rng.choice(["relu", "gelu", "swish", "leaky_relu"])}
    c["loss"] = {"name": rng.choice(["mse", "huber", "bce"]), "delta": rng.choice([0.5, 1.0])}
    c["lr"] = 10 ** rng.uniform(-3.6, -2.6)
    c["batch_size"] = rng.choice([256, 512, 1024, 2048])
    c["lr_schedule"] = rng.choice(["cosine", "linear"])
    c["warmup_frac"] = rng.choice([0.0, 0.02, 0.05])
    c["color_flip_augmentation"] = rng.random() < 0.7
    c["normalization"] = rng.choice(["layernorm", "none"])
    c["residual"] = rng.random() < 0.6
    c["input_extras"] = {"stm_castle": True, "en_passant": rng.random() < 0.5, "material": rng.random() < 0.7, "attacks": rng.random() < 0.6}
    c["init"] = rng.choice(["he", "xavier"])
    c["seed"] = rng.randrange(1000)
    return c


def train_one(raw: dict, data: ChessData, device, budget: float, cap: float, log, label: str):
    cfg, errors = resolve_config(raw)
    if errors:
        raise SystemExit(f"{label}: invalid config: {errors}")
    t0 = time.time()
    tr = CompetitionTrainer(cfg, data, device, budget_flops=budget, time_cap_s=cap, metrics_every_s=max(5.0, min(30.0, cap / 20)))
    reason = tr.run(lambda: False, on_metrics=lambda r: log.info("%s step %d val_mse %.5f train_loss %.5f", label, r["step"], r["val_mse"], r["train_loss"]))
    ev = tr.evaluate()
    log.info("%s: %s after %.0fs, %s samples, val_mse %.5f (%s params, tier %s)", label, reason, time.time() - t0, f"{tr.samples_seen:,}",
             ev["val_mse"], f"{cfg['param_count']:,}", cfg["tier"])
    return tr, cfg, reason, ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=12, help="random-search trials (0 = skip the search and train the hand-picked start config)")
    ap.add_argument("--trial-fraction", type=float, default=0.05, help="fraction of the budget given to each search trial")
    ap.add_argument("--config", type=Path, default=None, help="JSON file with the config to train (skips the search)")
    ap.add_argument("--no-default-baseline", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny budget and 2 trials (rehearsal / smoke test)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    s = get_settings()
    log = setup_logging("train_house_net", s)
    device = torch.device(detect_device().device)
    out_dir = s.models_dir / "house-net"
    if (out_dir / "meta.json").is_file() and not a.force:
        meta = json.loads((out_dir / "meta.json").read_text())
        print("SUMMARY train_house_net:", json.dumps({"skipped": True, "val_mse": meta.get("val_mse"), "stop_reason": meta.get("stop_reason")}))
        return 0
    budget = 3e9 if a.quick else s.budget_flops
    cap = 20.0 if a.quick else s.time_cap_seconds
    ld = s.data_dir / "lichess"
    if not (ld / "boards.npy").is_file():
        raise SystemExit(f"lichess data not found under {ld}")
    data = ChessData(LichessData(ld), device, val_rows=2000 if a.quick else s.val_rows)
    log.info("budget %.3e FLOPs, time cap %.0fs, device %s, data resident on device: %s", budget, cap, device, data.resident)

    chosen = HOUSE_START
    search_path = s.state_dir / "house_search.json"
    if a.config:
        chosen = json.loads(a.config.read_text())
        log.info("using config from %s", a.config)
    elif a.trials > 0 or a.quick:
        if search_path.is_file() and not a.force and not a.quick:
            chosen = json.loads(search_path.read_text())["best"]
            log.info("reusing cached search result %s", search_path)
        else:
            rng = random.Random(0)
            trials = [HOUSE_START] + [random_config(rng) for _ in range(max(1, (2 if a.quick else a.trials) - 1))]
            results = []
            for i, raw in enumerate(trials):
                tr, cfg, reason, ev = train_one(raw, data, device, budget * (0.1 if a.quick else a.trial_fraction),
                                                cap * a.trial_fraction * 3, log, f"trial {i + 1}/{len(trials)}")
                results.append({"val_mse": ev["val_mse"], "config": raw, "param_count": cfg["param_count"], "stop": reason})
            results.sort(key=lambda r: (r["val_mse"] != r["val_mse"], r["val_mse"]))
            chosen = results[0]["config"]
            if not a.quick:
                write_json(search_path, {"best": chosen, "results": results})
            log.info("search ranking (val_mse): %s", [round(r["val_mse"], 5) for r in results[:5]])

    tr, cfg, reason, ev = train_one(chosen, data, device, budget, cap, log, "HOUSE NET")
    identity = {"nickname": "House Net", "model_name": "House Net", "participant_code": "HOUSE", "submission_id": "house-net"}
    meta = tr.export(out_dir, reason, identity=identity)
    write_json(out_dir / "chosen_config.json", chosen)
    summary = {"house_net": {"val_mse": meta["val_mse"], "tier": cfg["tier"], "params": cfg["param_count"], "stop": reason,
                             "active_gpu_seconds": round(meta["active_gpu_seconds"], 1)}}
    if not a.no_default_baseline:
        dtr, dcfg, dreason, dev_ = train_one(copy.deepcopy(DEFAULT_CONFIG), data, device, budget, cap, log, "DEFAULT BASELINE")
        dmeta = dtr.export(s.state_dir / "baselines" / "default-config", dreason,
                           identity={"nickname": "Default", "model_name": "Default config", "participant_code": "DEFAULT", "submission_id": "default-config"})
        summary["default_config"] = {"val_mse": dmeta["val_mse"], "tier": dcfg["tier"], "params": dcfg["param_count"]}
        ratio = dmeta["val_mse"] / max(meta["val_mse"], 1e-12)
        summary["default_over_house_val_mse"] = round(ratio, 2)
        log.info("default val_mse / house val_mse = %.2f (want clearly above 1: the House Net must be clearly better)", ratio)
    print("SUMMARY train_house_net:", json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
