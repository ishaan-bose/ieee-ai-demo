#!/usr/bin/env python3
"""Train the showcase models and precompute the Act 3 cached curves (SPEC 9). Idempotent: finished pieces are skipped (--force redoes).

Outputs (STATE_DIR/showcase/*.pt + *.json, and the cached JSON the frontend loads):
  doodle.pt         MLP 784-256-128-10 for the Act 2 forward pass and the Act 4 rematch. Trained on full drawings PLUS random
                    stroke-prefix drawings, because the browser model sees partial drawings (after every stroke).
  raw_audio.pt      raw-waveform MLP (4000-128-64-10): expected to flail
  logmel_audio.pt   small log-mel CNN: expected to work
  frontend/public/cache/act3/trap.json     metrics trap: `marvin` detector at ~2% positives (always-"no" scores ~98%)
  frontend/public/cache/act3/overfit.json  overfitting: big model on a tiny subset + a training-set-size sweep

    cd ~/ieee-ai-demo/backend && python3 scripts/train_showcase.py [--only doodle raw logmel trap overfit] [--quick] [--force]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]

from app.config import get_settings  # noqa: E402
from app.data import audio_features as af  # noqa: E402
from app.data.loaders import QuickDrawData, SpeechData  # noqa: E402
from app.data.rasterizer import SIZE, rasterize  # noqa: E402
from app.device import detect_device  # noqa: E402
from app.training import audio_models as am  # noqa: E402
from app.training.classifier import accuracy, build_mlp_classifier  # noqa: E402
from scripts._common import setup_logging, write_json  # noqa: E402

DOODLE_HIDDEN = [256, 128]
STEPS = ("doodle", "raw", "logmel", "trap", "overfit")


# ------------------------------------------------------------------ partial (stroke-prefix) drawings
def _partial_work(args) -> tuple[np.ndarray, np.ndarray]:
    lines, seed = args
    rng = np.random.default_rng(seed)
    x = np.zeros((len(lines), SIZE, SIZE), np.uint8)
    y = np.zeros(len(lines), np.int8)
    for i, line in enumerate(lines):
        r = json.loads(line)
        k = int(rng.integers(1, len(r["d"]) + 1))  # keep the first k strokes, like a drawing in progress
        x[i] = rasterize(r["d"][:k])
        y[i] = r["c"]
    return x, y


def partial_set(path: Path, n: int, seed: int, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """~n stroke-prefix drawings from a JSONL file (every stride-th line; deterministic)."""
    with path.open("rb") as f:
        total = sum(1 for _ in f)
    stride = max(1, total // n)
    lines = []
    with path.open("rb") as f:
        for i, line in enumerate(f):
            if i % stride == 0 and line.strip():
                lines.append(line)
    chunks = [(lines[i:i + 4000], seed + i) for i in range(0, len(lines), 4000)]
    if workers > 1 and len(chunks) > 1:
        with mp.Pool(workers) as pool:
            res = pool.map(_partial_work, chunks)
    else:
        res = [_partial_work(c) for c in chunks]
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ------------------------------------------------------------------ doodle
def train_doodle(ctx) -> dict:
    s, log, dev, quick = ctx.s, ctx.log, ctx.device, ctx.a.quick
    qd = QuickDrawData(s.data_dir / "quickdraw")
    if not qd.has_tensors():
        raise SystemExit("quickdraw/tensors missing: run scripts/rasterize_quickdraw.py first")
    tx, ty = qd.tensors("train")
    vx, vy = qd.tensors("val")
    n_part = 3000 if quick else 200_000
    t0 = time.time()
    px, py = partial_set(qd.processed / "train.jsonl", n_part, 1, s.num_workers)
    vpx, vpy = partial_set(qd.processed / "val.jsonl", 500 if quick else 5000, 2, s.num_workers)
    log.info("doodle: %d partial train drawings and %d partial val drawings rasterized in %.0fs", len(py), len(vpy), time.time() - t0)

    def gpu(a):
        return torch.from_numpy(np.array(a)).to(dev)

    full_resident = tx.nbytes < 4e9 and (dev.type == "cuda" or not quick)
    TX = gpu(tx) if full_resident else tx
    TY = gpu(ty).long() if full_resident else ty
    PX, PY = gpu(px), gpu(py).long()
    n_full = len(ty)

    def batch(b_full: int, b_part: int, g):
        i = torch.randint(n_full, (b_full,), generator=g)
        if full_resident:
            x, y = TX[i.to(dev)], TY[i.to(dev)]
        else:
            o = np.sort(i.numpy())
            x, y = torch.from_numpy(TX[o]).to(dev), torch.from_numpy(TY[o].astype(np.int64)).to(dev)
        j = torch.randint(len(PY), (b_part,), generator=g).to(dev)
        x = torch.cat([x, PX[j]]).reshape(-1, 784).float() / 255.0
        return x, torch.cat([y, PY[j]])

    model = build_mlp_classifier(784, DOODLE_HIDDEN, 10, "relu", seed=0).to(dev)
    bs, epochs = 256, 1 if quick else ctx.a.doodle_epochs
    steps_per_epoch = 150 if quick else n_full // bs
    total = epochs * steps_per_epoch
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=total, pct_start=0.05)
    g = torch.Generator().manual_seed(0)
    vxf = torch.from_numpy(np.array(vx)).to(dev).reshape(len(vy), -1).float() / 255.0
    vyt = torch.from_numpy(np.array(vy).astype(np.int64)).to(dev)
    vpxf = gpu(vpx).reshape(len(vpy), -1).float() / 255.0
    vpyt = gpu(vpy).long()
    t0 = time.time()
    for ep in range(epochs):
        for st in range(steps_per_epoch):
            x, y = batch(bs * 3 // 4, bs // 4, g)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        log.info("doodle epoch %d/%d: val acc (full drawings) %.4f, (partial drawings) %.4f, %.0fs", ep + 1, epochs,
                 accuracy(model, vxf, vyt), accuracy(model, vpxf, vpyt), time.time() - t0)
    model.eval()
    with torch.no_grad():
        pred = model(vxf).argmax(1).cpu().numpy()
    true = vyt.cpu().numpy()
    per_class = {c: float((pred[true == i] == i).mean()) for i, c in enumerate(qd.classes)}
    res = {"val_acc": accuracy(model, vxf, vyt), "val_acc_partial": accuracy(model, vpxf, vpyt), "per_class_recall": per_class,
           "hidden": DOODLE_HIDDEN, "epochs": epochs, "n_val": int(len(vy)), "seconds": round(time.time() - t0, 1), "quick": quick}
    save(ctx, "doodle", model, res, {"classes": qd.classes})
    return res


# ------------------------------------------------------------------ audio helpers
def save(ctx, name: str, model: nn.Module, metrics: dict, extra: dict | None = None) -> None:
    d = ctx.s.state_dir / "showcase"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "metrics": metrics, **(extra or {})}, d / f"{name}.pt.tmp")
    (d / f"{name}.pt.tmp").replace(d / f"{name}.pt")
    write_json(d / f"{name}.json", metrics)


def audio_split(ctx, kind: str, split: str, idx: np.ndarray | None = None):
    sp: SpeechData = ctx.speech
    idx = sp.word_indices(split) if idx is None else idx
    if ctx.a.quick:
        idx = idx[:600]
    X = sp.X(split)[idx]
    return am.make_features(kind, X, ctx.device), sp.y(split)[idx]


def train_audio_model(ctx, kind: str) -> dict:
    name = "raw_audio" if kind == "raw" else "logmel_audio"
    xtr, ytr = audio_split(ctx, kind, "train")
    xva, yva = audio_split(ctx, kind, "val")
    xte, yte = audio_split(ctx, kind, "test")
    model = (am.build_raw_model if kind == "raw" else am.build_logmel_model)(seed=0)
    epochs = 2 if ctx.a.quick else ctx.a.audio_epochs
    t0 = time.time()
    hist = am.train_audio(model, xtr, ytr, xva, yva, epochs=epochs, device=ctx.device,
                          on_epoch=lambda r: ctx.log.info("%s epoch %d: train %.3f val %.3f", name, r["epoch"], r["train_acc"], r["val_acc"]))
    test_acc = accuracy(model.to(ctx.device).eval(), xte.to(ctx.device).float() if xte.dtype != torch.float32 else xte.to(ctx.device), torch.as_tensor(yte).long().to(ctx.device))
    res = {"val_acc": hist[-1]["val_acc"], "test_acc": test_acc, "train_acc": hist[-1]["train_acc"], "epochs": epochs,
           "history": hist, "seconds": round(time.time() - t0, 1), "quick": ctx.a.quick,
           "classes": ctx.speech.classes[:10]}
    save(ctx, name, model, res, {"classes": ctx.speech.classes[:10]})
    return {k: v for k, v in res.items() if k not in ("history", "classes")}


# ------------------------------------------------------------------ Act 3 step 2: metrics trap
def binary_metrics(model, x, y, device) -> dict:
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[s:s + 4096].to(device).float()).argmax(1).cpu() for s in range(0, len(y), 4096)])
    y = torch.as_tensor(y).long()
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    model.train()
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"accuracy": (tp + tn) / max(1, len(y)), "recall": rec, "precision": prec,
            "f1": (2 * prec * rec / (prec + rec)) if prec and rec else 0.0, "confusion": [[tn, fp], [fn, tp]]}


def build_trap(ctx) -> dict:
    sp: SpeechData = ctx.speech
    out: dict = {"task": "marvin detector", "positive_word": "marvin", "classes": ["not marvin", "marvin"]}
    data = {}
    for split in ("train", "val", "test"):
        idx, lab = sp.marvin_subset(split, 0.02, seed=0)
        if ctx.a.quick:
            keep = np.concatenate([np.flatnonzero(lab == 0)[:500], np.flatnonzero(lab == 1)[:12]])
            idx, lab = idx[np.sort(keep)], lab[np.sort(keep)]
        X = sp.X(split)[idx]
        data[split] = (am.make_features("logmel", X, ctx.device), lab)
    out["n"] = {k: {"total": int(len(v[1])), "positives": int(v[1].sum())} for k, v in data.items()}
    out["positive_fraction"] = {k: float(v[1].mean()) for k, v in data.items()}
    te = data["test"][1]
    out["always_no"] = {"accuracy": float(1 - te.mean()), "recall": 0.0, "precision": None, "confusion": [[int((te == 0).sum()), 0], [int(te.sum()), 0]]}
    epochs = 2 if ctx.a.quick else ctx.a.audio_epochs
    pos_w = float((1 - data["train"][1].mean()) / max(data["train"][1].mean(), 1e-9))
    for variant, weight in (("naive", None), ("weighted", [1.0, pos_w])):
        model = am.build_logmel_model(n_out=2, seed=0)
        epoch_rows = []

        def on_epoch(r, model=model, rows=epoch_rows):
            m = binary_metrics(model.to(ctx.device), data["val"][0], data["val"][1], ctx.device)
            rows.append({"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val_loss"], "val_accuracy": m["accuracy"],
                         "val_recall": m["recall"], "val_precision": m["precision"]})
            ctx.log.info("trap/%s epoch %d: val acc %.4f recall %s", variant, r["epoch"], m["accuracy"], m["recall"])
        am.train_audio(model, data["train"][0], data["train"][1], data["val"][0], data["val"][1], epochs=epochs, device=ctx.device,
                       on_epoch=on_epoch, weight=weight)
        out[variant] = {"epochs": epoch_rows, "test": binary_metrics(model.to(ctx.device), data["test"][0], data["test"][1], ctx.device),
                        "class_weights": weight}
    p = ctx.frontend / "public" / "cache" / "act3" / "trap.json"
    write_json(p, out)
    ctx.log.info("trap: wrote %s (always-no accuracy %.3f, naive test accuracy %.3f recall %s)", p, out["always_no"]["accuracy"],
                 out["naive"]["test"]["accuracy"], out["naive"]["test"]["recall"])
    return {"always_no_acc": out["always_no"]["accuracy"], "naive_test_acc": out["naive"]["test"]["accuracy"],
            "naive_recall": out["naive"]["test"]["recall"], "weighted_recall": out["weighted"]["test"]["recall"]}


# ------------------------------------------------------------------ Act 3 step 3: overfitting
def build_overfit(ctx) -> dict:
    sp: SpeechData = ctx.speech
    xva, yva = audio_split(ctx, "logmel", "val")
    tr_idx = sp.word_indices("train")
    sizes = [100, 300, 1000, 3000, 10000, len(tr_idx)]
    if ctx.a.quick:
        sizes, tr_idx = [50, 200], tr_idx[:600]
    sizes = [n for n in sizes if n <= len(tr_idx)]
    out = {"model": "MLP 3920-512-512-10 on log-mel (large for the data)", "default_n": 300 if 300 in sizes else sizes[0], "runs": []}
    rng = np.random.default_rng(0)
    order = rng.permutation(tr_idx)  # nested subsets: every larger set contains the smaller ones
    for n in sizes:
        idx = np.sort(order[:n])
        X = sp.X("train")[idx]
        xtr, ytr = am.make_features("logmel", X, ctx.device), sp.y("train")[idx]
        epochs = 4 if ctx.a.quick else int(np.clip(round(60 * (300 / n) ** 0.5), 12, 60))
        torch.manual_seed(0)
        model = nn.Sequential(nn.Flatten(), nn.Linear(af.N_MELS * af.N_FRAMES, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 10))
        hist = am.train_audio(model, xtr, ytr, xva, yva, epochs=epochs, device=ctx.device, batch=min(64, n), lr=1e-3)
        out["runs"].append({"n": n, "epochs": [{k: h[k] for k in ("epoch", "train_loss", "train_acc", "val_loss", "val_acc")} for h in hist]})
        ctx.log.info("overfit n=%d: final train %.3f val %.3f", n, hist[-1]["train_acc"], hist[-1]["val_acc"])
    p = ctx.frontend / "public" / "cache" / "act3" / "overfit.json"
    write_json(p, out)
    return {"sizes": sizes, "default_run_final": {k: out["runs"][min(1, len(out["runs"]) - 1)]["epochs"][-1][k] for k in ("train_acc", "val_acc")}}


# ------------------------------------------------------------------ main
class Ctx:
    pass


def done_marker(ctx, step: str) -> Path:
    return ctx.s.state_dir / "showcase" / f"{step}.done.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=STEPS, default=list(STEPS))
    ap.add_argument("--quick", action="store_true", help="tiny sizes for a rehearsal on the CPU plan / smoke test")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--doodle-epochs", type=int, default=6)
    ap.add_argument("--audio-epochs", type=int, default=15)
    ap.add_argument("--frontend-dir", type=Path, default=REPO / "frontend")
    a = ap.parse_args(argv)
    ctx = Ctx()
    ctx.a, ctx.s, ctx.frontend = a, get_settings(), a.frontend_dir
    ctx.log = setup_logging("train_showcase", ctx.s)
    ctx.device = torch.device(detect_device().device)
    speech_dir = ctx.s.data_dir / "speech" / "processed"
    ctx.speech = SpeechData(speech_dir) if (speech_dir / "X_train.npy").is_file() else None
    fns = {"doodle": train_doodle, "raw": lambda c: train_audio_model(c, "raw"), "logmel": lambda c: train_audio_model(c, "logmel"),
           "trap": build_trap, "overfit": build_overfit}
    summary = {}
    for step in a.only:
        marker = done_marker(ctx, step)
        if marker.is_file() and not a.force and json.loads(marker.read_text()).get("quick") == a.quick:
            ctx.log.info("%s: already done (use --force to redo)", step)
            summary[step] = {"skipped": True, **json.loads(marker.read_text())}
            continue
        if step != "doodle" and ctx.speech is None:
            ctx.log.error("%s: speech data missing under %s", step, speech_dir)
            summary[step] = {"error": "speech data missing"}
            continue
        t0 = time.time()
        ctx.log.info("=== %s ===", step)
        res = fns[step](ctx)
        res = {**res, "quick": a.quick, "seconds": round(time.time() - t0, 1)}
        write_json(marker, res)
        summary[step] = res
    print("SUMMARY train_showcase:", json.dumps(summary, default=float))
    return 1 if any("error" in v for v in summary.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
