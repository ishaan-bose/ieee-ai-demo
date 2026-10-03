#!/usr/bin/env python3
"""Benchmark report (SPEC section 11): prints ONE text block you can paste back, and saves it to ~/demo/logs/benchmark_report.txt.

    cd ~/ieee-ai-demo/backend
    python3 scripts/benchmark.py                 # full run on the GPU box (about 15-25 minutes)
    python3 scripts/benchmark.py --quick         # tiny sizes, for a CPU rehearsal or a smoke test (about 1-2 minutes)
    python3 scripts/benchmark.py --sections 1,6  # only some sections

Sections: 1 throughput, 2 doodles, 3 race drama, 4 audio, 5 chess search + strength, 6 budget calibration, 7 browser inference.
Section 6 writes STATE_DIR/calibration.json, which the backend reads (BUDGET_FLOPS, TIME_CAP_SECONDS).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from app.chess_net import encoder as E  # noqa: E402
from app.chess_net.config import flops_per_sample, resolve_config, tier_table  # noqa: E402
from app.chess_net.model import build_model  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.data.loaders import LichessData, QuickDrawData, SpeechData  # noqa: E402
from app.device import detect_device  # noqa: E402
from app.training import audio_models as am  # noqa: E402
from app.training.chess_data import ChessData  # noqa: E402
from app.training.classifier import accuracy, build_mlp_classifier  # noqa: E402
from app.training.race import DoodleData, Lane, LaneConfig, run_race  # noqa: E402
from app.training.race_grid import lane_configs  # noqa: E402
from app.training.trainer import CompetitionTrainer  # noqa: E402
from scripts._common import setup_logging  # noqa: E402

TARGET_MINUTES = 9.0  # SPEC 6.2: median config takes 8-10 minutes uncontended

MEDIAN = {"layers": 4, "width": 1024, "batch_size": 256}
TINY = {"layers": 2, "width": 64}
BIG = {"layers": 7, "width": 2048}  # ~27M params, just under the 30M cap


class Ctx:
    def __init__(self, a):
        self.a, self.quick = a, a.quick
        self.s = get_settings()
        self.dev = detect_device()
        self.device = torch.device(self.dev.device)
        self.facts: dict = {}
        self.verify: list[str] = []
        self.lines: list[str] = []

    def out(self, line: str = ""):
        self.lines.append(line)
        print(line, flush=True)

    def t(self, full: float, quick: float) -> float:
        return quick if self.quick else full

    def chess_dir(self) -> Path | None:
        d = self.s.data_dir / "lichess"
        return d if (d / "boards.npy").is_file() else None


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def timed_steps(fn, seconds: float, dev) -> tuple[float, int]:
    fn(); sync(dev)  # warm-up
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        fn(); n += 1
    sync(dev)
    return time.perf_counter() - t0, n


# ------------------------------------------------------------------ 1. throughput
def section1(c: Ctx):
    c.out("1. DEVICE AND THROUGHPUT (samples/s, synthetic features, fp32 + TF32 training)")
    d = c.dev
    c.out(f"   device={d.device} gpu={d.gpu_name} vram_gb={d.vram_gb} free_gb={d.vram_free_gb} torch={d.torch_version} cuda={d.cuda_version}")
    c.out(f"   NUM_WORKERS={c.s.num_workers} (env; os.cpu_count would lie)")
    sec = c.t(1.0, 0.15)
    rows = []
    for name, base in (("tiny", TINY), ("median", MEDIAN), ("big~27M", BIG)):
        cfg, err = resolve_config({**base})
        assert not err, err
        model = build_model(cfg).to(c.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        c.out(f"   {name}: {cfg['param_count']:,} params, {flops_per_sample(cfg['in_dim'], cfg['widths']) / 1e6:.1f} MFLOPs/sample, tier {cfg['tier']}")
        for bs in ([1, 64, 256] if c.quick else [1, 64, 256, 1024, 4096]):
            if name == "big~27M" and bs == 1 and not c.quick:
                continue
            x = torch.randn(bs, cfg["in_dim"], device=c.device)
            y = torch.rand(bs, device=c.device)

            def step():
                loss = ((model(x) - y) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            try:
                dt, n = timed_steps(step, sec, c.device)
                sps = n * bs / dt
                rows.append((name, bs, sps))
                c.out(f"      batch {bs:>5}: {sps:>12,.0f} samples/s  ({n / dt:,.0f} steps/s)")
            except torch.cuda.OutOfMemoryError:
                c.out(f"      batch {bs:>5}: OOM")
                torch.cuda.empty_cache()
        # full batch with gradient accumulation: 16384 samples as chunks of 2048
        total, chunk = (4096 if c.quick else 16384), (1024 if c.quick else 2048)
        xs = torch.randn(chunk, cfg["in_dim"], device=c.device)
        ys = torch.rand(chunk, device=c.device)

        def full_step():
            opt.zero_grad(set_to_none=True)
            for _ in range(total // chunk):
                (((model(xs) - ys) ** 2).mean() * chunk / total).backward()
            opt.step()
        try:
            dt, n = timed_steps(full_step, sec, c.device)
            c.out(f"      full batch {total} (accumulated in {total // chunk} chunks): {n * total / dt:,.0f} samples/s, {dt / n:.3f} s/update")
        except torch.cuda.OutOfMemoryError:
            c.out("      full batch: OOM"); torch.cuda.empty_cache()
        del model, opt
    # encoder throughput
    bd = torch.randint(0, 13, (4096, 64), dtype=torch.uint8, device=c.device)
    st = torch.randint(0, 2, (4096,), dtype=torch.uint8, device=c.device)
    ca = torch.randint(0, 16, (4096,), dtype=torch.uint8, device=c.device)
    ep = torch.full((4096,), -1, dtype=torch.int8, device=c.device)
    for label, ex in (("planes+stm/castle", E.Extras()), ("all extras (attack maps)", E.Extras(True, True, True, True))):
        dt, n = timed_steps(lambda: E.encode(bd, st, ca, ep, ex), c.t(1.0, 0.15), c.device)
        c.out(f"   encoder [{label}]: {n * 4096 / dt:,.0f} positions/s")
    c.facts["throughput"] = rows


# ------------------------------------------------------------------ 2/3 doodles + races
def doodle_data(c: Ctx) -> DoodleData | None:
    qd = QuickDrawData(c.s.data_dir / "quickdraw")
    if not qd.has_tensors(("train", "val")):
        return None
    return DoodleData.from_dir(qd, c.device)


def train_for(model, data: DoodleData, seconds: float, lr: float = 2e-3, batch: int = 128) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        x, y = data.batch(torch.randint(data.n, (batch,), generator=g))
        loss = torch.nn.functional.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return accuracy(model, data.val_x, data.val_y)


def section2(c: Ctx):
    c.out("2. DOODLES: linear-only vs ReLU (10 classes)")
    data = doodle_data(c)
    if data is None:
        c.out("   SKIPPED: quickdraw/tensors missing (run scripts/rasterize_quickdraw.py)")
        return
    secs = c.t(20.0, 2.0)
    lin = build_mlp_classifier(784, [], 10, seed=0).to(c.device)
    relu = build_mlp_classifier(784, [128, 64], 10, "relu", seed=0).to(c.device)
    a_lin, a_relu = train_for(lin, data, secs), train_for(relu, data, secs)
    c.out(f"   after {secs:.0f}s each: linear-only val acc {a_lin:.3f} | ReLU MLP 784-128-64-10 val acc {a_relu:.3f}")
    classes = QuickDrawData(c.s.data_dir / "quickdraw").classes
    with torch.no_grad():
        pred = relu(data.val_x).argmax(1).cpu().numpy()
    true = data.val_y.cpu().numpy()
    cm = np.zeros((10, 10), int)
    for t_, p_ in zip(true, pred):
        cm[t_, p_] += 1
    c.out("   confusion matrix of the ReLU model (rows = true, cols = predicted):")
    c.out("   " + " " * 11 + "".join(f"{n[:5]:>6}" for n in classes))
    for i, n in enumerate(classes):
        c.out(f"   {n:>10} " + "".join(f"{v:>6}" for v in cm[i]))
    recall = cm.diagonal() / np.maximum(cm.sum(1), 1)
    pairs = sorted(((cm[i, j] / max(cm[i].sum(), 1), classes[i], classes[j]) for i in range(10) for j in range(10) if i != j), reverse=True)[:5]
    c.out("   most confused (true -> predicted, share of true class): " + "; ".join(f"{a}->{b} {r:.0%}" for r, a, b in pairs))
    c.out(f"   per-class recall: " + ", ".join(f"{n} {r:.0%}" for n, r in zip(classes, recall)))
    easy = [n for n, r in zip(classes, recall) if r > 0.97]
    hard = [n for n, r in zip(classes, recall) if r < 0.6]
    c.out(f"   too easy (>97%): {easy or 'none'} | too hard (<60%): {hard or 'none'}  -> swap candidates")
    c.facts["doodle"] = {"linear": a_lin, "relu": a_relu}
    gap = a_relu - a_lin
    c.verify.append(f"{'PASS' if gap > 0.10 else 'NEEDS CHANGE'}: ReLU beats linear-only by {gap:.1%} on doodles (Act 1/2 story needs a clear gap; short runs here)")


def section3(c: Ctx):
    c.out("3. RACE DRAMA: accuracy vs time for every planned option (one lane each, same data and seed)")
    data = doodle_data(c)
    if data is None:
        c.out("   SKIPPED: quickdraw/tensors missing")
        return
    secs = c.t(20.0, 2.0)
    c.out(f"   each option runs {secs:.0f}s. columns: acc at 20/40/60/80/100% of the time | updates | flags")
    for kind in ("loss", "lr", "batch"):
        c.out(f"   -- {kind} race options")
        finals = {}
        opts = lane_configs(kind)
        if c.quick:
            opts = opts[::3]
        for key, cfg in opts:
            lane = Lane(LaneConfig(id=key, **cfg), data, seed=0)
            curve = []
            run_race([lane], secs, lambda k, d: curve.append(d["lanes"][0]["acc"]) if k == "tick" else None, lambda: False,
                     tick_s=secs / 5.0, slice_s=0.1)
            accs = curve[-5:] if len(curve) >= 5 else curve + [curve[-1]] * (5 - len(curve))
            final = accs[-1]
            finals[key] = final
            snap = lane.snapshot()
            flags = []
            if final > 0.2 and accs[0] >= 0.9 * final:
                flags.append("finishes within seconds (curve already flat)")
            if snap["loss"] is None or not np.isfinite(snap["loss"]):
                flags.append("DIVERGED (NaN loss)")
            c.out(f"      {key:<14} " + " ".join(f"{a:.3f}" for a in accs) + f" | {lane.updates:>8,} updates | {'; '.join(flags)}")
        spread = max(finals.values()) - min(finals.values())
        verdict = "PASS" if spread >= 0.05 else "NEEDS CHANGE"
        c.out(f"      -> spread of final accuracy across {kind} options: {spread:.3f}")
        c.verify.append(f"{verdict}: {kind} race options separate (final-accuracy spread {spread:.1%}; need >= 5%)")


# ------------------------------------------------------------------ 4. audio
def section4(c: Ctx):
    c.out("4. AUDIO: raw-waveform vs log-mel accuracy (10 words)")
    root = c.s.data_dir / "speech" / "processed"
    if not (root / "X_train.npy").is_file():
        c.out("   SKIPPED: speech data missing")
        return
    sp = SpeechData(root)
    tr, va, te = (sp.word_indices(s) for s in ("train", "val", "test"))
    if c.quick:
        tr, va, te = tr[:1500], va[:300], te[:300]
    ep = 2 if c.quick else 8
    res = {}
    for kind, builder in (("raw", am.build_raw_model), ("logmel", am.build_logmel_model)):
        f = {s: am.make_features(kind, sp.X(s)[idx], c.device) for s, idx in (("train", tr), ("val", va), ("test", te))}
        y = {s: sp.y(s)[idx] for s, idx in (("train", tr), ("val", va), ("test", te))}
        t0 = time.time()
        hist = am.train_audio(builder(), f["train"], y["train"], f["val"], y["val"], epochs=ep, device=c.device)
        model = builder().to(c.device)
        # report test accuracy of a fresh run's final weights via history (val used for model selection only)
        res[kind] = hist[-1]["val_acc"]
        c.out(f"   {kind:>7}: {ep} epochs in {time.time() - t0:.0f}s, train acc {hist[-1]['train_acc']:.3f}, val acc {hist[-1]['val_acc']:.3f}")
        del model
    gap = res["logmel"] - res["raw"]
    c.out(f"   gap (log-mel minus raw): {gap:+.3f}")
    c.facts["audio"] = res
    c.verify.append(f"{'PASS' if gap > 0.15 else 'NEEDS CHANGE'}: Act 3 accuracy gap log-mel vs raw = {gap:.1%} (want >= 15 points; short training here)")


# ------------------------------------------------------------------ 5. chess
def section5(c: Ctx):
    c.out("5. CHESS: search speed per tier, training throughput, default-config strength")
    try:
        from tournament.engine.evaluators import MaterialEvaluator, NetEvaluator
        from tournament.engine.play import RandomPlayer, SearchPlayer, play_game
        from tournament.engine.search import Searcher
        import chess
        from app.chess_net.config import DEFAULT_CONFIG
        from app.chess_net.model_io import LoadedModel
    except ImportError as e:
        c.out(f"   SKIPPED: tournament engine not importable ({e}); pip install -r tournament/requirements.txt")
        return
    fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 0 8",
            "2r3k1/pp3ppp/4p3/3pP3/3P4/1P3Q2/P4PPP/6K1 w - - 0 25",
            "8/5pk1/6p1/8/3R4/6P1/5PK1/3r4 w - - 0 40"]
    tier_cfgs = {"Light": {"layers": 3, "width": 512}, "Medium": {"layers": 4, "width": 1024}, "Heavy": {"layers": 7, "width": 2048}}
    node_cap = c.a.node_cap
    c.out(f"   search speed (random-init nets of the typical size of each tier; batched fp32 leaf evaluation; 5 positions; node cap {node_cap}):")
    suggestions = {}
    for tier in tier_table():
        raw = tier_cfgs.get(tier["name"], {"layers": 3, "width": 256})
        cfg, _ = resolve_config(raw)
        model = build_model(cfg).to(c.device).eval()
        doc = {"config": cfg, "search_depth_full_moves": tier["search_depth_full_moves"], "model_name": tier["name"]}
        ev = NetEvaluator(LoadedModel(model, doc, c.device))
        for depth_full in ([tier["search_depth_full_moves"]] if c.quick else [1, 2, 3]):
            s = Searcher(ev, depth_plies=2 * depth_full, node_cap=node_cap)
            t0, nodes = time.perf_counter(), 0
            for f in fens[:2] if c.quick else fens:
                _, _, st = s.best_move(chess.Board(f))
                nodes += st["nodes"]
            dt = time.perf_counter() - t0
            spm = dt / len(fens[:2] if c.quick else fens)
            games_h = 3600.0 / max(spm * 80, 1e-9)  # ~80 plies per game, both sides searched
            c.out(f"      {tier['name']:<7} ({cfg['param_count']:,} params) depth {depth_full} full moves ({2 * depth_full} plies): "
                  f"{nodes / dt:>10,.0f} nodes/s | {spm:6.2f} s/move | ~{games_h:,.0f} games/hour (single stream)")
            if spm <= c.a.sec_per_move:
                suggestions[tier["name"]] = max(suggestions.get(tier["name"], 0), depth_full)
    sug = {t["name"]: suggestions.get(t["name"], 1) for t in tier_table()}
    c.out(f"   target <= {c.a.sec_per_move}s/move. Deepest affordable depth per tier: {sug}  (tiers.json currently "
          f"{ {t['name']: t['search_depth_full_moves'] for t in tier_table()} })")
    ok = all(sug[t["name"]] >= t["search_depth_full_moves"] for t in tier_table())
    c.verify.append(f"{'PASS' if ok else 'NEEDS CHANGE'}: tier depths affordable at <= {c.a.sec_per_move}s/move; affordable = {sug}"
                    + ("" if ok else " -> shift tiers.json down (SPEC 8.5) or set a node cap"))
    suggested = {"tiers": [{**t, "search_depth_full_moves": sug[t["name"]]} for t in tier_table()]}
    c.s.state_dir.mkdir(parents=True, exist_ok=True)
    (c.s.state_dir / "tiers.suggested.json").write_text(json.dumps(suggested, indent=2))
    c.out(f"   wrote suggested depths to {c.s.state_dir / 'tiers.suggested.json'} (copy into shared/tiers.json if you agree)")

    # strength of the default config
    ld = c.chess_dir()
    if ld is None:
        c.out("   strength: SKIPPED (lichess data missing)")
        return
    secs = c.t(c.a.strength_seconds, 4.0)
    data = ChessData(LichessData(ld), c.device, val_rows=20_000 if not c.quick else 500)
    cfg, _ = resolve_config(DEFAULT_CONFIG)
    tr = CompetitionTrainer(cfg, data, device=c.device, budget_flops=1e30, time_cap_s=secs, metrics_every_s=1e9)
    reason = tr.run(lambda: False)
    ev = tr.evaluate()
    c.out(f"   default config trained {tr.active_s:.0f}s ({tr.samples_seen:,} samples, {tr.samples_seen / max(tr.active_s, 1e-9):,.0f}/s, stop={reason}): "
          f"val_mse={ev['val_mse']:.4f}")
    doc = {"config": cfg, "search_depth_full_moves": 2, "model_name": "default"}
    net = SearchPlayer("default", Searcher(NetEvaluator(LoadedModel(tr.model.eval(), doc, c.device)), depth_plies=2, node_cap=node_cap))
    mat = SearchPlayer("material", Searcher(MaterialEvaluator(), depth_plies=2, node_cap=node_cap))
    rnd = RandomPlayer("random")
    games = 2 if c.quick else c.a.games
    for opp in (rnd, mat):
        w = d = l = 0
        for g in range(games):
            white, black = (net, opp) if g % 2 == 0 else (opp, net)
            r = play_game(white, black, max_plies=60 if c.quick else 150)
            score = r.score_for(net.name)
            w, d, l = w + (score == 1), d + (score == 0.5), l + (score == 0)
        c.out(f"   default net (2 plies) vs {opp.name}: +{w} ={d} -{l} over {games} games")
        c.facts.setdefault("strength", {})[opp.name] = (w, d, l)
    st = c.facts["strength"]
    wr = st["random"][0] / max(1, sum(st["random"]))
    c.verify.append(f"{'PASS' if wr >= 0.5 or c.quick else 'NEEDS CHANGE'}: default config beats the random bot in {wr:.0%} of games "
                    f"(expected 'meh' but better than random; short training here)")


# ------------------------------------------------------------------ 6. calibration
def section6(c: Ctx):
    c.out("6. BUDGET CALIBRATION (median config ~4 layers x 1024, batch 256, uncontended)")
    ld = c.chess_dir()
    tmp = None
    if ld is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tests.fake_data import write_lichess
        import tempfile
        tmp = tempfile.mkdtemp(prefix="bench_fake_")
        ld = write_lichess(Path(tmp), 20_000 if c.quick else 200_000)
        c.out("   WARNING: real lichess data not found; calibrating on FAKE data (throughput is comparable, loss is not)")
    data = ChessData(LichessData(ld), c.device, val_rows=2000)
    cfg, _ = resolve_config(MEDIAN)
    secs = c.t(c.a.calib_seconds, 4.0)
    tr = CompetitionTrainer(cfg, data, device=c.device, budget_flops=1e30, time_cap_s=secs, metrics_every_s=1e9)
    tr.run(lambda: False)
    sps = tr.samples_seen / max(tr.active_s, 1e-9)
    fps = flops_per_sample(cfg["in_dim"], cfg["widths"])
    budget = fps * sps * TARGET_MINUTES * 60
    cap = 3 * TARGET_MINUTES * 60
    c.out(f"   median config: {cfg['param_count']:,} params, {fps / 1e6:.1f} MFLOPs/sample, measured {sps:,.0f} samples/s over {tr.active_s:.0f}s "
          f"= {fps * sps / 1e12:.2f} TFLOP/s (data pipeline included, data resident on device: {data.resident})")
    c.out(f"   => BUDGET_FLOPS = {budget:.3e}  (about {budget / fps / 1e6:,.0f}M samples = {budget / fps / 12e6:.1f} epochs of 12M positions, {TARGET_MINUTES:.0f} min)")
    c.out(f"   => TIME_CAP_SECONDS = {cap:.0f}  (3x nominal; should rarely fire)")
    c.s.state_dir.mkdir(parents=True, exist_ok=True)
    calib = {"budget_flops": budget, "time_cap_seconds": cap, "samples_per_second": sps, "device": c.dev.device, "gpu_name": c.dev.gpu_name,
             "quick_run": c.quick, "fake_data": tmp is not None, "measured_at": time.time()}
    if c.quick or tmp is not None:
        c.out("   (quick/fake-data run: NOT writing calibration.json; do the real run on the GPU box)")
    else:
        c.s.calibration_path.write_text(json.dumps(calib, indent=2))
        c.out(f"   wrote {c.s.calibration_path}: the backend picks it up on next start (env BUDGET_FLOPS overrides)")
    c.out(f"   or set manually:  export BUDGET_FLOPS={budget:.3e} TIME_CAP_SECONDS={cap:.0f}")
    c.facts["calibration"] = calib
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------ 7. browser inference
def section7(c: Ctx):
    c.out("7. BROWSER INFERENCE TIME (Node/V8 uses the same engine as Chrome; the number that matters is YOUR LAPTOP)")
    models = REPO / "frontend" / "public" / "models" / "manifest.json"
    if not models.is_file():
        c.out("   SKIPPED: frontend/public/models/manifest.json missing (run scripts/export_weights.py first)")
    elif shutil.which("npx") is None:
        c.out("   node/npx not found on this machine. On the LAPTOP run:  cd frontend && npm run bench:inference")
    else:
        try:
            r = subprocess.run(["npx", "tsx", "scripts/bench-inference.ts"], cwd=REPO / "frontend", capture_output=True, text=True, timeout=300)
            for ln in (r.stdout or r.stderr).strip().splitlines():
                c.out("   " + ln)
            if r.returncode == 0:
                c.verify.append("INFO: browser inference timings printed in section 7 (run again on the laptop; want < 50 ms per forward pass)")
        except Exception as e:  # noqa: BLE001
            c.out(f"   could not run the node benchmark: {e}")


SECTIONS = {1: section1, 2: section2, 3: section3, 4: section4, 5: section5, 6: section6, 7: section7}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sections", default="1,2,3,4,5,6,7", help="comma list; run_on_server.sh runs 1-6 first and 7 after the models are exported")
    ap.add_argument("--calib-seconds", type=float, default=30.0)
    ap.add_argument("--strength-seconds", type=float, default=120.0)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--node-cap", type=int, default=20000, help="per-move node cap for the search benchmark (0 = none)")
    ap.add_argument("--sec-per-move", type=float, default=1.5)
    a = ap.parse_args(argv)
    a.node_cap = a.node_cap or None
    c = Ctx(a)
    log = setup_logging("benchmark", c.s)
    t0 = time.time()
    c.out("=" * 72)
    c.out(f"BENCHMARK REPORT  ({time.strftime('%Y-%m-%d %H:%M:%S')})  quick={a.quick}")
    c.out("=" * 72)
    for n in [int(x) for x in a.sections.split(",")]:
        c.out()
        try:
            SECTIONS[n](c)
        except Exception as e:  # noqa: BLE001  one failing section must not hide the others
            c.out(f"   SECTION {n} FAILED: {type(e).__name__}: {e}")
            log.error(traceback.format_exc())
            c.verify.append(f"ERROR: section {n} failed ({type(e).__name__}: {e})")
    c.out()
    c.out("[verify] ITEMS FROM SPEC.md")
    for v in c.verify or ["(none evaluated)"]:
        c.out("   " + v)
    c.out(f"total time {time.time() - t0:.0f}s")
    c.out("=" * 72)
    name = "benchmark_browser.txt" if a.sections.strip() == "7" else "benchmark_report.txt"
    (c.s.log_dir / name).write_text("\n".join(c.lines) + "\n")
    print(f"(report saved to {c.s.log_dir / name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
