"""Tournament CLI (SPEC 10). Run by the owner over SSH, manually, AFTER the event. The API server never imports or starts this.

    cd ~/ieee-ai-demo
    python3 -m tournament.run --rounds 6 --out results/
    python3 -m tournament.run --rounds 6 --out results/ --stockfish /usr/games/stockfish   # optional shallow Stockfish anchor

Safety: refuses to run while the training queue is non-empty (jobs queued or running) unless --force.
Deterministic: repeat runs on the same models and settings play identical games (no randomness anywhere).
Outputs in --out: ratings.json, standings.csv, games.pgn, results.json, sanity.json, bracket.json (top-8 playoff, precomputed so playback never waits), run_config.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import chess

from tournament.engine.evaluators import EvalError, MaterialEvaluator, NetEvaluator
from tournament.engine.openings import OPENINGS
from tournament.engine.play import GameResult, RandomPlayer, SearchPlayer, StockfishPlayer, play_game
from tournament.engine.search import Searcher
from tournament.ratings import anchor_to, bradley_terry
from tournament.swiss import PairingError, max_rounds, pair_round, round_robin_round, use_round_robin


# ------------------------------------------------------------------ safety
def queue_status(state_dir: Path) -> tuple[bool, int]:
    """(busy, n_open_jobs) from the backend database. A missing database means nothing is training."""
    db = state_dir / "demo.db"
    if not db.is_file():
        return False, 0
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=10)
    try:
        n = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')").fetchone()[0]
    finally:
        conn.close()
    return n > 0, int(n)


# ------------------------------------------------------------------ models
class Entrant:
    def __init__(self, pid: str, name: str, player, meta: dict):
        self.id, self.name, self.player, self.meta = pid, name, player, meta


def _clean(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_") or "model"


def load_entrants(model_dirs: list[Path], device: str, depth_cap_plies: int | None, depth_override: int | None, node_cap: int | None, chunk: int) -> list[Entrant]:
    from app.chess_net.model_io import load_model

    out, used = [], set()
    for d in model_dirs:
        lm = load_model(d, device)
        doc = lm.doc
        pid = _clean(d.name)
        while pid in used:
            pid += "_"
        used.add(pid)
        depth = depth_override or 2 * doc["search_depth_full_moves"]
        if depth_cap_plies:
            depth = min(depth, depth_cap_plies)
        nick, mname = doc.get("nickname", ""), doc.get("model_name", d.name)
        display = _clean(f"{mname}" + (f"_by_{nick}" if nick and nick != mname else ""))
        player = SearchPlayer(display, Searcher(NetEvaluator(lm, chunk), depth_plies=depth, node_cap=node_cap))
        out.append(Entrant(pid, display, player, {"id": pid, "name": display, "nickname": nick, "model_name": mname, "tier": doc.get("tier"),
                                                   "params": doc.get("param_count"), "depth_plies": depth, "dir": str(d)}))
    return out


def reference_bots(node_cap: int | None, stockfish: str | None, sf_depth: int) -> dict[str, object]:
    bots: dict[str, object] = {"bot_random": RandomPlayer("bot_random"),
                               "bot_material": SearchPlayer("bot_material", Searcher(MaterialEvaluator(), depth_plies=2, node_cap=node_cap))}
    if stockfish:
        bots[f"bot_stockfish_d{sf_depth}"] = StockfishPlayer(stockfish, sf_depth, f"bot_stockfish_d{sf_depth}")
    return bots


# ------------------------------------------------------------------ sanity gate
def sanity_gate(entrants: list[Entrant], max_plies: int = 80) -> dict[str, dict]:
    """Each model plays ONE quick game against the random mover (as White, 2-ply search). It passes unless it forfeits (NaN/inf, illegal
    move) or loses to a mover that plays at random."""
    rnd = RandomPlayer("bot_random")
    out = {}
    for e in entrants:
        orig = e.player.searcher.depth
        e.player.searcher.depth = min(orig, 2)
        try:
            g = play_game(e.player, rnd, OPENINGS[0], max_plies=max_plies)
        finally:
            e.player.searcher.depth = orig
        ok = g.forfeit is None and g.score_for(e.player.name) > 0
        out[e.id] = {"passed": ok, "result": g.result, "reason": g.reason, "plies": len(g.moves)}
    return out


# ------------------------------------------------------------------ tournament
def run_swiss(entrants: list[Entrant], bots: dict[str, object], rounds: int, max_plies: int, seed: int, log=print):
    players = [e.id for e in entrants]
    by_id = {e.id: e.player for e in entrants}
    by_id.update(bots)
    scores = {p: 0.0 for p in players}
    played: set[frozenset] = set()
    byes: set[str] = set()
    games: list[dict] = []
    results: list[tuple[str, str, float]] = []
    bot_cycle = sorted(bots)
    n_open = 0

    def play_pair(a: str, b: str, rnd: int, opening_idx: int) -> None:
        nonlocal n_open
        op = OPENINGS[opening_idx % len(OPENINGS)]
        for white, black in ((a, b), (b, a)):
            g = play_game(by_id[white], by_id[black], op, max_plies=max_plies)
            n_open += 1
            games.append({"round": rnd, "white": white, "black": black, "result": g.result, "reason": g.reason, "opening": op[0], "plies": len(g.moves),
                          "forfeit": g.forfeit, "pgn": g.pgn(round_=str(rnd)), "moves": g.moves})
            sa = g.score_for(by_id[a].name)
            results.append((a, b, sa))
            if a in scores: scores[a] += sa
            if b in scores: scores[b] += 1 - sa

    rounds_played = 0
    round_robin = use_round_robin(len(players), rounds)
    if round_robin:
        log(f"{rounds} rounds is close to the number of models: using a fixed round-robin schedule instead of score-based pairing")
    for r in range(1, rounds + 1):
        try:
            pairs, bye = round_robin_round(players, r) if round_robin else pair_round(players, scores, played, byes, seed)
        except PairingError as e:
            log(f"stopping after {rounds_played} round(s): {e}. Everyone has played the same number of games.")
            break
        rounds_played += 1
        log(f"round {r}/{rounds}: {len(pairs)} pairings" + (f", bye -> {bye} plays a reference bot" if bye else ""))
        for i, (a, b) in enumerate(pairs):
            played.add(frozenset((a, b)))
            play_pair(a, b, r, (r - 1) * 5 + i)
        if bye:
            byes.add(bye)
            bot = bot_cycle[(r - 1) % len(bot_cycle)]
            play_pair(bye, bot, r, (r - 1) * 5 + len(pairs))
        log(f"   {len(games)} games so far")
    return games, results, scores, rounds_played


def run_bracket(entrants: list[Entrant], elo: dict[str, float], max_plies: int, log=print) -> dict:
    """Top-8 (or the largest power of two available) single elimination. Each match: two games with colours swapped from the same opening;
    a tie goes to a third game with the higher seed as White; if that is drawn too, the higher seed advances. All moves are stored."""
    ranked = sorted(entrants, key=lambda e: (-elo[e.id], e.id))
    size = 1
    while size * 2 <= min(8, len(ranked)):
        size *= 2
    if size < 2:
        return {"rounds": [], "champion": None, "note": "fewer than 2 models"}
    seeds = ranked[:size]
    order = []  # standard bracket order: 1 v 8, 4 v 5, 2 v 7, 3 v 6
    def slots(n):
        if n == 1:
            return [0]
        prev = slots(n // 2)
        return [x for s in prev for x in (s, n - 1 - s)]
    order = slots(size)
    current = [seeds[i] for i in order]
    rounds, rnd = [], 1
    names = {8: "quarter-finals", 4: "semi-finals", 2: "final"}
    seed_of = {e.id: i + 1 for i, e in enumerate(seeds)}
    while len(current) > 1:
        matches, nxt = [], []
        for m, (a, b) in enumerate(zip(current[::2], current[1::2])):
            op = OPENINGS[(rnd * 3 + m) % len(OPENINGS)]
            gl: list[GameResult] = [play_game(a.player, b.player, op, max_plies=max_plies), play_game(b.player, a.player, op, max_plies=max_plies)]
            sa = sum(g.score_for(a.player.name) for g in gl)
            if sa == 1.0:
                hi, lo = (a, b) if seed_of[a.id] < seed_of[b.id] else (b, a)
                gl.append(play_game(hi.player, lo.player, OPENINGS[(rnd * 3 + m + 1) % len(OPENINGS)], max_plies=max_plies))
                sa = sum(g.score_for(a.player.name) for g in gl)
            if sa == 1.5:  # drawn after the tie-break game too: the higher seed advances
                winner = a if seed_of[a.id] < seed_of[b.id] else b
            else:
                winner = a if sa > 1.5 else b if sa < 1.5 else (a if seed_of[a.id] < seed_of[b.id] else b)
            matches.append({"a": a.id, "b": b.id, "seed_a": seed_of[a.id], "seed_b": seed_of[b.id], "winner": winner.id, "score_a": sa,
                            "games": [{"white": next(e.id for e in (a, b) if e.player.name == g.white), "black": next(e.id for e in (a, b) if e.player.name == g.black),
                                       "result": g.result, "reason": g.reason, "opening": g.opening, "moves_uci": g.moves, "moves_san": g.san_moves()} for g in gl]})
            nxt.append(winner)
            log(f"   {names.get(len(current), 'round')}: {a.name} vs {b.name} -> {winner.name}")
        rounds.append({"name": names.get(len(current), f"round {rnd}"), "matches": matches})
        current, rnd = nxt, rnd + 1
    return {"rounds": rounds, "champion": current[0].id, "seeds": {e.id: i + 1 for i, e in enumerate(seeds)}}


# ------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    s = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, required=True, help="Swiss rounds; every model plays exactly 2 x rounds games")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models-dir", type=Path, default=s.models_dir)
    ap.add_argument("--state-dir", type=Path, default=s.state_dir)
    ap.add_argument("--extra-model", type=Path, action="append", default=[], help="a model folder outside --models-dir (e.g. baselines/default-config)")
    ap.add_argument("--exclude", action="append", default=[], help="folder names to leave out")
    ap.add_argument("--force", action="store_true", help="run even though the training queue is not empty")
    ap.add_argument("--override-sanity", action="store_true", help="keep models that fail the sanity gate")
    ap.add_argument("--stockfish", default=None, help="path to a stockfish binary (optional anchor)")
    ap.add_argument("--stockfish-depth", type=int, default=2)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto")
    ap.add_argument("--max-plies", type=int, default=150, help="games are adjudicated by material at this ply count")
    ap.add_argument("--depth-cap-plies", type=int, default=None, help="cap every model's search depth (SPEC 8.5: e.g. 4 if 6 is too slow)")
    ap.add_argument("--depth-override-plies", type=int, default=None, help="all models search this deep (tests / quick runs)")
    ap.add_argument("--node-cap", type=int, default=None, help="per-move cap on evaluated positions (iterative deepening keeps the last completed depth)")
    ap.add_argument("--eval-chunk", type=int, default=64, help="fixed batch size of leaf evaluations")
    ap.add_argument("--seed", type=int, default=0, help="only influences the order of the first pairings")
    ap.add_argument("--no-bracket", action="store_true")
    a = ap.parse_args(argv)

    busy, n_open = queue_status(a.state_dir)
    if busy and not a.force:
        print(f"REFUSING TO RUN: the training queue is not empty ({n_open} job(s) queued or running). Wait for training to finish, or pass --force.")
        return 3

    import torch
    from app.chess_net.model_io import list_models

    device = "cuda" if a.device == "auto" and torch.cuda.is_available() else "cpu" if a.device == "auto" else a.device
    dirs = [d for d in list_models(a.models_dir) if d.name not in a.exclude] + [p for p in a.extra_model if p.is_dir()]
    if len(dirs) < 2:
        print(f"need at least 2 models, found {len(dirs)} in {a.models_dir}")
        return 2
    entrants = load_entrants(dirs, device, a.depth_cap_plies, a.depth_override_plies, a.node_cap, a.eval_chunk)
    print(f"{len(entrants)} models on {device}; depths (plies): " + ", ".join(f"{e.name}={e.meta['depth_plies']}" for e in entrants))

    sanity = sanity_gate(entrants)
    failed = [e for e in entrants if not sanity[e.id]["passed"]]
    for e in failed:
        print(f"SANITY GATE FAILED: {e.name} ({sanity[e.id]['reason']}, {sanity[e.id]['result']})" + (" -> kept (--override-sanity)" if a.override_sanity else " -> excluded"))
    if failed and not a.override_sanity:
        entrants = [e for e in entrants if sanity[e.id]["passed"]]
    if len(entrants) < 2:
        print("fewer than 2 models passed the sanity gate")
        return 2
    try:
        pair_round([e.id for e in entrants], {}, set(), set(), a.seed)
        if a.rounds > max_rounds(len(entrants)):
            raise PairingError(f"{a.rounds} rounds need more models: with {len(entrants)} models at most {max_rounds(len(entrants))} rounds avoid repeat pairings")
    except PairingError as e:
        print("ERROR:", e)
        return 2

    a.out.mkdir(parents=True, exist_ok=True)
    bots = reference_bots(a.node_cap, a.stockfish, a.stockfish_depth)
    t0 = time.time()
    games, results, scores, rounds_played = run_swiss(entrants, bots, a.rounds, a.max_plies, a.seed)
    ids = [e.id for e in entrants] + sorted(bots)
    elo = bradley_terry(results, ids)
    elo = anchor_to(elo, "bot_random", 0.0)

    per = {i: {"games": 0, "w": 0, "d": 0, "l": 0, "score": 0.0} for i in ids}
    for gm in games:
        for side, other in (("white", "black"), ("black", "white")):
            pid = gm[side]
            sc = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}[gm["result"]] if side == "white" else {"1-0": 0.0, "0-1": 1.0, "1/2-1/2": 0.5}[gm["result"]]
            per[pid]["games"] += 1; per[pid]["score"] += sc
            per[pid]["w" if sc == 1 else "d" if sc == 0.5 else "l"] += 1
    meta = {e.id: e.meta for e in entrants}
    rows = sorted(({"id": i, **meta.get(i, {"name": i, "tier": "reference bot"}), "elo": round(elo[i], 1), **per[i]} for i in ids), key=lambda r: (-r["elo"], r["id"]))
    (a.out / "ratings.json").write_text(json.dumps({"anchor": "bot_random = 0", "rounds": rounds_played, "games_per_model": 2 * rounds_played, "ratings": rows}, indent=2))
    with (a.out / "standings.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "name", "nickname", "tier", "params", "elo", "games", "wins", "draws", "losses", "score"])
        for k, r in enumerate(rows, 1):
            w.writerow([k, r["id"], r.get("name"), r.get("nickname", ""), r.get("tier"), r.get("params", ""), r["elo"], r["games"], r["w"], r["d"], r["l"], r["score"]])
    (a.out / "games.pgn").write_text("".join(g["pgn"] for g in games))
    (a.out / "results.json").write_text(json.dumps([{k: v for k, v in g.items() if k != "pgn"} for g in games]))
    (a.out / "sanity.json").write_text(json.dumps(sanity, indent=2))
    if not a.no_bracket:
        bracket = run_bracket([e for e in entrants], {e.id: elo[e.id] for e in entrants}, a.max_plies)
        (a.out / "bracket.json").write_text(json.dumps(bracket))
    (a.out / "run_config.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "python_chess": chess.__version__, "device": device,
                                                        "games": len(games), "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"\n{'rank':<5}{'model':<34}{'tier':<8}{'elo':>7}{'score':>8}  (anchor: random mover = 0)")
    for k, r in enumerate(rows, 1):
        print(f"{k:<5}{str(r.get('name'))[:32]:<34}{str(r.get('tier') or 'bot'):<8}{r['elo']:>7.0f}{r['score']:>5.1f}/{r['games']}")
    print(f"\n{len(games)} games in {time.time() - t0:.0f}s. Wrote ratings.json, standings.csv, games.pgn, results.json, sanity.json" + ("" if a.no_bracket else ", bracket.json") + f" to {a.out}")
    for b in bots.values():
        if hasattr(b, "close"):
            b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
