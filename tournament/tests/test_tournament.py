import json
import sqlite3
import random
from pathlib import Path

import chess
import pytest

from app import db as backend_db
from tournament import run as trun
from tournament.engine.play import RandomPlayer
from tournament.ratings import anchor_to, bradley_terry
from tournament.swiss import PairingError, max_rounds, pair_round
from tournament.tests.conftest import make_fake_model

FAST = ["--depth-override-plies", "1", "--max-plies", "30", "--no-bracket"]


# ---------------------------------------------------------------- swiss
@pytest.mark.parametrize("n,rounds", [(8, 3), (9, 3), (6, 2), (7, 2), (35, 6), (35, 16), (36, 17)])
def test_pairing_never_repeats_and_everyone_plays_every_round(n, rounds):
    rng = random.Random(n)
    players = [f"m{i}" for i in range(n)]
    scores = {p: 0.0 for p in players}
    played, byes = set(), set()
    for _ in range(rounds):
        pairs, bye = pair_round(players, scores, played, byes, seed=3)
        seen = [x for pr in pairs for x in pr] + ([bye] if bye else [])
        assert sorted(seen) == sorted(players)  # exactly one slot per player per round
        assert (bye is None) == (n % 2 == 0)
        for a, b in pairs:
            assert frozenset((a, b)) not in played
            played.add(frozenset((a, b)))
            scores[a] += rng.choice([0, 0.5, 1]); scores[b] += rng.choice([0, 0.5, 1])
        if bye:
            assert bye not in byes or len(byes) >= n  # byes go round before anyone repeats
            byes.add(bye)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 9, 12])
def test_round_robin_covers_every_pair_exactly_once_and_rests_everyone_once(n):
    from tournament.swiss import round_robin_round, use_round_robin
    players = [f"m{i}" for i in range(n)]
    rounds = max_rounds(n)
    seen, byes = set(), []
    for r in range(1, rounds + 1):
        pairs, bye = round_robin_round(players, r)
        flat = [x for pr in pairs for x in pr] + ([bye] if bye else [])
        assert sorted(flat) == sorted(players)
        for a, b in pairs:
            assert frozenset((a, b)) not in seen
            seen.add(frozenset((a, b)))
        if bye:
            byes.append(bye)
    assert len(seen) == n * (n - 1) // 2 if rounds == max_rounds(n) and n % 2 == 0 else len(seen) <= n * (n - 1) // 2
    assert len(byes) == len(set(byes))  # nobody rests twice
    assert use_round_robin(8, 7) and not use_round_robin(35, 6) and use_round_robin(4, 3) and not use_round_robin(4, 1)


def test_pairing_is_deterministic_and_can_run_out():
    p = [f"m{i}" for i in range(4)]
    a = pair_round(p, {x: 0 for x in p}, set(), set(), 1)
    assert a == pair_round(p, {x: 0 for x in p}, set(), set(), 1)
    played = {frozenset(c) for c in [("m0", "m1"), ("m0", "m2"), ("m0", "m3"), ("m1", "m2"), ("m1", "m3"), ("m2", "m3")]}
    with pytest.raises(PairingError):
        pair_round(p, {}, played, set(), 1)
    assert max_rounds(6) == 5 and max_rounds(7) == 7 and max_rounds(2) == 1


def test_everyone_gets_exactly_two_games_per_round():
    ids = [f"p{i}" for i in range(7)]  # odd: one bye per round against a reference bot
    entrants = [trun.Entrant(i, i, RandomPlayer(i), {}) for i in ids]
    bots = {"bot_random": RandomPlayer("bot_random"), "bot_b": RandomPlayer("bot_b")}
    games, results, scores, played_rounds = trun.run_swiss(entrants, bots, 5, 24, 0, log=lambda *_: None)
    assert played_rounds == 5
    for i in ids:
        mine = [g for g in games if i in (g["white"], g["black"])]
        assert len(mine) == 10, i  # exactly 2N games
        assert sum(g["white"] == i for g in mine) == 5  # and colours balanced
    assert sum(1 for g in games if g["white"].startswith("bot") or g["black"].startswith("bot")) == 10  # 5 byes x 2 games
    # the two games of a pairing share an opening and swap colours
    by_round = {}
    for g in games:
        by_round.setdefault((g["round"], frozenset((g["white"], g["black"]))), []).append(g)
    assert all(len(v) == 2 and v[0]["opening"] == v[1]["opening"] and v[0]["white"] == v[1]["black"] for v in by_round.values())


# ---------------------------------------------------------------- ratings
def test_bradley_terry_recovers_strengths_and_handles_perfect_records():
    true = {"a": 600, "b": 300, "c": 0, "d": -300}
    rng = random.Random(1)
    res = []
    for _ in range(40):
        for x in true:
            for y in true:
                if x < y:
                    p = 1 / (1 + 10 ** ((true[y] - true[x]) / 400))
                    res.append((x, y, 1.0 if rng.random() < p else 0.0))
    elo = anchor_to(bradley_terry(res, list(true)), "d", -300)
    assert [k for k, _ in sorted(elo.items(), key=lambda kv: -kv[1])] == ["a", "b", "c", "d"]
    for k in true:
        assert abs(elo[k] - true[k]) < 90, (k, elo[k])
    # a player who won every game must still get a finite rating
    perfect = [("x", "y", 1.0)] * 10 + [("x", "z", 1.0)] * 10 + [("y", "z", 0.5)] * 4
    e = bradley_terry(perfect, ["x", "y", "z"])
    assert all(abs(v) < 5000 for v in e.values()) and e["x"] > e["y"] >= e["z"] - 1
    # draws count half
    even = bradley_terry([("p", "q", 0.5)] * 10, ["p", "q"])
    assert abs(even["p"] - even["q"]) < 1e-6
    assert anchor_to(e, "z", 0.0)["z"] == 0.0


# ---------------------------------------------------------------- safety + gate
def make_queue_db(state: Path, status: str | None):
    state.mkdir(parents=True, exist_ok=True)
    backend_db.init_db(state / "demo.db")
    if status:
        c = backend_db.connect(state / "demo.db")
        c.execute("INSERT INTO jobs (kind, status, queue_order, created_at, updated_at) VALUES ('competition', ?, 1, 0, 0)", (status,))
        c.close()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_refuses_to_run_while_the_queue_is_not_empty(models_dir, tmp_path, capsys, status):
    state = tmp_path / "state"
    make_queue_db(state, status)
    rc = trun.main(["--rounds", "1", "--out", str(tmp_path / "o"), "--models-dir", str(models_dir), "--state-dir", str(state)] + FAST)
    assert rc == 3 and "REFUSING TO RUN" in capsys.readouterr().out and not (tmp_path / "o").exists()
    rc = trun.main(["--rounds", "1", "--out", str(tmp_path / "o"), "--models-dir", str(models_dir), "--state-dir", str(state), "--force"] + FAST)
    assert rc == 0 and (tmp_path / "o" / "ratings.json").is_file()


def test_runs_when_queue_is_empty_or_missing(models_dir, tmp_path):
    for i, state in enumerate((tmp_path / "none", tmp_path / "done")):
        if i:
            make_queue_db(state, "done")
        assert trun.queue_status(state) == (False, 0)
    assert trun.main(["--rounds", "1", "--out", str(tmp_path / "o"), "--models-dir", str(models_dir), "--state-dir", str(tmp_path / "done")] + FAST) == 0


def test_sanity_gate_excludes_broken_models_unless_overridden(models_dir, tmp_path, capsys):
    make_fake_model(models_dir / "broken", seed=9, nickname="x", model_name="NaN Net", nan=True)
    args = ["--rounds", "1", "--models-dir", str(models_dir), "--state-dir", str(tmp_path / "s")] + FAST
    assert trun.main(args + ["--out", str(tmp_path / "a")]) == 0
    out = capsys.readouterr().out
    assert "SANITY GATE FAILED: NaN_Net" in out and "excluded" in out
    ids = [r["id"] for r in json.loads((tmp_path / "a" / "ratings.json").read_text())["ratings"]]
    assert "broken" not in ids
    assert json.loads((tmp_path / "a" / "sanity.json").read_text())["broken"]["passed"] is False
    assert trun.main(args + ["--out", str(tmp_path / "b"), "--override-sanity"]) == 0
    assert "broken" in [r["id"] for r in json.loads((tmp_path / "b" / "ratings.json").read_text())["ratings"]]


def test_too_many_rounds_and_too_few_models(models_dir, tmp_path, capsys):
    args = ["--models-dir", str(models_dir), "--state-dir", str(tmp_path / "s"), "--out", str(tmp_path / "o")] + FAST
    assert trun.main(["--rounds", "5"] + args) == 2 and "repeat pairings" in capsys.readouterr().out  # 4 models: at most 3 rounds
    one = tmp_path / "one"; make_fake_model(one / "only", 0)
    assert trun.main(["--rounds", "1", "--models-dir", str(one), "--state-dir", str(tmp_path / "s"), "--out", str(tmp_path / "o")] + FAST) == 2


# ---------------------------------------------------------------- the real thing, end to end
def read_outputs(out: Path):
    return {n: (out / n).read_bytes() for n in ("games.pgn", "ratings.json", "standings.csv", "results.json", "sanity.json")}


def test_full_tournament_repeat_runs_are_identical(models_dir, tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"run{k}"
        assert trun.main(["--rounds", "3", "--out", str(out), "--models-dir", str(models_dir), "--state-dir", str(tmp_path / "s"),
                          "--depth-override-plies", "2", "--max-plies", "40", "--no-bracket", "--override-sanity"]) == 0  # (an untrained net may lose the gate game)
        outs.append(read_outputs(out))
    assert outs[0] == outs[1]  # identical games, results and ratings, byte for byte
    rat = json.loads(outs[0]["ratings.json"])
    nets = [r for r in rat["ratings"] if r["id"].startswith("sub")]
    assert len(nets) == 4 and all(r["games"] == 6 for r in nets)  # exactly 2N games each
    assert rat["games_per_model"] == 6 and {r["id"] for r in rat["ratings"]} >= {"bot_random", "bot_material"}
    assert [r["elo"] for r in rat["ratings"]] == sorted((r["elo"] for r in rat["ratings"]), reverse=True)
    assert next(r for r in rat["ratings"] if r["id"] == "bot_random")["elo"] == 0.0  # the random mover anchors the scale
    pgn = outs[0]["games.pgn"].decode()
    assert pgn.count("[Event ") == sum(r["games"] for r in rat["ratings"]) // 2
    csv_lines = outs[0]["standings.csv"].decode().splitlines()
    assert csv_lines[0].startswith("rank,id,name") and len(csv_lines) == 1 + len(rat["ratings"])


def test_bracket_is_precomputed_with_replayable_moves(models_dir, tmp_path):
    out = tmp_path / "o"
    assert trun.main(["--rounds", "2", "--out", str(out), "--models-dir", str(models_dir), "--state-dir", str(tmp_path / "s"),
                      "--depth-override-plies", "1", "--max-plies", "30", "--override-sanity"]) == 0
    br = json.loads((out / "bracket.json").read_text())
    assert [r["name"] for r in br["rounds"]] == ["semi-finals", "final"]
    assert len(br["rounds"][0]["matches"]) == 2 and len(br["rounds"][1]["matches"]) == 1
    assert br["champion"] in br["seeds"] and br["rounds"][1]["matches"][0]["winner"] == br["champion"]
    seeds = br["seeds"]
    m = br["rounds"][0]["matches"]
    assert {(x["seed_a"], x["seed_b"]) for x in m} == {(1, 4), (2, 3)}  # 1v4, 2v3 in a 4-player bracket
    for match in br["rounds"][0]["matches"] + br["rounds"][1]["matches"]:
        assert len(match["games"]) in (2, 3)
        for g in match["games"]:
            b = chess.Board()
            for u in g["moves_uci"]:
                assert chess.Move.from_uci(u) in b.legal_moves
                b.push_uci(u)
            assert len(g["moves_san"]) == len(g["moves_uci"])
    assert (out / "run_config.json").is_file()


def test_the_api_server_never_imports_the_tournament():
    import subprocess, sys
    code = "import sys; import app.main; print(any(m == 'tournament' or m.startswith('tournament.') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2] / "backend", capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
