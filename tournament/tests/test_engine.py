import chess
import chess.pgn
import io
import numpy as np
import pytest

from app.chess_net.pychess import arrays_to_fen
from tournament.engine.evaluators import EvalError, MaterialEvaluator
from tournament.engine.openings import OPENINGS
from tournament.engine.play import RandomPlayer, SearchPlayer, material_diff, play_game
from tournament.engine.search import MATE, Searcher, ordered_moves

SAMPLES = "data/samples/lichess"


def sample_fens(n=25):
    root = __import__("pathlib").Path(__file__).resolve().parents[2] / SAMPLES
    b, s, c, e = (np.load(root / f"{k}.npy") for k in ("boards", "stm", "castle", "ep"))
    return [arrays_to_fen(b[i], int(s[i]), int(c[i]), int(e[i])) for i in range(0, 2000, 2000 // n)][:n]


def brute(board, depth, ev, ply=0):
    """Plain negamax with the same terminal handling and leaf evaluation as the searcher, no pruning, no batching."""
    if board.is_checkmate():
        return -(MATE - ply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if depth == 0:
        from app.chess_net.pychess import board_to_arrays
        b, st, ca, ep = board_to_arrays(board)
        return float(ev.evaluate(b[None], np.array([st]), np.array([ca]), np.array([ep]))[0])
    best = -MATE * 2
    for m in board.legal_moves:
        board.push(m); best = max(best, -brute(board, depth - 1, ev, ply + 1)); board.pop()
    return best


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_alpha_beta_equals_plain_minimax(depth):
    ev = MaterialEvaluator()
    for fen in sample_fens(12 if depth < 3 else 5):
        board = chess.Board(fen)
        if board.is_game_over() or board.legal_moves.count() < 2:
            continue
        _, score, st = Searcher(ev, depth).best_move(board)
        assert score == pytest.approx(brute(board, depth, ev)), fen
        assert st["depth"] == depth


def test_finds_mates():
    s = Searcher(MaterialEvaluator(), 3)
    m1 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")  # Ra8#
    mv, sc, _ = s.best_move(m1)
    assert mv.uci() == "a1a8" and sc > MATE - 10
    m2 = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")  # Qxf7#
    assert Searcher(MaterialEvaluator(), 2).best_move(m2)[0].uci() == "h5f7"
    # mate in 2: 1.Qg6+ hxg6 2.Bxg6# style pattern is covered by depth; here a simple queen+king mate in 2
    m3 = chess.Board("7k/8/6K1/8/8/8/8/Q7 w - - 0 1")
    mv, sc, _ = Searcher(MaterialEvaluator(), 4).best_move(m3)
    b = m3.copy(); b.push(mv)
    assert sc > MATE - 10


def test_does_not_hang_its_queen():
    b = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    mv = Searcher(MaterialEvaluator(), 2).best_move(b)[0]
    b.push(mv)
    assert not any(b.is_capture(m) and b.piece_type_at(m.to_square) == chess.QUEEN for m in b.legal_moves)


def test_exact_ties_break_alphabetically_by_uci():
    for depth in (1, 2, 3):
        mv, score, _ = Searcher(MaterialEvaluator(), depth).best_move(chess.Board())
        assert score == 0 and mv.uci() == "a2a3", (depth, mv)


def test_move_ordering_is_fixed_and_captures_first():
    b = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    o = ordered_moves(b)
    assert o[0].uci() == "e4d5" and o == ordered_moves(b.copy())
    rest = [m.uci() for m in o[1:]]
    assert rest == sorted(rest)


def test_deterministic_and_leaf_batches():
    sizes = []

    class Spy(MaterialEvaluator):
        def evaluate(self, boards, stm, castle, ep):
            sizes.append(len(boards))
            return super().evaluate(boards, stm, castle, ep)

    b = chess.Board(sample_fens(1)[0])
    r1 = Searcher(Spy(), 2).best_move(b)
    r2 = Searcher(Spy(), 2).best_move(b)
    assert r1[0] == r2[0] and r1[1] == r2[1]
    assert max(sizes) > 10 and len(sizes) < sum(sizes) / 4  # leaves are evaluated in batches, not one by one


def test_node_cap_returns_last_completed_depth_and_never_crashes():
    b = chess.Board(sample_fens(3)[1])
    full = Searcher(MaterialEvaluator(), 3).best_move(b)[2]
    capped = Searcher(MaterialEvaluator(), 3, node_cap=full["leaves"] // 3).best_move(b)
    assert 1 <= capped[2]["depth"] < 3 and capped[0] in b.legal_moves
    tiny = Searcher(MaterialEvaluator(), 3, node_cap=1).best_move(b)
    assert tiny[0] in b.legal_moves and tiny[2]["depth"] == 0


def test_nan_evaluation_forfeits_the_game():
    class Bad:
        def evaluate(self, boards, stm, castle, ep):
            return np.full(len(boards), np.nan)

    bad = SearchPlayer("bad", Searcher(Bad(), 2))
    with pytest.raises(EvalError):
        bad.choose(chess.Board())
    g = play_game(bad, RandomPlayer(), OPENINGS[0], max_plies=40)
    assert g.forfeit == "bad" and g.result == "0-1" and g.score_for("random") == 1.0
    g2 = play_game(RandomPlayer(), bad, OPENINGS[0], max_plies=40)
    assert g2.result == "1-0" if len(OPENINGS[0][1]) % 2 == 1 else True


def test_openings_are_legal():
    assert len(OPENINGS) >= 6
    for name, moves in OPENINGS:
        b = chess.Board()
        for u in moves:
            m = chess.Move.from_uci(u)
            assert m in b.legal_moves, (name, u)
            b.push(m)


def test_games_pgn_roundtrip_and_determinism():
    mat = SearchPlayer("mat", Searcher(MaterialEvaluator(), 2))
    g1 = play_game(mat, RandomPlayer(), OPENINGS[0], max_plies=70)
    g2 = play_game(SearchPlayer("mat", Searcher(MaterialEvaluator(), 2)), RandomPlayer(), OPENINGS[0], max_plies=70)
    assert g1.moves == g2.moves and g1.pgn() == g2.pgn()  # no randomness anywhere
    pg = chess.pgn.read_game(io.StringIO(g1.pgn()))
    assert [m.uci() for m in pg.mainline_moves()] == g1.moves and pg.headers["Result"] == g1.result
    assert g1.moves[: len(OPENINGS[0][1])] == OPENINGS[0][1] and g1.san_moves()[0] == "e4"
    assert g1.score_for("mat") in (0.0, 0.5, 1.0) and g1.score_for("mat") + g1.score_for("random") == 1.0


def test_material_bot_beats_random_and_adjudication():
    wins = 0
    for i, (white, black) in enumerate([("mat", "rnd"), ("rnd", "mat")] * 2):
        mat, rnd = SearchPlayer("mat", Searcher(MaterialEvaluator(), 2)), RandomPlayer("rnd")
        g = play_game(mat if white == "mat" else rnd, rnd if white == "mat" else mat, OPENINGS[i % 4], max_plies=100)
        wins += g.score_for("mat")
    assert wins >= 3
    # a game cut at the ply cap is adjudicated by material
    mat = SearchPlayer("mat", Searcher(MaterialEvaluator(), 2))
    g = play_game(mat, RandomPlayer("rnd"), OPENINGS[0], max_plies=len(OPENINGS[0][1]) + 12)
    if g.reason == "adjudicated by material":
        b = chess.Board()
        for u in g.moves:
            b.push_uci(u)
        d = material_diff(b)
        assert g.result == ("1-0" if d >= 2 else "0-1" if d <= -2 else "1/2-1/2")
