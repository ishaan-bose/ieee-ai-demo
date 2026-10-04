"""Players and games. Deterministic: no randomness anywhere (the 'random mover' picks by a hash of the position)."""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import chess
import chess.pgn

from tournament.engine.evaluators import EvalError
from tournament.engine.search import Searcher

MAX_PLIES = 150
ADJUDICATE_MATERIAL = 2  # a side that is ahead by at least this many pawns at the ply cap wins
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class SearchPlayer:
    def __init__(self, name: str, searcher: Searcher):
        self.name, self.searcher = name, searcher
        self.moves_searched = 0
        self.leaves = 0
        self.seconds = 0.0

    def choose(self, board: chess.Board) -> chess.Move:
        move, _score, st = self.searcher.best_move(board)  # may raise EvalError
        self.moves_searched += 1
        self.leaves += st["leaves"]
        self.seconds += st["seconds"]
        return move


class RandomPlayer:
    """The random mover, made deterministic: the move is chosen by a CRC of the position (so repeat runs play identical games)."""

    def __init__(self, name: str = "random"):
        self.name = name

    def choose(self, board: chess.Board) -> chess.Move:
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        return moves[zlib.crc32(board.fen().encode()) % len(moves)]


class StockfishPlayer:
    """Optional shallow Stockfish anchor (needs a stockfish binary: --stockfish PATH). Fixed depth, 1 thread, hash 16 MB: deterministic."""

    def __init__(self, path: str, depth: int = 2, name: str | None = None):
        import chess.engine

        self.name = name or f"stockfish-d{depth}"
        self.depth = depth
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine.configure({"Threads": 1, "Hash": 16})

    def choose(self, board: chess.Board) -> chess.Move:
        import chess.engine

        return self.engine.play(board, chess.engine.Limit(depth=self.depth)).move  # type: ignore[return-value]

    def close(self) -> None:
        self.engine.quit()


@dataclass
class GameResult:
    white: str
    black: str
    result: str  # "1-0" | "0-1" | "1/2-1/2"
    reason: str
    moves: list[str]  # UCI, including the opening moves
    opening: str = ""
    opening_plies: int = 0
    forfeit: str | None = None  # name of the side that forfeited
    extra: dict = field(default_factory=dict)

    def score_for(self, name: str) -> float:
        if name not in (self.white, self.black):
            raise KeyError(name)
        if self.result == "1/2-1/2":
            return 0.5
        return 1.0 if (self.result == "1-0") == (name == self.white) else 0.0

    def san_moves(self) -> list[str]:
        b, out = chess.Board(), []
        for u in self.moves:
            m = chess.Move.from_uci(u)
            out.append(b.san(m))
            b.push(m)
        return out

    def pgn(self, event: str = "AI tournament", round_: str = "?") -> str:
        g = chess.pgn.Game()
        g.headers.update({"Event": event, "Site": "?", "Date": "????.??.??", "Round": round_, "White": self.white, "Black": self.black,
                          "Result": self.result, "Termination": self.reason, "Opening": self.opening})
        node = g
        for u in self.moves:
            node = node.add_variation(chess.Move.from_uci(u))
        return str(g) + "\n\n"


def material_diff(board: chess.Board) -> int:
    """White minus black material in pawns, computed from the pieces on the board."""
    return sum(v * (len(board.pieces(pt, True)) - len(board.pieces(pt, False))) for pt, v in _VAL.items())


def play_game(white, black, opening: tuple[str, list[str]] | None = None, max_plies: int = MAX_PLIES) -> GameResult:
    """White and Black are players with .name and .choose(board). Opening moves are played without searching."""
    board = chess.Board()
    uci: list[str] = []
    name, plies = ("", [])
    if opening:
        name, plies = opening
        for u in plies:
            m = chess.Move.from_uci(u)
            if m not in board.legal_moves:
                raise ValueError(f"illegal opening move {u} in {name}")
            board.push(m); uci.append(u)

    def done(result: str, reason: str, forfeit: str | None = None) -> GameResult:
        return GameResult(white.name, black.name, result, reason, uci, name, len(plies), forfeit)

    while True:
        if board.is_checkmate():
            return done("0-1" if board.turn else "1-0", "checkmate")
        if board.is_stalemate():
            return done("1/2-1/2", "stalemate")
        if board.is_insufficient_material():
            return done("1/2-1/2", "insufficient material")
        if board.can_claim_threefold_repetition():
            return done("1/2-1/2", "threefold repetition")
        if board.can_claim_fifty_moves():
            return done("1/2-1/2", "fifty-move rule")
        if len(board.move_stack) >= max_plies:
            d = material_diff(board)
            return done("1-0" if d >= ADJUDICATE_MATERIAL else "0-1" if d <= -ADJUDICATE_MATERIAL else "1/2-1/2", "adjudicated by material")
        player = white if board.turn else black
        try:
            move = player.choose(board)
        except EvalError as e:  # NaN/inf (or a crash) forfeits the game
            return done("0-1" if board.turn else "1-0", f"forfeit: {e}", forfeit=player.name)
        if move not in board.legal_moves:
            return done("0-1" if board.turn else "1-0", f"forfeit: illegal move {move}", forfeit=player.name)
        board.push(move); uci.append(move.uci())
