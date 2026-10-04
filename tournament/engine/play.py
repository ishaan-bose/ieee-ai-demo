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


EVENT = "IEEE CS @ PESU - Build Your Own AI tournament"
SITE = "IEEE CS @ PESU"
# PGN Termination vocabulary: internal reason -> tag value. Nothing here (or anywhere in the PGN) records time: the games are deterministic and so is the file.
TERMINATIONS = {"checkmate": "checkmate", "stalemate": "stalemate", "insufficient material": "insufficient material", "threefold repetition": "repetition",
                "fifty-move rule": "50-move", "adjudicated by material": "adjudicated"}


def termination_tag(reason: str) -> str:
    return "forfeit" if reason.startswith("forfeit") else TERMINATIONS.get(reason, reason)


def full_moves_tag(full_moves: float | None) -> str:
    """Search depth in FULL moves (2 plies each) as a tag value; a player that does not search (the random mover) is 0."""
    if not full_moves:
        return "0"
    return str(int(full_moves)) if float(full_moves).is_integer() else f"{full_moves:g}"


class SearchPlayer:
    def __init__(self, name: str, searcher: Searcher, pgn_name: str | None = None):
        self.name, self.searcher = name, searcher
        self.pgn_name = pgn_name or name  # "Bot Name (submitter)" in the PGN tags
        self.moves_searched = 0
        self.leaves = 0
        self.seconds = 0.0

    @property
    def depth_full_moves(self) -> float:
        return self.searcher.depth / 2

    def choose(self, board: chess.Board) -> chess.Move:
        move, _score, st = self.searcher.best_move(board)  # may raise EvalError
        self.moves_searched += 1
        self.leaves += st["leaves"]
        self.seconds += st["seconds"]
        return move


class RandomPlayer:
    """The random mover, made deterministic: the move is chosen by a CRC of the position (so repeat runs play identical games)."""

    depth_full_moves = 0

    def __init__(self, name: str = "random", pgn_name: str | None = None):
        self.name = name
        self.pgn_name = pgn_name or "Random Bot (reference)"

    def choose(self, board: chess.Board) -> chess.Move:
        moves = sorted(board.legal_moves, key=lambda m: m.uci())
        return moves[zlib.crc32(board.fen().encode()) % len(moves)]


class StockfishPlayer:
    """Optional shallow Stockfish anchor (needs a stockfish binary: --stockfish PATH). Fixed depth, 1 thread, hash 16 MB: deterministic."""

    def __init__(self, path: str, depth: int = 2, name: str | None = None):
        import chess.engine

        self.name = name or f"stockfish-d{depth}"
        self.pgn_name = f"Stockfish depth {depth} (reference)"
        self.depth_full_moves = depth / 2
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
    start_fen: str = chess.STARTING_FEN  # the position after the opening moves: the PGN starts here (SetUp/FEN tags)
    white_pgn: str = ""  # "Bot Name (submitter)" for the PGN tags (falls back to the internal name)
    black_pgn: str = ""
    white_depth: float | None = None  # search depth in full moves
    black_depth: float | None = None
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

    def pgn(self, event: str = EVENT, round_: str = "?") -> str:
        """One game as PGN. Tags: Event, Site, Date (unknown: games carry no time), Round, White, Black, Result, SetUp + FEN (the opening position),
        Opening, Termination and the search depth of each side in full moves. No time tags, no clocks, no comments: repeat runs are byte-identical."""
        g = chess.pgn.Game()
        h = g.headers
        h["Event"], h["Site"], h["Date"], h["Round"] = event, SITE, "????.??.??", round_
        h["White"], h["Black"], h["Result"] = self.white_pgn or self.white, self.black_pgn or self.black, self.result
        h["SetUp"], h["FEN"] = "1", self.start_fen
        if self.opening:
            h["Opening"] = self.opening
        h["Termination"] = termination_tag(self.reason)
        h["WhiteDepth"], h["BlackDepth"] = full_moves_tag(self.white_depth), full_moves_tag(self.black_depth)
        node = g
        for u in self.moves[self.opening_plies:]:
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

    start_fen = board.fen()

    def done(result: str, reason: str, forfeit: str | None = None) -> GameResult:
        return GameResult(white.name, black.name, result, reason, uci, name, len(plies), forfeit, start_fen,
                          getattr(white, "pgn_name", white.name), getattr(black, "pgn_name", black.name),
                          getattr(white, "depth_full_moves", None), getattr(black, "depth_full_moves", None))

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
