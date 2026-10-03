"""Deterministic alpha-beta (negamax) with fixed move ordering and BATCHED leaf evaluation (SPEC 10).

- Parents of leaves evaluate all their children in one batch (fixed-size fp32 chunks in the evaluator), so a position's score never
  depends on which positions share its batch.
- Move ordering is a pure function of the position (captures by MVV-LVA, then UCI order); exact ties at the root break alphabetically by UCI.
- No randomness anywhere. No capture extension (add only if the benchmark shows horizon blunders).
- Iterative deepening; an optional node cap aborts the current depth and returns the best move of the last completed one.
- A NaN/inf evaluation raises EvalError (the caller forfeits the game).
"""

from __future__ import annotations

import time

import chess
import numpy as np

from app.chess_net.pychess import board_to_arrays
from tournament.engine.evaluators import EvalError

MATE = 100_000.0
TIE_EPS = 1e-9
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10}


class NodeCapReached(Exception):
    pass


def ordered_moves(board: chess.Board) -> list[chess.Move]:
    """Fixed ordering: captures (most valuable victim, least valuable attacker) first, then everything else; ties by UCI."""
    def key(m: chess.Move):
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or chess.PAWN  # en passant: the victim is a pawn
            attacker = board.piece_type_at(m.from_square)
            return (0, -(10 * _VAL[victim] - _VAL[attacker]), m.uci())
        return (1, 0, m.uci())
    return sorted(board.legal_moves, key=key)


class Searcher:
    def __init__(self, evaluator, depth_plies: int, node_cap: int | None = None):
        if depth_plies < 1:
            raise ValueError("depth_plies must be >= 1")
        self.ev, self.depth, self.node_cap = evaluator, depth_plies, node_cap
        self.nodes = 0  # internal nodes visited
        self.leaves = 0  # positions sent to the evaluator
        self._cache: dict[bytes, float] = {}

    # ------------------------------------------------------------ evaluation of leaves
    def _terminal(self, board: chess.Board, ply: int) -> float | None:
        if board.is_checkmate():
            return -(MATE - ply)
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        return None

    def _eval_children(self, board: chess.Board, moves: list[chess.Move], ply: int) -> list[float]:
        """Score (from the PARENT's side to move) of every child position; ONE evaluator batch for all non-terminal ones."""
        scores: list[float | None] = [None] * len(moves)
        todo, rows = [], []
        for i, m in enumerate(moves):
            board.push(m)
            t = self._terminal(board, ply + 1)
            if t is not None:
                scores[i] = -t
            else:
                arrs = board_to_arrays(board)
                key = arrs[0].tobytes() + bytes([arrs[1], arrs[2], arrs[3] & 0xFF])
                hit = self._cache.get(key)
                if hit is not None:
                    scores[i] = -hit
                else:
                    todo.append((i, key)); rows.append(arrs)
            board.pop()
        if rows:
            self.leaves += len(rows)
            if self.node_cap is not None and self.leaves > self.node_cap:
                raise NodeCapReached
            vals = self.ev.evaluate(np.stack([r[0] for r in rows]), np.array([r[1] for r in rows], np.uint8),
                                    np.array([r[2] for r in rows], np.uint8), np.array([r[3] for r in rows], np.int8))
            if not np.all(np.isfinite(vals)):
                raise EvalError("the evaluator returned NaN or infinity")
            for (i, key), v in zip(todo, vals):
                v = float(v)
                self._cache[key] = v  # cache is keyed by the position, so it can never change a result
                scores[i] = -v
        return scores  # type: ignore[return-value]

    # ------------------------------------------------------------ search
    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        t = self._terminal(board, ply)
        if t is not None:
            return t
        moves = ordered_moves(board)
        if depth == 1:  # all children are leaves: evaluate them as one batch
            return max(self._eval_children(board, moves, ply))
        best = -MATE * 2
        for m in moves:
            board.push(m)
            v = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    def _root(self, board: chess.Board, depth: int, order: list[chess.Move]) -> tuple[chess.Move, float, dict[chess.Move, float]]:
        best_move, best = order[0], -MATE * 2
        scores: dict[chess.Move, float] = {}
        if depth == 1:
            vals = self._eval_children(board, order, 0)
            for m, v in zip(order, vals):
                scores[m] = v
                if v > best + TIE_EPS or (abs(v - best) <= TIE_EPS and m.uci() < best_move.uci()):
                    best, best_move = v, m
            return best_move, best, scores
        for m in order:
            board.push(m)
            # window just below the current best so that exact ties are searched exactly (alphabetical tie-break is then well defined)
            v = -self._negamax(board, depth - 1, -MATE * 2, -(best - TIE_EPS) if best > -MATE else MATE * 2, 1)
            board.pop()
            scores[m] = v
            if v > best + TIE_EPS or (abs(v - best) <= TIE_EPS and m.uci() < best_move.uci()):
                best, best_move = v, m
        return best_move, best, scores

    def best_move(self, board: chess.Board) -> tuple[chess.Move, float, dict]:
        """-> (move, score for the side to move, stats). Raises EvalError on a non-finite evaluation."""
        t0 = time.perf_counter()
        self.nodes = self.leaves = 0
        moves = ordered_moves(board)
        if not moves:
            raise ValueError("no legal moves")
        if len(moves) == 1:
            return moves[0], 0.0, {"nodes": 1, "leaves": 0, "depth": 0, "seconds": 0.0}
        move, score, reached = moves[0], 0.0, 0
        order = moves
        for d in range(1, self.depth + 1):
            try:
                mv, sc, scores = self._root(board, d, order)
            except NodeCapReached:
                break
            move, score, reached = mv, sc, d
            order = sorted(order, key=lambda m: (-scores[m], m.uci()))  # best first for the next, deeper iteration
        if reached == 0:  # node cap hit even at depth 1: fall back to the fixed ordering's first move
            move = moves[0]
        return move, score, {"nodes": self.nodes, "leaves": self.leaves, "depth": reached, "seconds": time.perf_counter() - t0}
