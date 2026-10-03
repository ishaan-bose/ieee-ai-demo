"""Leaf evaluators. Interface: evaluate(boards uint8 (N,64), stm (N,), castle (N,), ep (N,)) -> float64 (N,) scores in
centipawn-like units FOR THE SIDE TO MOVE. Non-finite values are reported by the searcher as an EvalError (the game is forfeited)."""

from __future__ import annotations

import numpy as np

# P N B R Q K p n b r q k  ->  centipawns (kings 0: they are always on the board)
_CP = np.array([0, 100, 300, 300, 500, 900, 0, -100, -300, -300, -500, -900, 0], dtype=np.float64)


class EvalError(Exception):
    """A model produced NaN/inf (or crashed): it forfeits the game."""


class MaterialEvaluator:
    """Material only (the reference 'material bot'). Always computed from the boards array."""

    name = "material"

    def evaluate(self, boards, stm, castle, ep) -> np.ndarray:
        white_view = _CP[np.minimum(np.asarray(boards).astype(np.int64), 12)].sum(axis=1)
        return np.where(np.asarray(stm) == 1, white_view, -white_view)


class NetEvaluator:
    """A trained network loaded with app.chess_net.model_io.load_model; evaluates in fixed-size fp32 chunks."""

    def __init__(self, loaded, chunk: int = 64):
        self.model, self.chunk = loaded, chunk
        self.name = loaded.name

    def evaluate(self, boards, stm, castle, ep) -> np.ndarray:
        try:
            out = self.model.score(boards, stm, castle, ep, chunk=self.chunk).numpy().astype(np.float64)
        except Exception as e:  # noqa: BLE001  a broken model must not crash the tournament
            raise EvalError(f"{type(e).__name__}: {e}") from e
        return out
