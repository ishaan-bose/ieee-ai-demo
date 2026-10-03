"""python-chess <-> array format (SPEC 4.3). Used by the tournament and tests; python-chess is imported lazily."""

from __future__ import annotations

import numpy as np

_CODE = {c: i + 1 for i, c in enumerate("PNBRQKpnbrqk")}
_CHAR = {v: k for k, v in _CODE.items()}


def board_to_arrays(board) -> tuple[np.ndarray, int, int, int]:
    """chess.Board -> (boards uint8 (64,), stm, castle bits, ep file or -1). Index = row*8+col, row 0 = rank 8."""
    arr = np.zeros(64, np.uint8)
    for sq, piece in board.piece_map().items():
        arr[(7 - sq // 8) * 8 + sq % 8] = _CODE[piece.symbol()]
    castle = (1 if board.has_kingside_castling_rights(True) else 0) | (2 if board.has_queenside_castling_rights(True) else 0) \
        | (4 if board.has_kingside_castling_rights(False) else 0) | (8 if board.has_queenside_castling_rights(False) else 0)
    ep = -1
    if board.ep_square is not None:
        ep = board.ep_square % 8
    return arr, 1 if board.turn else 0, castle, ep


def arrays_to_fen(boards: np.ndarray, stm: int, castle: int, ep: int) -> str:
    rows = []
    for r in range(8):
        row, empty = "", 0
        for c in range(8):
            v = int(boards[r * 8 + c])
            if v == 0:
                empty += 1
            else:
                row += (str(empty) if empty else "") + _CHAR[v]
                empty = 0
        rows.append(row + (str(empty) if empty else ""))
    cs = "".join(ch for bit, ch in ((1, "K"), (2, "Q"), (4, "k"), (8, "q")) if castle & bit) or "-"
    ep_sq = "-" if ep < 0 else f"{chr(97 + ep)}{6 if stm == 1 else 3}"
    return f"{'/'.join(rows)} {'w' if stm else 'b'} {cs} {ep_sq} 0 1"


def boards_to_batch(boards_list) -> dict[str, np.ndarray]:
    """List of chess.Board -> numpy arrays ready for torch.from_numpy."""
    n = len(boards_list)
    b = np.zeros((n, 64), np.uint8)
    stm = np.zeros(n, np.uint8)
    castle = np.zeros(n, np.uint8)
    ep = np.full(n, -1, np.int8)
    for i, bd in enumerate(boards_list):
        b[i], stm[i], castle[i], ep[i] = board_to_arrays(bd)
    return {"boards": b, "stm": stm, "castle": castle, "ep": ep}
