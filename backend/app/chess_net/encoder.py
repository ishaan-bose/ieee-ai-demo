"""Position encoder: the single place that turns (boards, stm, castle, ep) into network input.

Imported by training (app/training) AND by the tournament (via the model loader), so both convert a
board exactly the same way (SPEC 10). Everything is torch, runs on the GPU, and is device-agnostic.

Board format (SPEC 4.3): uint8 (B, 64), 0 empty, 1-6 white P N B R Q K, 7-12 black p n b r q k;
index = row*8 + col, row 0 = rank 8, col 0 = file a (index 0 = a8, 63 = h1).
stm: 1 = white to move. castle bits: 1 = white K-side, 2 = white Q-side, 4 = black K-side, 8 = black Q-side.
ep: file 0-7 or -1.

Material is always computed from the boards (the stored `mat` array is not trusted).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

N_PLANES = 12 * 64  # 768 one-hot piece planes

PIECE_MATERIAL = torch.tensor([0, 1, 3, 3, 5, 9, 0, -1, -3, -3, -5, -9, 0])
# swap colors: code c -> c+6 (white<->black), empty stays empty
COLOR_SWAP = torch.tensor([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=torch.long)


@dataclass(frozen=True)
class Extras:
    """Optional input features (SPEC 8.3 `input_extras`). Side to move + castling are on by default."""

    stm_castle: bool = True  # 1 (stm) + 4 (castling) inputs
    en_passant: bool = False  # 8 inputs (one-hot file)
    material: bool = False  # 10 inputs: counts of P N B R Q for white then black, / 8, 2, 2, 2, 1
    attacks: bool = False  # 128 inputs: squares attacked by white, then by black

    @staticmethod
    def from_dict(d: dict | None) -> "Extras":
        d = d or {}
        return Extras(**{k: bool(d.get(k, f.default)) for k, f in Extras.__dataclass_fields__.items()})

    def to_dict(self) -> dict:
        return asdict(self)


def feature_dim(extras: Extras) -> int:
    return N_PLANES + (5 if extras.stm_castle else 0) + (8 if extras.en_passant else 0) \
        + (10 if extras.material else 0) + (128 if extras.attacks else 0)


# ---------------------------------------------------------------- attack maps

def _build_ray_tables():
    """For each direction d and step k: target square of the piece at s (or -1)."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]  # (drow, dcol); row 0 = rank 8
    rays = torch.full((8, 7, 64), -1, dtype=torch.long)
    for di, (dr, dc) in enumerate(dirs):
        for s in range(64):
            r, c = divmod(s, 8)
            for k in range(1, 8):
                rr, cc = r + dr * k, c + dc * k
                if 0 <= rr < 8 and 0 <= cc < 8:
                    rays[di, k - 1, s] = rr * 8 + cc
    knight = torch.full((8, 64), -1, dtype=torch.long)
    king = torch.full((8, 64), -1, dtype=torch.long)
    pawn = {1: torch.full((2, 64), -1, dtype=torch.long), 0: torch.full((2, 64), -1, dtype=torch.long)}
    for s in range(64):
        r, c = divmod(s, 8)
        for i, (dr, dc) in enumerate([(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]):
            if 0 <= r + dr < 8 and 0 <= c + dc < 8:
                knight[i, s] = (r + dr) * 8 + c + dc
        for i, (dr, dc) in enumerate(dirs):
            if 0 <= r + dr < 8 and 0 <= c + dc < 8:
                king[i, s] = (r + dr) * 8 + c + dc
        for i, dc in enumerate((-1, 1)):
            if 0 <= c + dc < 8:
                if r - 1 >= 0:
                    pawn[1][i, s] = (r - 1) * 8 + c + dc  # white pawns attack toward rank 8 (row decreasing)
                if r + 1 < 8:
                    pawn[0][i, s] = (r + 1) * 8 + c + dc
    return rays, knight, king, pawn


_RAYS, _KNIGHT, _KING, _PAWN = _build_ray_tables()
_ORTHO = (0, 1, 2, 3)
_DIAG = (4, 5, 6, 7)
_tables_cache: dict[str, tuple] = {}


def _tables(device):
    key = str(device)
    if key not in _tables_cache:
        _tables_cache[key] = (_RAYS.to(device), _KNIGHT.to(device), _KING.to(device),
                              {k: v.to(device) for k, v in _PAWN.items()})
    return _tables_cache[key]


def attack_maps(boards: torch.Tensor) -> torch.Tensor:
    """Squares attacked by each side, ignoring pins (pseudo-legal attacks; defended own pieces count).

    boards: uint8/long (B, 64). Returns float32 (B, 2, 64): [:,0] = attacked by white, [:,1] = by black.
    """
    b = boards.long()
    dev = b.device
    rays, knight, king, pawn = _tables(dev)
    occupied = b > 0
    out = torch.zeros(b.shape[0], 2, 64, dtype=torch.bool, device=dev)
    for color in (0, 1):  # 0 = white pieces (codes 1-6), 1 = black (7-12)
        base = 1 + 6 * color
        is_p, is_n, is_b, is_r, is_q, is_k = (b == base + i for i in range(6))
        att = out[:, color]

        def add(src_mask, targets):  # targets: (64,) square index of target for each source square, -1 = none
            valid = targets >= 0
            att[:, targets[valid]] |= src_mask[:, valid]

        for i in range(2):
            add(is_p, pawn[1 if color == 0 else 0][i])
        for i in range(8):
            add(is_n, knight[i])
            add(is_k, king[i])
        for di in range(8):
            slider = (is_r | is_q) if di in _ORTHO else (is_b | is_q)
            alive = slider.clone()
            for k in range(7):
                tgt = rays[di, k]
                valid = tgt >= 0
                att[:, tgt[valid]] |= alive[:, valid]
                # a ray stops AFTER the first occupied square it reaches; `alive` is indexed by ORIGIN square
                alive = alive & _ray_shift_free(occupied, tgt, valid)
    return out.float()


def _ray_shift_free(occupied: torch.Tensor, tgt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """For each origin square s: is the square it just attacked (tgt[s]) empty? (False when off-board.)"""
    free = torch.zeros_like(occupied)
    free[:, valid] = ~occupied[:, tgt[valid]]
    return free


# ---------------------------------------------------------------- encoding

def encode(boards, stm, castle, ep, extras: Extras = Extras()) -> torch.Tensor:
    """-> float32 (B, feature_dim(extras)). All inputs are tensors on the same device."""
    b = boards.long()
    B = b.shape[0]
    planes = F.one_hot(b, 13)[..., 1:]  # (B, 64, 12)
    parts = [planes.permute(0, 2, 1).reshape(B, N_PLANES).float()]
    if extras.stm_castle:
        c = castle.long()
        parts.append(stm.float().unsqueeze(1))
        parts.append(torch.stack([(c >> i) & 1 for i in range(4)], dim=1).float())
    if extras.en_passant:
        e = ep.long()
        oh = F.one_hot(e.clamp(min=0), 8).float() * (e >= 0).unsqueeze(1).float()
        parts.append(oh)
    if extras.material:
        counts = [(b == code).sum(dim=1).float() for code in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)]
        scale = torch.tensor([8, 2, 2, 2, 1, 8, 2, 2, 2, 1], dtype=torch.float32, device=b.device)
        parts.append(torch.stack(counts, dim=1) / scale)
    if extras.attacks:
        parts.append(attack_maps(b).reshape(B, 128))
    return torch.cat(parts, dim=1)


def material_from_boards(boards) -> torch.Tensor:
    """White minus black material (P=1, N=B=3, R=5, Q=9) computed from the boards, int64 (B,)."""
    table = PIECE_MATERIAL.to(boards.device)
    return table[boards.long()].sum(dim=1)


# ---------------------------------------------------------------- exact color-flip symmetry

def flip_position(boards, stm, castle, ep):
    """Mirror ranks, swap colors, flip side to move, swap castling rights. The eval negates (SPEC 8.3)."""
    swap = COLOR_SWAP.to(boards.device)
    nb = swap[boards.long()].reshape(-1, 8, 8).flip(1).reshape(-1, 64).to(boards.dtype)
    c = castle.long()
    nc = ((c & 3) << 2) | ((c >> 2) & 3)
    return nb, (1 - stm.long()).to(stm.dtype), nc.to(castle.dtype), ep


# ---------------------------------------------------------------- targets / outputs

EVAL_K = 400.0  # canonical scale used for the comparable validation number shown to participants


def effective_cp(cp, mate, mate_clip: float) -> torch.Tensor:
    """White-view centipawns with mate scores substituted by +-mate_clip (sign from `mate`)."""
    cpf = cp.float()
    mf = mate.float()
    return torch.where(mate != 0, torch.sign(mf) * mate_clip, cpf.clamp(-mate_clip, mate_clip))


def make_target(cp, mate, stm, *, target_type: str, K: float, mate_clip: float, perspective_flip: bool) -> torch.Tensor:
    """Training target. winprob: sigmoid(cp/K) in [0,1]; cp: clipped centipawns / K in [-mate_clip/K, +mate_clip/K].

    With perspective_flip the target is for the side to move (negate for Black to move).
    """
    v = effective_cp(cp, mate, mate_clip)
    if perspective_flip:
        v = torch.where(stm.long() == 1, v, -v)
    return torch.sigmoid(v / K) if target_type == "winprob" else v / K


def head_to_target_space(z, *, head: str, target_type: str, K: float, mate_clip: float) -> torch.Tensor:
    """Raw network output z -> prediction in target space (see make_target)."""
    if target_type == "winprob":
        if head == "tanh":
            return (torch.tanh(z) + 1) / 2
        return torch.sigmoid(z)  # linear and sigmoid heads: z is a logit
    s = mate_clip / K
    if head == "tanh":
        return s * torch.tanh(z)
    if head == "sigmoid":
        return s * (2 * torch.sigmoid(z) - 1)
    return z


def score_cp_for_stm(pred, stm, *, target_type: str, K: float, mate_clip: float, perspective_flip: bool) -> torch.Tensor:
    """Prediction (target space) -> centipawn-like score for the SIDE TO MOVE, the engine's leaf value."""
    if target_type == "winprob":
        p = pred.clamp(1e-6, 1 - 1e-6)
        cp = K * torch.log(p / (1 - p))
    else:
        cp = pred * K
    cp = cp.clamp(-mate_clip * 2, mate_clip * 2)
    if not perspective_flip:  # network predicts White's view -> convert to side to move
        cp = torch.where(stm.long() == 1, cp, -cp)
    return cp
