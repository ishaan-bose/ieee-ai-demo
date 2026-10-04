import chess
import numpy as np
import pytest
import torch

from app.chess_net import encoder as E
from app.chess_net.pychess import arrays_to_fen, board_to_arrays, boards_to_batch
from tests.conftest import SAMPLES_DIR


def sample_batch(n=200):
    d = {k: np.load(SAMPLES_DIR / "lichess" / f"{k}.npy") for k in ("boards", "stm", "castle", "ep", "cp", "mate", "mat", "npc")}
    return {k: torch.from_numpy(v[:n]) for k, v in d.items()}


def test_feature_dims():
    assert E.feature_dim(E.Extras(stm_castle=False)) == 768
    assert E.feature_dim(E.Extras()) == 773
    assert E.feature_dim(E.Extras(True, True, True, True)) == 773 + 8 + 10 + 128
    x = E.encode(*(sample_batch(8)[k] for k in ("boards", "stm", "castle", "ep")), E.Extras(True, True, True, True))
    assert x.shape == (8, E.feature_dim(E.Extras(True, True, True, True))) and x.dtype == torch.float32


def test_planes_and_orientation():
    s = sample_batch(50)
    x = E.encode(s["boards"], s["stm"], s["castle"], s["ep"], E.Extras(stm_castle=False))
    assert (x.sum(dim=1) == s["npc"].float()).all()  # one hot per piece
    pl = x.reshape(-1, 12, 64)
    for i in range(50):
        for sq in range(64):
            code = int(s["boards"][i, sq])
            assert (pl[i, :, sq].argmax().item() + 1 == code) if code else pl[i, :, sq].sum() == 0


def test_stm_castle_ep_features():
    s = sample_batch(100)
    x = E.encode(s["boards"], s["stm"], s["castle"], s["ep"], E.Extras(True, True, False, False))
    assert (x[:, 768] == s["stm"].float()).all()
    for bit in range(4):
        assert (x[:, 769 + bit] == ((s["castle"].long() >> bit) & 1).float()).all()
    ep = x[:, 773:781]
    assert (ep.sum(dim=1) == (s["ep"] >= 0).float()).all()
    assert (ep.argmax(dim=1)[s["ep"] >= 0] == s["ep"][s["ep"] >= 0].long()).all()


def test_material_computed_from_boards_matches_python_chess():
    s = sample_batch(100)
    m = E.material_from_boards(s["boards"])
    vals = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
    for i in range(100):
        bd = chess.Board(arrays_to_fen(*(s[k][i].numpy() for k in ("boards", "stm", "castle", "ep"))))
        ref = sum(v * (len(bd.pieces(pt, True)) - len(bd.pieces(pt, False))) for pt, v in vals.items())
        assert int(m[i]) == ref
    x = E.encode(s["boards"], s["stm"], s["castle"], s["ep"], E.Extras(False, False, True, False))
    assert x.shape[1] == 778 and x[:, 768:].max() <= 1.0


def test_fen_roundtrip_with_python_chess():
    s = sample_batch(200)
    for i in range(200):
        arrs = tuple(s[k][i].numpy() for k in ("boards", "stm", "castle", "ep"))
        bd = chess.Board(arrays_to_fen(*arrs))
        b2, stm2, c2, ep2 = board_to_arrays(bd)
        # python-chess drops castling rights whose king/rook is not on its home square (a few data rows)
        assert np.array_equal(b2, arrs[0]) and stm2 == int(arrs[1]) and (c2 & ~int(arrs[2])) == 0
        # ep only survives python-chess if a capture is legal; the arrays' ep can be unreachable
        assert ep2 in (-1, int(arrs[3]))


def test_attack_maps_match_python_chess():
    s = sample_batch(200)
    maps = E.attack_maps(s["boards"]).numpy()
    for i in range(200):
        bd = chess.Board(arrays_to_fen(*(s[k][i].numpy() for k in ("boards", "stm", "castle", "ep"))))
        for color, ch in ((0, chess.WHITE), (1, chess.BLACK)):
            for sq in range(64):
                idx = (7 - sq // 8) * 8 + sq % 8
                assert bool(maps[i, color, idx]) == bd.is_attacked_by(ch, sq), (i, color, sq)


def test_attack_maps_start_position():
    b = chess.Board()
    arr = boards_to_batch([b])
    m = E.attack_maps(torch.from_numpy(arr["boards"]))[0]
    assert m[0].sum() == 22 - 0 or m[0].sum() > 0  # sanity
    for color, ch in ((0, chess.WHITE), (1, chess.BLACK)):
        for sq in range(64):
            assert bool(m[color, (7 - sq // 8) * 8 + sq % 8]) == b.is_attacked_by(ch, sq)


def test_flip_is_an_exact_symmetry():
    s = sample_batch(200)
    fb, fs, fc, fe = E.flip_position(s["boards"], s["stm"], s["castle"], s["ep"])
    assert set(fb.unique().tolist()) <= set(range(13))
    for i in range(200):
        bd = chess.Board(arrays_to_fen(*(s[k][i].numpy() for k in ("boards", "stm", "castle", "ep"))))
        mirrored = bd.mirror()  # python-chess: flip vertically and swap colors, side to move flips
        b2, stm2, c2, _ = board_to_arrays(mirrored)
        assert np.array_equal(b2, fb[i].numpy()) and stm2 == int(fs[i])
        assert (c2 & ~int(fc[i])) == 0  # (see the roundtrip test about castling rights)
    # flipping twice returns the original
    bb, ss, cc, ee = E.flip_position(fb, fs, fc, fe)
    assert torch.equal(bb, s["boards"]) and torch.equal(ss, s["stm"]) and torch.equal(cc, s["castle"])
    # attack maps of the flipped position are the mirrored/swapped originals
    a = E.attack_maps(s["boards"]).reshape(-1, 2, 8, 8)
    af = E.attack_maps(fb).reshape(-1, 2, 8, 8)
    assert torch.equal(af, a.flip(1).flip(2))


@pytest.mark.parametrize("target_type", ["winprob", "cp"])
def test_targets_and_flip_negation(target_type):
    s = sample_batch(300)
    kw = dict(target_type=target_type, K=400.0, mate_clip=2000.0)
    y_w = E.make_target(s["cp"], s["mate"], s["stm"], perspective_flip=False, **kw)
    y_s = E.make_target(s["cp"], s["mate"], s["stm"], perspective_flip=True, **kw)
    white = s["stm"] == 1
    assert torch.allclose(y_w[white], y_s[white])
    if target_type == "cp":
        assert torch.allclose(y_w[~white], -y_s[~white])
        assert y_s.abs().max() <= 5.0 + 1e-6
    else:
        assert torch.allclose(y_w[~white], 1 - y_s[~white], atol=1e-6) and 0 <= y_s.min() and y_s.max() <= 1
    # mate rows: sign from `mate`, magnitude mate_clip
    m = s["mate"] != 0
    assert m.any()
    v = E.effective_cp(s["cp"], s["mate"], 2000.0)
    assert (v[m].abs() == 2000).all() and (torch.sign(v[m]) == torch.sign(s["mate"][m].float())).all()


def test_score_conversion_roundtrip():
    stm = torch.tensor([1, 0, 1, 0])
    cp_white = torch.tensor([150.0, 150.0, -300.0, -300.0])
    y = E.make_target(cp_white.short(), torch.zeros(4, dtype=torch.short), stm, target_type="winprob", K=400, mate_clip=2000, perspective_flip=True)
    score = E.score_cp_for_stm(y, stm, target_type="winprob", K=400, mate_clip=2000, perspective_flip=True)
    assert torch.allclose(score, torch.tensor([150.0, -150.0, -300.0, 300.0]), atol=0.5)
    # without perspective flip the network speaks White's view and the score is converted to stm
    yw = E.make_target(cp_white.short(), torch.zeros(4, dtype=torch.short), stm, target_type="cp", K=400, mate_clip=2000, perspective_flip=False)
    score = E.score_cp_for_stm(yw, stm, target_type="cp", K=400, mate_clip=2000, perspective_flip=False)
    assert torch.allclose(score, torch.tensor([150.0, -150.0, -300.0, 300.0]), atol=0.5)
