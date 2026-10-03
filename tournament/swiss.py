"""Swiss rounds with GUARANTEED games (SPEC 10): every round each model plays its opponent TWICE from the same opening with colours
swapped, so after N rounds everyone has exactly 2N games. No repeat pairings. With an odd number of models the unpaired one plays a
reference bot (also twice). Pairing is a deterministic function of the results so far (no randomness)."""

from __future__ import annotations

import zlib


class PairingError(Exception):
    pass


def _tie(seed: int, pid: str) -> int:
    return zlib.crc32(f"{seed}:{pid}".encode())


def pair_round(players: list[str], scores: dict[str, float], played: set[frozenset], byes: set[str], seed: int = 0) -> tuple[list[tuple[str, str]], str | None]:
    """-> (pairs, bye). Order: by score (high first), ties by a seeded hash, then id. The bye goes to the lowest-ranked player without one yet.
    Pairs are found by backtracking so that nobody meets twice; raises PairingError when that is impossible (too many rounds)."""
    order = sorted(players, key=lambda p: (-scores.get(p, 0.0), _tie(seed, p), p))
    bye = None
    if len(order) % 2:
        for p in reversed(order):
            if p not in byes:
                bye = p
                break
        if bye is None:
            bye = order[-1]
        order = [p for p in order if p != bye]

    def solve(rest: list[str]) -> list[tuple[str, str]] | None:
        if not rest:
            return []
        a = rest[0]
        for i in range(1, len(rest)):
            b = rest[i]
            if frozenset((a, b)) in played:
                continue
            tail = solve(rest[1:i] + rest[i + 1:])
            if tail is not None:
                return [(a, b)] + tail
        return None

    pairs = solve(order)
    if pairs is None:
        raise PairingError("no pairing without repeats exists (more rounds than the number of models allows)")
    return pairs, bye


def max_rounds(n_models: int) -> int:
    """Theoretical most rounds without a repeat pairing: n-1 for an even count, n for an odd count (everybody rests exactly once)."""
    return 1 if n_models <= 2 else n_models - 1 if n_models % 2 == 0 else n_models


def use_round_robin(n_models: int, rounds: int) -> bool:
    """Swiss pairing by score is safe (a pairing without repeats always exists) while rounds <= n/2 - 1: after r played rounds every
    model still has at least n/2 unplayed opponents, and a graph with minimum degree >= n/2 has a perfect matching (Dirac).
    Beyond that the fixed circle-method round-robin is used instead, which can never get stuck."""
    return rounds > max(1, n_models // 2 - 1)


def round_robin_round(players: list[str], r: int) -> tuple[list[tuple[str, str]], str | None]:
    """Circle method, round r (1-based). Players are sorted by id, so the schedule is a pure function of the entrants."""
    L: list[str | None] = sorted(players)
    if len(L) % 2:
        L.append(None)  # the dummy: whoever is paired with it rests this round (a "bye": plays a reference bot instead)
    m = len(L)
    rot = L[1:]
    k = (r - 1) % (m - 1)
    rot = rot[-k:] + rot[:-k] if k else rot
    pos = [L[0]] + rot
    pairs, bye = [], None
    for i in range(m // 2):
        a, b = pos[i], pos[m - 1 - i]
        if a is None:
            bye = b
        elif b is None:
            bye = a
        else:
            pairs.append((a, b) if a < b else (b, a))
    return pairs, bye
