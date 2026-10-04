"""Bradley-Terry ratings from all games (draws count half), expressed in Elo points. The reference bots play in the same pool, so they
anchor the scale: the random mover is pinned to 0 when present."""

from __future__ import annotations

import math


def bradley_terry(results: list[tuple[str, str, float]], players: list[str], prior: float = 0.5, iters: int = 5000, tol: float = 1e-12) -> dict[str, float]:
    """results: (a, b, score_of_a in {0, .5, 1}). A weak prior (a virtual average opponent, `prior` drawn games each) keeps players with
    only wins or only losses finite. MM algorithm (Hunter 2004). Returns Elo (400 * log10 strength), mean 1000 over `players`."""
    w = {p: 0.0 for p in players}
    games: dict[tuple[str, str], float] = {}
    for a, b, s in results:
        w[a] += s; w[b] += 1 - s
        k = (a, b) if a < b else (b, a)
        games[k] = games.get(k, 0.0) + 1.0
    nbrs: dict[str, list[tuple[str, float]]] = {p: [] for p in players}
    for (a, b), n in games.items():
        nbrs[a].append((b, n)); nbrs[b].append((a, n))
    for p in players:
        w[p] += 0.5 * prior
    s = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            denom = prior / (s[p] + 1.0) + sum(n / (s[p] + s[q]) for q, n in nbrs[p])
            new[p] = w[p] / denom if denom > 0 else s[p]
        gm = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / gm for p, v in new.items()}
        delta = max(abs(new[p] - s[p]) for p in players)
        s = new
        if delta < tol:
            break
    return {p: 1000.0 + 400.0 * math.log10(s[p]) for p in players}


def anchor_to(elo: dict[str, float], name: str, value: float = 0.0) -> dict[str, float]:
    shift = value - elo[name]
    return {p: v + shift for p, v in elo.items()}
