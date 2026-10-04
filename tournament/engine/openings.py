"""A fixed suite of openings (UCI move lists) that gives the games variety (SPEC 10). Each pairing plays BOTH colours from the same opening."""

OPENINGS: list[tuple[str, list[str]]] = [
    ("Ruy Lopez", "e2e4 e7e5 g1f3 b8c6 f1b5".split()),
    ("Italian Game", "e2e4 e7e5 g1f3 b8c6 f1c4".split()),
    ("Sicilian Defence", "e2e4 c7c5 g1f3 d7d6 d2d4".split()),
    ("French Defence", "e2e4 e7e6 d2d4 d7d5".split()),
    ("Queen's Gambit", "d2d4 d7d5 c2c4 e7e6".split()),
    ("King's Indian", "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7".split()),
    ("English Opening", "c2c4 e7e5 b1c3 g8f6".split()),
    ("Caro-Kann", "e2e4 c7c6 d2d4 d7d5".split()),
]
