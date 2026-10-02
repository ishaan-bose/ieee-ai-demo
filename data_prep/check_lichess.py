import json, os, collections, random
import numpy as np

P = os.path.expanduser("~/demo/data/lichess_head.jsonl")
VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}

rows = []          # cp, material(white-black), white_to_move, piece_count
errors = 0
n_evals = collections.Counter()
n_fields = collections.Counter()
depths = []
mates = 0
fens = []
total = 0

for line in open(P):
    try:
        r = json.loads(line)
        fen = r["fen"]
        ev = max(r["evals"], key=lambda e: e["depth"])
        pv = ev["pvs"][0]
    except Exception:
        errors += 1
        continue
    total += 1
    parts = fen.split()
    n_fields[len(parts)] += 1
    n_evals[len(r["evals"])] += 1
    depths.append(ev["depth"])
    fens.append(fen)
    if "cp" not in pv:
        mates += 1
        continue
    mat, npieces = 0, 0
    for ch in parts[0]:
        if ch.isalpha():
            npieces += 1
            v = VAL.get(ch.lower(), 0)
            mat += v if ch.isupper() else -v
    rows.append((pv["cp"], mat, 1 if parts[1] == "w" else 0, npieces))

print("lines parsed:", total, "| parse errors:", errors)
print("FEN field counts:", dict(n_fields))
print("evals per position (top 5):", n_evals.most_common(5))
print("deepest-eval depth percentiles 5/50/95:", np.percentile(depths, [5, 50, 95]))
print("share with mate as deepest eval: %.3f" % (mates / max(total, 1)))
print("duplicate FENs in sample:", len(fens) - len(set(fens)))

a = np.array(rows, dtype=np.float64)
print("cp abs percentiles 50/90/99:", np.percentile(np.abs(a[:, 0]), [50, 90, 99]))
print("share |cp| > 1000: %.3f" % (np.mean(np.abs(a[:, 0]) > 1000)))
print("white to move share: %.3f" % a[:, 2].mean())
print("piece count mean: %.1f | share with <= 10 pieces: %.3f" % (a[:, 3].mean(), np.mean(a[:, 3] <= 10)))

cp_clip = np.clip(a[:, 0], -1000, 1000)
for name, m in (("white to move", a[:, 2] == 1), ("black to move", a[:, 2] == 0)):
    c = np.corrcoef(cp_clip[m], a[m, 1])[0, 1]
    print("corr(cp, white-minus-black material), %s: %+.3f" % (name, c))
print("(both positive = cp is from White's view; white +, black - = side-to-move view)")

random.seed(0)
for f in random.sample(fens, 3):
    print("sample fen:", f)
