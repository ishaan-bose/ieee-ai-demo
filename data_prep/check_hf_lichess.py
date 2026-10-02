import os, collections
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "Lichess/chess-position-evaluations"
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".parquet"))
print("parquet files:", len(files), "| first:", files[0], "| last:", files[-1])

path = hf_hub_download(REPO, files[0], repo_type="dataset",
                       local_dir=os.path.expanduser("~/demo/data/lichess_hf"))
print("downloaded:", path, "| %.2f GB" % (os.path.getsize(path) / 1e9))
pf = pq.ParquetFile(path)
print("rows in shard:", pf.metadata.num_rows, "| row groups:", pf.metadata.num_row_groups)
print("schema:", pf.schema_arrow)

VAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9}
fens, cps, mates, depths = [], [], [], []
for batch in pf.iter_batches(batch_size=250_000, columns=["fen", "depth", "cp", "mate"]):
    d = batch.to_pydict()
    fens += d["fen"]; cps += d["cp"]; mates += d["mate"]; depths += d["depth"]
    if len(fens) >= 1_000_000:
        break
n = len(fens)
print("sampled rows:", n)
for i in range(3):
    print("row", i, "|", fens[i], "| cp", cps[i], "| mate", mates[i], "| depth", depths[i])

runs = 1 + sum(1 for i in range(1, n) if fens[i] != fens[i - 1])
distinct = len(set(fens))
print("distinct FENs:", distinct, "| runs of consecutive equal FENs:", runs)
print("(runs close to distinct = rows for a position sit next to each other)")
per = collections.Counter(collections.Counter(fens).values())
print("rows-per-FEN histogram (top 6):", per.most_common(6))
print("cp null share: %.3f | mate null share: %.3f" % (
    sum(c is None for c in cps) / n, sum(m is None for m in mates) / n))

# sign convention check, same as before
rows = []
for f, c in zip(fens[:500_000], cps[:500_000]):
    if c is None:
        continue
    parts = f.split()
    mat = 0
    for ch in parts[0]:
        if ch.isalpha():
            v = VAL.get(ch.lower(), 0)
            mat += v if ch.isupper() else -v
    rows.append((max(-1000, min(1000, c)), mat, parts[1] == "w"))
a = np.array(rows, dtype=np.float64)
for name, m in (("white to move", a[:, 2] == 1), ("black to move", a[:, 2] == 0)):
    print("corr(cp, material), %s: %+.3f" % (name, np.corrcoef(a[m, 0], a[m, 1])[0, 1]))
print("(both positive = White's view, same as the jsonl dump)")
