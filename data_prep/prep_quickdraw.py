import json, hashlib, os, random

D = os.path.expanduser("~/demo/data/quickdraw")
OUT = os.path.join(D, "processed")
os.makedirs(OUT, exist_ok=True)
CLASSES = ["cat","bicycle","house","pizza","lightning","star","fish","tree","umbrella","sword"]
TRAIN_CAP = 110_000

def bucket(k):
    return int(hashlib.md5(k.encode()).hexdigest(), 16) % 1000

files = {n: open(f"{OUT}/{n}.jsonl", "w") for n in ("train", "val", "pool")}
counts = {n: [0]*10 for n in files}
for ci, c in enumerate(CLASSES):
    with open(f"{D}/{c}.ndjson") as f:
        for line in f:
            r = json.loads(line)
            if not r["recognized"]:
                continue
            b = bucket(r["key_id"])
            split = "val" if b < 20 else "pool" if b < 22 else "train"
            if split == "train" and counts["train"][ci] >= TRAIN_CAP:
                continue
            rec = {"c": ci, "k": r["key_id"], "d": r["drawing"]}
            files[split].write(json.dumps(rec, separators=(",", ":")) + "\n")
            counts[split][ci] += 1
for f in files.values():
    f.close()

pool = [json.loads(l) for l in open(f"{OUT}/pool.jsonl")]
rng = random.Random(0)
by_class = {i: [r for r in pool if r["c"] == i] for i in range(10)}
duel = []
for i in range(10):
    duel += rng.sample(by_class[i], 20)
rng.shuffle(duel)
used = {r["k"] for r in duel}
rest = [r for r in pool if r["k"] not in used]
probe = [rng.choice([r for r in rest if r["c"] == i]) for i in range(10)]
probe += rng.sample([r for r in rest if r not in probe], 6)
rng.shuffle(probe)

samples = []
seen = [0]*10
for l in open(f"{OUT}/val.jsonl"):
    r = json.loads(l)
    if seen[r["c"]] < 5:
        samples.append(r)
        seen[r["c"]] += 1
    if sum(seen) == 50:
        break

json.dump(CLASSES, open(f"{OUT}/classes.json", "w"))
json.dump(duel, open(f"{OUT}/duel.json", "w"), separators=(",", ":"))
json.dump(probe, open(f"{OUT}/probe.json", "w"), separators=(",", ":"))
json.dump(samples, open(f"{OUT}/samples.json", "w"), separators=(",", ":"))
for n in counts:
    print(n, sum(counts[n]), counts[n])
print("duel", len(duel), "probe", len(probe), "samples", len(samples))
