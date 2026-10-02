import os, json
import numpy as np

H = os.path.expanduser("~/demo/data")
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + ((" | " + str(detail)) if detail != "" else ""), flush=True)


def jload(p):
    with open(p) as f:
        return json.load(f)


# ---------------- Quick, Draw! ----------------
Q = f"{H}/quickdraw/processed"
for s, e in {"train": 1090830, "val": 25106, "pool": 2579}.items():
    try:
        n = bad = 0
        classes = set()
        with open(f"{Q}/{s}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                good = (0 <= r["c"] <= 9 and len(r["d"]) >= 1 and
                        all(len(st) == 2 and len(st[0]) == len(st[1]) for st in r["d"]))
                bad += (not good)
                classes.add(r["c"])
        check(f"quickdraw {s}.jsonl", n == e and bad == 0 and len(classes) == 10,
              f"lines={n} expected={e} bad_records={bad} classes={len(classes)}")
    except Exception as ex:
        check(f"quickdraw {s}.jsonl", False, repr(ex))
for fn, ln in (("classes.json", 10), ("duel.json", 200), ("probe.json", 16), ("samples.json", 50)):
    try:
        d = jload(f"{Q}/{fn}")
        ok = len(d) == ln
        if fn != "classes.json":
            ok = ok and all(("c" in r and "k" in r and "d" in r) for r in d)
        check(f"quickdraw {fn}", ok, f"entries={len(d)} expected={ln}")
    except Exception as ex:
        check(f"quickdraw {fn}", False, repr(ex))

# ---------------- Speech Commands ----------------
S = f"{H}/speech/processed"
spk_sets = {}
for s, e in {"train": 32479, "val": 3898, "test": 4269}.items():
    try:
        X = np.load(f"{S}/X_{s}.npy", mmap_mode="r")
        y = np.load(f"{S}/y_{s}.npy")
        files = jload(f"{S}/files_{s}.json")
        spk = jload(f"{S}/speakers_{s}.json")
        spk_sets[s] = set(spk)
        silent = 0
        for i in range(0, X.shape[0], 4000):
            silent += int((~(np.asarray(X[i:i + 4000]) != 0).any(axis=1)).sum())
        ok = (X.shape == (e, 16000) and X.dtype == np.int16 and y.shape == (e,) and
              int(y.min()) == 0 and int(y.max()) == 10 and len(files) == e and len(spk) == e)
        check(f"speech {s}", ok, f"X={X.shape} {X.dtype} y={y.shape} {y.dtype} all-zero clips={silent}")
    except Exception as ex:
        check(f"speech {s}", False, repr(ex))
if len(spk_sets) == 3:
    ov = (len(spk_sets["train"] & spk_sets["val"]) + len(spk_sets["train"] & spk_sets["test"]) +
          len(spk_sets["val"] & spk_sets["test"]))
    check("speech speaker-separated splits", ov == 0, f"overlapping speakers={ov}")
try:
    sm = jload(f"{S}/samples.json")
    missing = [d["file"] for d in sm if not os.path.exists(f"{S}/samples/{d['file']}")]
    check("speech samples", len(sm) == 20 and not missing, f"entries={len(sm)} missing={len(missing)}")
    check("speech classes.json", len(jload(f"{S}/classes.json")) == 11)
    check("speech LICENSE copied", os.path.exists(f"{S}/LICENSE"))
except Exception as ex:
    check("speech samples/classes/license", False, repr(ex))

# ---------------- Lichess ----------------
L = f"{H}/lichess"
try:
    meta = jload(f"{L}/meta.json")
    N = meta["kept"]
    spec = {"boards": (np.uint8, (N, 64)), "stm": (np.uint8, (N,)), "castle": (np.uint8, (N,)),
            "ep": (np.int8, (N,)), "cp": (np.int16, (N,)), "mate": (np.int16, (N,)),
            "depth": (np.int16, (N,)), "mat": (np.int8, (N,)), "npc": (np.uint8, (N,)),
            "is_val": (np.uint8, (N,)), "shard_of": (np.uint8, (N,))}
    arrs = {}
    for name, (dt, shape) in spec.items():
        a = np.load(f"{L}/{name}.npy", mmap_mode="r")
        arrs[name] = a
        check(f"lichess {name}", a.shape == shape and a.dtype == dt, f"{a.shape} {a.dtype}")
    kings_ok = npc_ok = 0
    max_piece = 0
    for i in range(0, N, 1_000_000):
        b = np.asarray(arrs["boards"][i:i + 1_000_000])
        max_piece = max(max_piece, int(b.max()))
        kings_ok += int(((b == 6).sum(1) == 1).__and__((b == 12).sum(1) == 1).sum())
        npc_ok += int(((b > 0).sum(1) == np.asarray(arrs["npc"][i:i + 1_000_000])).sum())
    check("lichess boards: codes 0-12", max_piece <= 12, f"max={max_piece}")
    check("lichess boards: exactly one king per side", kings_ok >= 0.999 * N, f"{kings_ok}/{N}")
    check("lichess npc matches boards", npc_ok == N, f"{npc_ok}/{N}")
    stm, castle, ep = (np.asarray(arrs[k]) for k in ("stm", "castle", "ep"))
    cp, mate, isv = np.asarray(arrs["cp"]), np.asarray(arrs["mate"]), np.asarray(arrs["is_val"])
    check("lichess stm in {0,1}", stm.max() <= 1)
    check("lichess castle in 0-15", castle.max() <= 15)
    check("lichess ep in -1..7", ep.min() >= -1 and ep.max() <= 7)
    check("lichess cp within +-10000", cp.min() >= -10000 and cp.max() <= 10000, f"min={cp.min()} max={cp.max()}")
    check("lichess mate rows have cp=0", (cp[mate != 0] == 0).all(), f"mate rows={int((mate != 0).sum())}")
    check("lichess validation share ~2%", 0.015 < isv.mean() < 0.025, f"{isv.mean():.4f}")
    print("meta:", {k: meta[k] for k in ("kept", "bad", "rows_read")})
except Exception as ex:
    check("lichess", False, repr(ex))

print()
print("DISK:")
os.system(f"du -sh {H}/quickdraw/processed {H}/speech/processed {H}/lichess")
print()
print("ALL PASS" if all(results) else f"{results.count(False)} CHECK(S) FAILED", f"({sum(results)}/{len(results)} passed)")
