import sys, os, json, time, zlib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

OUT = os.path.expanduser("~/demo/data/lichess")
os.makedirs(OUT, exist_ok=True)
paths = sys.argv[1:]
TOTAL = 12_000_000
PER = TOTAL // len(paths)

MAT = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0,
       "p": -1, "n": -3, "b": -3, "r": -5, "q": -9, "k": 0}
CODE = {c: i + 1 for i, c in enumerate("PNBRQKpnbrqk")}  # 1-6 white PNBRQK, 7-12 black

boards = np.zeros((TOTAL, 64), np.uint8)   # index = row*8+col, row 0 = rank 8 (FEN order)
stm = np.zeros(TOTAL, np.uint8)            # 1 = white to move
castle = np.zeros(TOTAL, np.uint8)         # bits: K=1 Q=2 k=4 q=8
ep = np.full(TOTAL, -1, np.int8)           # en passant file 0-7 or -1
cp = np.zeros(TOTAL, np.int16)             # White's view, clipped +-10000; 0 if mate
mate = np.zeros(TOTAL, np.int16)           # mate-in-N as in source; 0 if none
depth_out = np.zeros(TOTAL, np.int16)
mat = np.zeros(TOTAL, np.int8)             # white minus black material
npc = np.zeros(TOTAL, np.uint8)            # piece count
is_val = np.zeros(TOTAL, np.uint8)         # ~2% validation
hashes = np.zeros(TOTAL, np.int64)         # only for the duplicate count, not saved
shard_of = np.zeros(TOTAL, np.uint8)
n = 0
bad = 0
rows_read = 0
t0 = time.time()


def decode(placement):
    row = [0] * 64
    idx = 0
    m = 0
    cnt = 0
    for ch in placement:
        if ch == "/":
            continue
        if "1" <= ch <= "8":
            idx += ord(ch) - 48
            continue
        code = CODE.get(ch)
        if code is None or idx > 63:
            return None
        row[idx] = code
        idx += 1
        cnt += 1
        m += MAT[ch]
    if idx != 64:
        return None
    return row, m, cnt


def select(fen_arr, depth_np):
    k = len(fen_arr)
    if k == 1:
        starts = np.array([0])
    else:
        neq = pc.not_equal(fen_arr.slice(1), fen_arr.slice(0, k - 1)).to_numpy(zero_copy_only=False)
        starts = np.concatenate(([0], np.flatnonzero(neq) + 1))
    sizes = np.diff(np.append(starts, k))
    gid = np.repeat(np.arange(len(starts)), sizes)
    gmax = np.maximum.reduceat(depth_np, starts)
    idxs = np.flatnonzero(depth_np == gmax[gid])
    _, first = np.unique(gid[idxs], return_index=True)
    return starts, gid, idxs[first]


for si, path in enumerate(paths):
    pf = pq.ParquetFile(path)
    ngroups = pf.metadata.num_row_groups
    limit_n = n + PER
    carry = None
    print("shard", si, path, "| row groups:", ngroups, "| target positions:", PER, flush=True)
    for g in range(ngroups):
        if n >= limit_n:
            break
        tbl = pf.read_row_group(g, columns=["fen", "depth", "cp", "mate"])
        rows_read += tbl.num_rows
        fen = tbl.column("fen").combine_chunks()
        dep = tbl.column("depth").combine_chunks().to_numpy(zero_copy_only=False)
        mt = tbl.column("mate").combine_chunks()
        cpa = tbl.column("cp").combine_chunks()
        hm = pc.is_valid(mt).to_numpy(zero_copy_only=False)
        mt_np = pc.fill_null(mt, 0).to_numpy(zero_copy_only=False)
        cp_np = pc.fill_null(cpa, 0).to_numpy(zero_copy_only=False)
        if carry is not None:
            fen = pa.concat_arrays([carry[0], fen])
            dep = np.concatenate([carry[1], dep])
            cp_np = np.concatenate([carry[2], cp_np])
            mt_np = np.concatenate([carry[3], mt_np])
            hm = np.concatenate([carry[4], hm])
        last = (g == ngroups - 1)
        starts, gid, chosen = select(fen, dep)

        if si == 0 and g == 0:
            okc = totc = 0
            for c in chosen[:20000]:
                if c + 1 < len(dep) and gid[c + 1] == gid[c] and dep[c + 1] == dep[c] and not hm[c] and not hm[c + 1]:
                    black = fen[int(c)].as_py().split()[1] == "b"
                    totc += 1
                    okc += int(cp_np[c] <= cp_np[c + 1]) if black else int(cp_np[c] >= cp_np[c + 1])
            print("rank check (first row = best line for side to move): %d / %d ok" % (okc, totc), flush=True)

        if not last:
            s = int(starts[-1])   # the final group may continue in the next row group
            carry = (fen.slice(s), dep[s:], cp_np[s:], mt_np[s:], hm[s:])
            chosen = chosen[:-1]
        else:
            carry = None

        room = limit_n - n
        chosen = chosen[:room]
        if len(chosen) == 0:
            continue
        fens = fen.take(pa.array(chosen)).to_pylist()
        k0 = n
        rows = []
        for f, c in zip(fens, chosen):
            parts = f.split()
            if len(parts) < 4:
                bad += 1
                continue
            d = decode(parts[0])
            if d is None:
                bad += 1
                continue
            row, m, cnt = d
            rows.append(row)
            stm[n] = 1 if parts[1] == "w" else 0
            cs = parts[2]
            castle[n] = (1 if "K" in cs else 0) | (2 if "Q" in cs else 0) | (4 if "k" in cs else 0) | (8 if "q" in cs else 0)
            ep[n] = (ord(parts[3][0]) - 97) if parts[3] != "-" else -1
            if hm[c]:
                mate[n] = max(-30000, min(30000, int(mt_np[c])))
            else:
                cp[n] = max(-10000, min(10000, int(cp_np[c])))
            depth_out[n] = int(dep[c])
            mat[n] = max(-127, min(127, m))
            npc[n] = cnt
            is_val[n] = 1 if zlib.crc32(f.encode()) % 50 == 0 else 0
            hashes[n] = hash(f)
            shard_of[n] = si
            n += 1
        if rows:
            boards[k0:n] = np.array(rows, dtype=np.uint8)
        sl = slice(k0, n)
        if n > k0:
            nm = mate[sl] == 0
            med = np.median(np.abs(cp[sl][nm])) if nm.any() else 0
            print("shard %d rg %d/%d | kept %d | batch: pieces %.1f, white-to-move %.3f, median|cp| %.0f, mate share %.3f | %.0fs" % (
                si, g + 1, ngroups, n, npc[sl].mean(), stm[sl].mean(), med, 1 - nm.mean(), time.time() - t0), flush=True)

print("rows read:", rows_read, "| positions kept:", n, "| bad:", bad)
sl = slice(0, n)
for name, arr in (("boards", boards), ("stm", stm), ("castle", castle), ("ep", ep), ("cp", cp),
                  ("mate", mate), ("depth", depth_out), ("mat", mat), ("npc", npc), ("is_val", is_val),
                  ("shard_of", shard_of)):
    np.save(os.path.join(OUT, name + ".npy"), arr[sl])
json.dump({"shards": paths, "per_shard_cap": PER, "kept": n, "bad": bad, "rows_read": rows_read,
           "convention": "cp and mate are White's view; boards index=row*8+col, row0=rank8; pieces 1-6 PNBRQK white, 7-12 black",
           "selection": "per FEN, first row among rows at the greatest depth"},
          open(os.path.join(OUT, "meta.json"), "w"))

print("duplicate FENs in the kept set:", n - len(np.unique(hashes[sl])))
print("--- per shard ---")
for s in range(len(paths)):
    m_ = shard_of[sl] == s
    nm = mate[sl][m_] == 0
    print("shard %d: %d positions | mean pieces %.1f | <=10 pieces %.3f | white-to-move %.3f | median|cp| %.0f | mate share %.3f" % (
        s, m_.sum(), npc[sl][m_].mean(), (npc[sl][m_] <= 10).mean(), stm[sl][m_].mean(),
        np.median(np.abs(cp[sl][m_][nm])), 1 - nm.mean()))
print("validation positions:", int(is_val[sl].sum()))
mp, mn = mate[sl] > 0, mate[sl] < 0
print("mate>0: %d, mean material %.2f | mate<0: %d, mean material %.2f" % (
    mp.sum(), mat[sl][mp].mean() if mp.any() else 0, mn.sum(), mat[sl][mn].mean() if mn.any() else 0))
print("(mate>0 should have the higher material if positive means White mates)")
