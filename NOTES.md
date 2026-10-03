# NOTES — assumptions, findings, open issues

Items are tagged **[owner]** when the owner must check or decide something.

## Phase 1 findings

### Quick, Draw! coordinates: the SPEC 4.1 [verify] item only partly holds
Checked on all 266 drawings in `data/samples/quickdraw/` (see `tests/test_samples.py::test_quickdraw_coordinates_verify_item`):
- Every coordinate is an integer in 0..255. ✅
- "Aligned to the top-left, larger side exactly 255" holds for only ~76% of drawings. The rest are
  off by 1-2 px: min x or min y is 1-2 instead of 0, or the larger side spans 253-254 instead of 255.
  This looks like rounding in Google's simplification.
- **Consequence:** the rasterizer must re-normalize every drawing (translate to the origin, scale the
  larger side to 255), which SPEC 4.1 already requires. Never assume the stored coordinates are
  exactly normalized. The contract checker allows 3 px of slack (`QUICKDRAW_NORM_TOL`), and
  `fake_data.py` reproduces the same 1-2 px imperfections so loaders get tested against them.

### Other facts confirmed on the samples
- Lichess samples: exactly one king per side, `npc` = pieces on the board, `mat` = white minus black
  material, no pawns on rank 1/8, `mate != 0` ⇒ `cp == 0`, both shards present (1000 rows each),
  ~13% mate rows, ~1.8% validation rows, and positions with positive `mate` have higher average material.
  All consistent with SPEC 4.3.
- Quick, Draw! split provenance: every duel/probe key hashes into the pool bucket and every samples
  key into the val bucket (md5 rule from `data_prep/prep_quickdraw.py`).
- Speech samples: mono, 16-bit, 16 kHz, at most 16000 frames, 2 per word for labels 0-9.
- The Lichess prep script clips `mate` to ±30000 (the samples range only ±56) and `mat` to int8.

### Bug found and fixed: backend hung on shutdown while a browser was connected
Uvicorn's graceful shutdown waits for open connections, and an SSE stream never ends on its own.
So Ctrl+C or `kill` left the backend stuck at "Waiting for connections to close" for as long as the
page was open. Fix: `timeout_graceful_shutdown=3` in `app/main.py`. Regression test:
`tests/test_server_process.py` (verified to fail without the fix). Every future SSE endpoint
(race streams, admin monitor) benefits automatically.

### Bug found and fixed: page didn't notice a dead backend through the Vite proxy
Vite's dev proxy kept the browser's side of an SSE stream open after the backend died, so the page
never noticed. Two fixes:
1. `vite.config.ts` destroys the browser connection when the upstream closes mid-response.
2. The page runs a watchdog that reopens the stream if no tick arrives for 4 s. This also covers a
   silently dead tunnel (Wi-Fi drop), where no close ever arrives. Tested in a real browser
   (Playwright): backend killed → "reconnecting" → restarted → "live"; backend frozen → detected in ~5 s.
   The phase 4 offline logic (`/api/health` polling) should reuse the same idea.

## Decisions / assumptions (phase 1)

- **Branch.** SPEC 0 says "small commits to `main`", but this Claude Code session is set up to push to
  `claude/upbeat-gauss-3pb7rg`. Work is pushed there. **[owner]** Merge it into `main` (pull request) or
  tell Claude to push to `main` directly in future sessions.
- **`/api/health` includes `ok`** (SPEC 7 lists it; the phase 1 prompt did not). Always `true` when
  the server answers. `queue_length` = jobs with status `queued` or `running`.
- **`demo_mode` and `paused` are stored as admin events**, not in memory, so they survive a power
  cycle (SPEC 0.1, 6.3). The latest `pause`/`resume` event decides `paused`; the latest `demo_mode`
  event (`{"enabled": bool}`) decides `demo_mode`. No extra table was needed beyond the six in SPEC 6.3.
- **Where state lives:** `STATE_DIR` (default `~/demo/state`) holds `demo.db`. Later phases put
  `models/` and `checkpoints/` there too. SPEC 6.3 says "models/<submission_id>/" without a base
  directory; `STATE_DIR/models` is the assumption. `DATA_DIR` stays read-only.
- **Schema** (`app/db.py`) is a first draft for phases 3-6: columns for the stop reasons, FLOPs,
  active GPU seconds, preemptions, queue order (preempted jobs go to the front), checkpoint
  bookkeeping, and race lanes. `PRAGMA user_version` = 1. Later phases may add columns. There is no
  migration framework yet; while there is no real data, deleting `demo.db` is the migration.
- **Self-check in phase 1** reports dataset files by presence, dtype/shape (from `.npy` headers via
  mmap) and size. It doesn't validate contents (that's `python -m app.data.contract`, which is
  slower). The "one-step training smoke test" is one SGD step on a tiny standalone MLP; it moves to the
  real training core in phase 3. If the self-check fails, the server still starts (so `/api/health`
  works) and prints `PROBLEMS`. `quickdraw/tensors/` is reported as "not built yet" until phase 2.
- **`app/data/contract.py`** is the executable SPEC 4 contract. One checker runs against fake data,
  `data/samples/` and the real `DATA_DIR`. `--real` additionally checks the exact counts from SPEC 4.
  It reads big arrays with mmap in 1M-row chunks; quickdraw JSONL lines are all counted but only the
  first `--max-records` per split are fully parsed (a full parse of 1.1M lines also works, just slower).
- **`fake_data.py` sizes.** `n` scales everything; explicit sizes can be passed. Fixed-size derived
  files (duel 200, probe 16, samples 50, speech samples 20) force minimums: quickdraw pool ≥ 220,
  val ≥ 50; speech test ≥ 22. Fake keys are chosen so the md5 split rule holds, fake speech splits are
  speaker-separated, fake lichess positions obey all audited invariants, and some include castling
  and en passant consistent with the board.
- **Vite proxy target is `http://127.0.0.1:8000`**, not `localhost:8000`: same port and same tunnel,
  but avoids Node resolving `localhost` to IPv6 `::1` when ssh only listens on IPv4.
- **Frontend stack versions:** Vite 8, React 19, Tailwind 4 (via `@tailwindcss/vite`, no
  `tailwind.config.js`), TypeScript 7. Requires Node 20.19+ / 22.12+.
- **No `torch` in any requirements file.** `numpy` is listed without a version so pip keeps the
  server's existing one. In this sandbox, tests ran with a CPU-only PyPI torch installed outside the repo,
  and also with torch hidden (torch-dependent tests skip cleanly).
- **CORS**: not enabled. The Vite proxy is used instead (SPEC 3 allows either).
- `python3` is used in all commands (some images have no `python` alias).

## Open issues / to check on the real server

1. **[owner]** Run the README steps 1-5 on the server. In particular, check that
   `pip install --dry-run -r backend/requirements.txt` doesn't touch torch or numpy (Python 3.12.3,
   NVIDIA torch 2.8.0a0).
2. **[owner]** Self-check output on the CPU plan and later on the L40S: device name, VRAM, all dataset
   files present with the expected shapes, smoke test PASS.
3. **[owner]** `REAL_DATA=1 python3 -m pytest tests/test_real_data.py -v` on the server (checks the real
   counts and invariants of all three datasets).
4. **[owner]** Phase 1 "done when": the laptop page shows the live counter through the tunnel.
   Also try killing the tunnel: the page should show "reconnecting…" and recover when it's back.
5. **[owner] Start at boot.** SPEC 0.1 wants the backend to come back after power-on. Check whether the
   server has systemd: `ps -p 1 -o comm=` (prints `systemd` if so). Jupyter-style containers often
   don't. Without systemd, options are `crontab -e` with
   `@reboot /home/jovyan/ieee-ai-demo/backend/start_backend.sh`, if cron runs in the container,
   or the provider's startup-script setting. Tell Claude which exists and it will wire it up.
6. **[owner]** Is the laptop Windows or Mac? README covers both for the tunnel loop.
7. The `httpx`-based Starlette TestClient prints a deprecation warning with the newest Starlette.
   Harmless; revisit if it breaks.
