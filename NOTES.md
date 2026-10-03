# NOTES: assumptions, findings, open issues

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


---

# Phases 2-11 (one-shot build): decisions, findings, open issues

## Owner decisions applied
- **Accepted data imperfections are statistics, never failures.** The Quick, Draw! "larger side spans N" (normalisation slack) and the
  Lichess `mat`-vs-boards mismatches are counted and printed by `python -m app.data.contract` (see "Statistics") and by
  `check_data_dir(..., stats=...)`. Nothing in the code reads the stored `mat` array: material is always computed from the boards
  (`encoder.material_from_boards`, `tournament/engine/evaluators.py`, `play.material_diff`).
- **Branch.** Everything is on `claude/upbeat-gauss-3pb7rg` (restarted from `main` after the phase 1 merge). Nothing was pushed to `main`.
- **The metrics trap uses `marvin`** (SPEC 4.2 / 5.3: ~2% positives, "always no" scores ~98%), not the "cat-vs-rest" wording of the build prompt:
  SPEC.md is the source of truth. The cached JSON also contains a class-weighted variant that shows the fix.

## Design calls (things SPEC left open)
**Data and features**
- *Rasterizer* (`shared/RASTERIZER.md`): normalise (translate to the origin, larger side to 255, aspect ratio kept, **not centred**, matching the
  top-left alignment of the data), map into 28x28 with a 2 px margin, analytic anti-aliased lines of width 2 (`clamp(1.5 - dist, 0, 1)`), union by max.
  Python and TypeScript agree to within 1 grey level on the golden fixtures (Python reproduces its own fixture exactly).
- *Log-mel* (`shared/AUDIO_FEATURES.md`): 25 ms / 10 ms periodic-Hann frames, 512-point FFT, 40 HTK-mel triangles 20-7600 Hz, `ln(x + 1e-6)`,
  per-clip standardisation; 40 x 98. The raw-waveform model gets `int16/32768` average-pooled by 4 (4000 inputs) so it stays small in the browser.
- *Chess input*: 768 piece planes (+5 side-to-move/castling, +8 en passant, +10 material counts, +128 attack maps, each optional). Attack maps ignore pins
  (pseudo-attacks, defended own pieces count) and are verified against python-chess on 200 real positions. The colour-flip augmentation is an exact
  symmetry (verified against `chess.Board.mirror`).
- *Targets*: `winprob` = `sigmoid(cp/K)`, `cp` = clipped centipawns / K; mates become +-`mate_clip`; with `perspective_flip` the target is for the side to move
  (the **input is not mirrored**, only the target and the stm flag). Output heads map into target space (`encoder.head_to_target_space`); `bce` needs `winprob`.
- *Comparable validation number*: every run reports `val_mse` = win-probability MSE at K=400 / clip 2000 (the same for every config) next to its own `val_loss`.
- *Parameters / FLOPs*: the first layer is included; biases and LayerNorm affine are counted in parameters; FLOPs = 6 x matmul weights x samples (biases/norms excluded).
  The activation list includes `hard_step` on purpose (it cannot learn: a fun way to lose).

**Training and scheduling**
- `BUDGET_FLOPS` default is a placeholder `1e16`; `scripts/benchmark.py` section 6 calibrates it (9-minute median config) into `STATE_DIR/calibration.json`.
- Doodle race net: MLP 784-128-64-10, momentum SGD by default, MSE/L1/Huber on output *probabilities* averaged over elements (slow on purpose: the drama).
- **Recorded races are one lane alone on the GPU.** The frontend composes up to 3 recorded lanes, so a replayed lane gets ~3x more updates per second than the same
  lane in a live 3-lane race. For batch-size drama this is mostly fine; if you want exact parity, re-record with 3 identical dummy lanes sharing the GPU.
- Pause and Demo Mode both **preempt** a running competition job (it returns to the front and loses only the seconds since its last checkpoint). Kill/redo beat
  an automatic preemption (abort-reason priority). A race that is queued behind another race waits FCFS; the browser falls back to the recording after 9 s without a tick.
- A diverged (NaN) run ships its **last finite checkpoint**, or no model at all if it diverged before the first checkpoint. The sanity gate then excludes broken ones.
- Nuclear reset keeps `models/house-net/` (it is not a participant run) and snapshots the DB first. The default-config baseline lives in `baselines/`, outside `models/`.
- The admin export leaves out `contact` unless `include_contact=true` (SPEC 13).
- DB migration v1 -> v2 adds `submissions.client_id` (idempotent uploads after a network drop). Participants poll their own status by the unguessable `submission_id`.
- Submissions require the consent tick (the form says it covers the contact and anonymised logging).

**Frontend**
- The layout is designed for 1600x900 and scaled to the viewport with CSS `zoom` (0.55x-1.6x). Projector resolution is untested.
- Duel: drawings replay at a constant 420 data-units/s; the 4 options appear after 3 strokes (or 40% of the path); the model in the rematch sees exactly the part drawn when the human answers.
- Act 1 uses a tiny TS MLP (Adam, mini-batch 64, BCE) in a Web Worker; "depth" = hidden layers (linear stack width 8), "width" = hidden units; the gallery trains 3 x 16.
- The offline marker is visible only after pressing `P` (so the audience never sees it); the phone's `/presenter` page shows a red banner when the server is down.
- Presenter sync: `BroadcastChannel` (same browser) and `PUT/GET /api/presenter` (phone over the laptop's LAN address). The latter needs `FRONTEND_HOST=0.0.0.0` (exposes the dev server on Wi-Fi).
- Mic: `ScriptProcessorNode` (deprecated but universal), hold-to-talk, the 1-second window with the most energy, peak-normalised only when very quiet. Failure falls back to the 20 sample clips automatically.
- The 3D valley is a one-dimensional cartoon drawn as a surface (height depends on one knob); WebGL missing -> 2D version.
- The Vite proxy serves the SPA for browser navigations to `/admin` and proxies everything else under `/admin` to the backend (they share a prefix).
- Stage hint for Act 3: `,` and `.` switch clips (the arrow keys are stage navigation).

**Tournament**
- Search: negamax alpha-beta, iterative deepening, leaf batches per parent node, fixed ordering (MVV-LVA then UCI), alphabetical exact-tie break at the root, no capture
  extension, no repetition detection inside the search (the game loop adjudicates threefold/fifty-move). Mate scores are +-(100000 - ply).
- Depth per model = 2 x `search_depth_full_moves` plies, optionally capped (`--depth-cap-plies 4`) or limited by `--node-cap`.
- Swiss pairing by score while `rounds <= n/2 - 1` (a pairing without repeats then always exists); beyond that a deterministic circle-method round robin.
  An unpaired model plays a reference bot twice; with an odd count nobody rests twice before everyone has.
- Sanity gate: one game vs the random mover as White at 2 plies, 80-ply cap; it fails if the model forfeits (NaN, illegal move) or **loses** (a draw passes).
  An untrained net can lose that game: `--override-sanity` keeps it.
- Games end at 150 plies: a side ahead by >= 2 pawns wins by adjudication. Bracket: top 8 by rating, 2 games per match (colours swapped), a third game on a tie (higher seed White), then the higher seed advances.
- The random mover is deterministic (CRC of the position), so repeat runs are byte-identical (tested). Stockfish is optional (`--stockfish PATH`, fixed depth, 1 thread).
- Ratings: Bradley-Terry (MM) with a weak prior so perfect records stay finite; Elo = 400 log10; **the random mover is pinned to 0**.

## Things that could not be verified here (need the real box / data / laptop)
See FINAL_REPORT.md section 3. In short: every CUDA code path, the real data volume (12M rows, 1.09M doodles), training quality and timing, the real microphone and
browsers other than Chromium, projector scaling, Windows laptop commands, the server's Python 3.12 + NVIDIA torch 2.8 install, systemd/boot start.

## Open issues
1. **Tier thresholds and depths are placeholders** (`shared/tiers.json`). The benchmark writes `tiers.suggested.json`. On this CPU sandbox even depth 2 was slower than 1.5 s/move: the GPU numbers will tell.
2. **`torch.cuda.utilization()` needs `pynvml`**; without it the admin monitor shows GPU "n/a" (memory is still shown).
3. **Browser inference**: Node/V8 measured < 10 ms for every bundled model and 4 ms for the log-mel features; run `npm run bench:inference` on the laptop for the real figure.
4. The first competition job after a restart loads the 12M-row arrays (about 850 MB to the GPU, or memory-mapped on the CPU plan): expect a pause of tens of seconds before step 1.
5. `ScriptProcessorNode` may be removed from browsers some day; replace with an AudioWorklet then.
6. Chess strength of the House Net is whatever the random search finds in the time it is given (`train_house_net.py --trials N`); you can also hand it a config with `--config file.json`.
7. Frontend playback of the tournament (bracket.json) is not built (SPEC 12 cut order lists it as a first thing to drop); the data for it is ready.
8. Start-at-boot is still manual (see the phase 1 note); `start_backend.sh` is idempotent.
