# SPEC v2: "Build Your Own AI" — IEEE CS @ PESU recruitment demo

This file is the single source of truth. If code and spec disagree, fix the code or flag it in `NOTES.md`. Items marked **[verify]** are guesses that a benchmark or manual check must settle.

**What changed in v2:** all datasets are now downloaded, parsed and audited by the owner *before* you start. Section 4 is a data contract built from the real files. You never download or parse raw data. Section 0.1 lists facts about the runtime environment that differ from what you would assume.

---

## 0. Rules for Claude Code

- You run in a cloud sandbox with **no GPU and none of the real data**. You see only this repo. Do not assume network access beyond installing packages.
- **Never download or stream datasets, and never write code that does.** All data already exists on the server in the formats in section 4. Your loaders read those files. Raw-data parsers are not your job (the owner's finished prep scripts live in `data_prep/` for reference only; do not modify or run them).
- Everything must be **testable in your sandbox on fake data**. Write `backend/tests/fake_data.py`, which generates tiny synthetic files in **exactly** the section 4 formats (same names, dtypes, shapes, conventions, with small N). CPU smoke tests run on that. Real GPU runs happen later on the owner's box.
- A few **real sample files** are committed in `data/samples/` (section 4.5). Use them to build golden tests and to see the true formats.
- All code is **device-agnostic**: `cuda` if available, else `cpu`. Never hardcode a GPU model. Chunk sizes and dataset residency adapt to detected VRAM (`torch.cuda.mem_get_info`).
- Read `DATA_DIR` from an environment variable (default `~/demo/data`). Never hardcode data paths.
- Build against the **API contract (section 7)**. Update this file in the same commit if you change an endpoint.
- Small commits to `main`; tag `demo-ready` only when the owner approves.
- Prefer boring, standard libraries. Nothing here is research. If something seems hard, it is probably mis-specified: make a call, document it in `NOTES.md`, move on.
- Write a `README.md` with exact copy-paste commands for: server setup, starting the backend, the SSH tunnel, starting the frontend, running tests, running benchmarks. The owner is new to web development.

### 0.1 Runtime environment facts (the server)

- **Python 3.12.3. PyTorch is preinstalled** (NVIDIA build, `2.8.0a0+...nv25.06`) in the default environment. There is **no virtualenv**. **Never `pip install torch` and never list torch in `requirements.txt`**; installing it would replace the NVIDIA build. Other packages are installed into the default environment with plain `pip install`. If an install wants to change torch or numpy, stop and flag it.
- **`nproc` and `os.cpu_count()` lie** (the box reports the physical host: 256 CPUs, 1.5 TB RAM). Never size anything from them. Worker/thread counts come from the `NUM_WORKERS` env var (default 4).
- **Hardware over time.** Setup and testing happen on a CPU-only plan (8 vCPU, 16 GB RAM, ~50 GB disk). The plan is later changed in place to an L40S (48 GB VRAM) for benchmarks, training and the event. After the event it may move to a cheaper large-VRAM GPU for the tournament. The same code must run on all of these. On the CPU plan, never load a whole dataset into RAM: stream or memory-map (`np.load(..., mmap_mode="r")`).
- **Home directory** is `/home/jovyan`. Data is in `~/demo/data`. Long jobs run in `tmux`; the backend runs as a systemd service (or tmux) that starts at boot.
- **The server will be powered down and up during the event** (to save cost). Nothing may depend on in-memory state surviving. See sections 5.1 and 6.3.
- **Participant count** is about 35 (an estimate, not a hard limit; nothing may break at 60).

---

## 1. Goal and context

The owner is tech head of PESU's IEEE CS student chapter. In a few days there is a live recruitment event with a short demo slot among other events. The audience is mostly first years who know at most vectors and maybe matrix multiplication. The demo should be fun, visual, interactive and light on math, and it should show that the club does AI research that makes training cheaper, faster or more accurate.

Two experiences share one backend:

1. **Live demo** (projected website on the owner's laptop): staged acts, audience makes choices, a few real training runs happen live on a rented GPU.
2. **"Build your own AI" competition**: participants configure a **chess evaluation network** at the booth (on the owner's laptop). Their model trains in a queue on the GPU box. A **tournament runs after the event**, started manually by the owner over SSH, never by the server.

At the event: laptop A (a friend's) runs a PPT, laptop B (the owner's) runs this website, projected, and a phone shows `/presenter` notes. Only laptop B is interactive for the audience.

---

## 2. Scope and priorities

**Must work:** Acts 0, 1, 2 with cached fallbacks for every server race; the doodle showcase model trained and bundled; the submission form, training queue, default config and House Net; the `/admin` page.

**Should work:** Act 3 (audio) with pre-recorded clips and cached curves; Act 4 (finale and rematch); the 3D valley.

**May slip:** the live microphone path in Act 3.

**Written now, run later by the owner manually:** the tournament script (section 10).

The owner decides what to cut. Every act and sub-step must be independently skippable (stage menu, key G).

---

## 3. Architecture

```
Laptop B (owner)                        Rented GPU box
┌──────────────────────┐   SSH tunnel   ┌─────────────────────────────┐
│ Vite+React frontend  │ ─────────────▶ │ FastAPI (127.0.0.1:8000)    │
│ served locally       │ localhost:8000 │  - scheduler (1 worker)     │
│ localStorage backups │                │  - training (PyTorch)       │
└──────────────────────┘                │  - SQLite (WAL)             │
                                        │  - SSE streams              │
                                        └─────────────────────────────┘
```

- The frontend is **never published**. It runs locally (e.g. `localhost:5173`) and calls `http://localhost:8000`, which an SSH tunnel forwards to the server (`ssh -N -L 8000:localhost:8000 e2e`, kept alive by a reconnect loop; `ServerAliveInterval 15` in the SSH config). The backend binds to `127.0.0.1` only. Because the frontend and backend are on different localhost ports, either use a **Vite dev proxy** for `/api` and `/admin`, or allow CORS for `http://localhost:*`.
- Backend: FastAPI + PyTorch + SQLite (WAL) + Server-Sent Events.
- One GitHub monorepo. Claude Code pushes; the owner `git pull`s on the laptop and the server.
- Everything not needing live compute is done **before** the event: showcase models, cached races, House Net. Live, only the buttons that launch real training do work.

### Repo layout

```
SPEC.md  README.md  NOTES.md
data_prep/         # owner's finished prep scripts + logs. REFERENCE ONLY.
data/samples/      # small real samples (section 4.5), committed
shared/            # JSON schemas, rasterizer + audio feature specs, golden fixtures
backend/
  app/             # main.py, api/, scheduler/, training/, data/, db.py, config.py
  scripts/         # rasterize_quickdraw.py, benchmark.py, train_showcase.py,
                   # record_races.py, train_house_net.py, export_weights.py
  tests/           # includes fake_data.py
tournament/        # engine/, run.py (CLI), ratings.py, tests/
frontend/          # Vite + React + TS + Tailwind
```

---

## 4. Data contract (the files that already exist on the server)

All under `DATA_DIR` (default `~/demo/data`). Total about 2.5 GB. Everything below was audited by the owner; counts are exact. **Do not download, re-slice or re-parse anything.** `backend/scripts/rasterize_quickdraw.py` is the only data-processing script you write (stroke JSON to image tensors), plus feature extraction for audio.

### 4.1 Quick, Draw! — `quickdraw/processed/`

| File | Content |
|---|---|
| `train.jsonl` | 1,090,830 lines (per class: cat 100,830, all others 110,000) |
| `val.jsonl` | 25,106 lines |
| `pool.jsonl` | 2,579 lines (held out, never in train or val) |
| `classes.json` | `["cat","bicycle","house","pizza","lightning","star","fish","tree","umbrella","sword"]` (index = label) |
| `duel.json` | 200 records (20 per class, from `pool`), for the Act 0 / Act 4 duel |
| `probe.json` | 16 records (one per class plus 6 extra, from `pool`), for the race "probe wall" |
| `samples.json` | 50 records (5 per class, from `val`) |

- **Record format** (each JSONL line, and each element of the JSON arrays): `{"c": <int 0-9>, "k": "<key_id string>", "d": [[[x...],[y...]], ...]}`. `d` is a list of strokes; each stroke is `[xs, ys]` with equal-length integer lists. Coordinates come from Google's "simplified" format: aligned to the top-left, uniformly scaled so the larger side is **255** (**[verify]** with `data/samples/` and the golden test). **There is no timing data**: Act 0's replay animates strokes at constant speed.
- Only drawings Google marked `recognized: true` were kept. Splits are deterministic (md5 of `key_id`).
- **Rasterizer.** Write one spec in `shared/RASTERIZER.md`: output **28×28** grayscale, uint8, line width, anti-aliasing, and how input strokes are normalized (translate to the origin, scale the larger side to 255, **preserving aspect ratio**, then render). Implement it in Python and in TypeScript. User-drawn strokes in the browser go through the **same normalization** before rasterizing, or the model sees off-scale input. Both must pass **golden fixtures** (`shared/golden/`, built from `data/samples/quickdraw/`) within a stated tolerance.
- `rasterize_quickdraw.py` writes `quickdraw/tensors/{train,val,pool}_x.npy` (uint8, N×28×28) and `*_y.npy` (int8) and is idempotent. Training keeps train tensors resident in GPU memory (about 0.85 GB).

### 4.2 Speech Commands v0.02 — `speech/processed/`

| File | Content |
|---|---|
| `X_train.npy`, `X_val.npy`, `X_test.npy` | int16, shape `(N, 16000)`: 1 second at 16 kHz, mono. N = **32,479 / 3,898 / 4,269** |
| `y_train.npy`, `y_val.npy`, `y_test.npy` | int8 labels, shape `(N,)` |
| `files_*.json`, `speakers_*.json` | original relative paths and speaker ids, one per row |
| `classes.json` | `["yes","no","up","down","left","right","on","off","stop","go","marvin"]` (index = label) |
| `samples.json`, `samples/` | 20 real `.wav` clips (2 per word, labels 0-9, from the test split) with `{file,label,word}` |
| `LICENSE`, `README.md` | dataset licence (CC BY 4.0) and readme |

- Labels **0-9** are the ten command words; **label 10 (`marvin`)** is reserved for the metrics-trap detector and must be excluded from the 10-word classifier. `marvin` counts: train 1,710, val 195, test 195.
- Splits are the **official** speaker-separated lists (verified: no speaker appears in two splits). Clips shorter than 1 second (3,840 of them) are **zero-padded at the end**. Convert to float with `x / 32768.0`.
- Write `shared/AUDIO_FEATURES.md` for the log-mel pipeline (FFT size, hop, mel bins, log floor, normalization) and implement it in Python and TS, with golden tests against `samples/`.
- **Metrics-trap subset** (built at training time, fixed seed): positives = `marvin`; keep all negatives and subsample positives so they are about **2%** of the data (about 630 positives against about 30,800 negatives in train).
- **Overfitting subset:** a tiny fixed-seed subset of train (for example 300 clips) with a large model.

### 4.3 Lichess evaluated positions — `lichess/`

Source: the Lichess evaluation database (CC0), 12,000,000 positions taken from two shards, one position per row. Files (N = **12,000,000**), all `.npy`:

| File | dtype, shape | Meaning |
|---|---|---|
| `boards` | uint8 `(N,64)` | Piece codes: **0 empty; 1-6 white P N B R Q K; 7-12 black p n b r q k**. Index = `row*8+col` with **row 0 = rank 8**, col 0 = file a (so index 0 = a8, index 63 = h1) |
| `stm` | uint8 `(N,)` | Side to move: **1 = white, 0 = black** |
| `castle` | uint8 `(N,)` | Bits: 1 = white kingside, 2 = white queenside, 4 = black kingside, 8 = black queenside |
| `ep` | int8 `(N,)` | En passant file 0-7, or **-1** |
| `cp` | int16 `(N,)` | Centipawns from **White's point of view** (always, regardless of side to move), clipped to ±10,000. **0 when the position has a mate score** |
| `mate` | int16 `(N,)` | Mate-in-N; **positive = White mates** (verified), **0 = no mate score**. A row with `mate != 0` always has `cp == 0` |
| `depth` | int16 `(N,)` | Search depth of the chosen evaluation |
| `mat` | int8 `(N,)` | White minus black material (P=1, N=B=3, R=5, Q=9) |
| `npc` | uint8 `(N,)` | Piece count |
| `is_val` | uint8 `(N,)` | **1 for the validation set** (239,546 positions, about 2%); all others are training |
| `shard_of` | uint8 `(N,)` | 0 or 1 (source shard) |
| `meta.json` | | Provenance, conventions, counts |

- **Selection rule used:** per position, the first row among those at the greatest search depth (the engine's best line). Verified: no duplicate positions; best-line ordering confirmed. The two shards differ slightly in composition (shard 0 has more endgames and a lower median |cp|); the blend is intentional.
- The data has **no game results, no player ratings, and no move or capture information**. Do not build features that need them.
- Rough composition: about 18% of positions have 10 or fewer pieces; about 12% have a mate score; median |cp| is about 80; the mean piece count is about 20.
- Memory: on the GPU box load the arrays to GPU as uint8 (about 1 GB) and expand boards to 768 planes per batch on the GPU. On the CPU box use `mmap_mode="r"`.

### 4.4 Playground
Generated in the browser (clusters, spirals). No files.

### 4.5 Samples committed in the repo — `data/samples/`
Real but tiny. Use these to learn the formats and to build golden fixtures:
- `data/samples/quickdraw/`: `classes.json`, `duel.json`, `probe.json`, `samples.json` (copied verbatim).
- `data/samples/speech/`: `classes.json`, `samples.json`, `samples/*.wav`, `LICENSE`.
- `data/samples/lichess/`: a 2,000-row subsample of every `.npy` listed in 4.3 (every 6,000th row, so both source shards are represented), plus `meta.json` (which records the subsampling).

`fake_data.py` must produce the same layout at any requested size, so the same loaders work on fake data, samples and the real data.

---

## 5. Frontend

**Stack:** Vite, React, TypeScript, Tailwind, Framer Motion for transitions, D3 and canvas for custom charts, react-three-fiber only for the 3D valley. Act 1 training runs in a **Web Worker**.

**Look and feel.** A beautiful, attention-grabbing educational site (Desmos / Brilliant style). Dark theme, minimum ~24px projected text, colorblind-safe palette (blue vs orange for classes). Polish budget goes into three hero visuals: the bending decision boundary, the glowing forward pass, the rolling ball. Everything else stays clean and simple.

### 5.1 Stage machine, progressive disclosure, offline behavior
- The site is an ordered list of stages. Each stage declares which UI elements it **unlocks**. Elements not yet unlocked do **not exist** on screen; they pop in with a short animation when unlocked.
- Keys: `→` next build, `←` back, `Space` train/run, `S` skip to cached result, `R` reset stage, `G` hidden stage-jump menu, `F` toggle force-fallback for the current stage.
- A thin timer ring shows the time cap while a run is live.
- **Accuracy gauge** (not a bare number): a bar or dial whose color blends red → yellow → green, with thresholds **relative to chance for each task** (spirals 50%, doodles 10%, audio 10%). The numeric value may appear small beside it.
- Credits footer: tiny box with dataset credits and links (section 14).
- **Server offline is a normal state.** The frontend polls `/api/health` every few seconds. When it fails: switch races to cached playback automatically, show a discreet "offline" marker visible only to the presenter, and keep every browser-only act working. Submissions made while offline are saved locally as **pending** and upload automatically when the server returns. When the server comes back, the status marker clears itself.
- Every submission is saved to `localStorage` before it is sent.

### 5.2 Routes
`/` demo. `/presenter` talk track for the current stage (phone). `/build` submission form. `/admin` admin panel (token). Admin is never linked from other pages.

### 5.3 Acts

**Act 0: The duel (browser only).**
A doodle from `duel.json` replays stroke by stroke. After a few strokes four option buttons fade in and a 3-second ring counts down. The volunteer presses 1-4. "The AI" is a random guesser answering at the same moment. Five rounds. Only the doodle, buttons, score and gauge are on screen. The human's score is kept for the finale.

**Act 1: Playground (browser only, Web Worker).**
1. Two clusters, a random line, gauge near 50%, a "Guess again" button.
2. Three sliders appear (tilt, tilt, slide) and the equation `w1·x + w2·y + b`. "Find best knobs" animates to the least-squares line (a small linear solve). The word *parameters* appears.
3. The dots morph into two spirals. The line fails; the gauge goes red.
4. A depth slider (1-8) appears. Training with **linear layers only** keeps a flat boundary. On the reveal click the stacked blocks fuse into one block labeled `W₃·W₂·W₁ = W`.
5. ReLU appears between layers with a `max(0, x)` plot; a width slider (2-64) unlocks. Training bends the boundary around the spirals (hero visual 1). Depth 1 / width 2 still fails.
6. **Activation gallery** with live f(x) and f′(x) plots; sliders reshape both:
   - Fixed: ReLU, sigmoid, tanh, GELU, hard step (slope is zero everywhere, so it cannot learn; shows a red "broke" badge).
   - Families with a slider: Leaky ReLU (α; at α=1 it is linear and the collapse returns), Swish/SiLU (β), Softplus (β), ELU (α), Clipped ReLU (cap).
   - A "broke" badge appears on NaN loss or flat accuracy. A user-defined function card is **removed for now**; keep the activation list data-driven so adding one later is a one-entry change.
- Implement a small MLP + backprop in TS in a Web Worker. Training takes seconds; a hidden presenter option slows it for talking time.

**Act 2: Doodles (server races plus a bundled model).**
1. **Forward pass (hero visual 2).** A volunteer draws on a canvas. Pixels feed into the network as glowing signals; a top-5 bar chart updates after every stroke, using the **bundled showcase model** in the browser (works offline). Input goes through the shared rasterizer.
2. **Loss race.** The audience picks up to 3 lanes from: MSE on output probabilities, cross-entropy, L1/MAE, Huber (δ slider), Focal (γ slider), label-smoothed cross-entropy (ε slider). All lanes train the same network on the same data at **equal wall-clock time**. The main canvas is a **probe wall** of the 16 `probe.json` doodles that flip red→green as each model gets them right, with gauges beside it. Loss curves go on separate small charts (different losses are not comparable).
3. **Backward pass and learning rate (hero visual 3).** Backward-arrow animation over the network diagram, next to a 3D ball-in-a-valley toy (browser only; label it honestly as a cartoon). A step-size slider makes the ball crawl, glide or overshoot. Then a 3-lane race: too small, good, audience's pick.
4. **Batch size.** Three lanes: batch 1, full batch, audience pick. Fixed learning rate, equal wall-clock time, x-axis in seconds. Live counters: updates done and doodles seen.
- **Fallback:** every race has a cached stream (section 9), played with `S` or `F`, with a small hideable "recorded" badge.

**Act 3: Audio.**
1. **Preprocessing (live mic if possible).** Push-to-talk, close mic. A waveform appears, then a spectrogram (plain canvas). Two pre-trained models run in the browser on the same recording: a raw-waveform model (expected to flail) and a log-mel model (expected to work). A step view shows waveform → windows → Fourier transform → mel scale → log. Audio never leaves the browser and is never stored. **[verify]** the accuracy gap and browser inference speed (fp16 or a smaller input if needed). Fallback: the 20 clips in `speech/samples` run through the same visualization.
2. **Metrics trap (cached).** The `marvin` detector, where "always say no" scores about 98% accuracy; a confusion matrix and recall reveal the trap.
3. **Overfitting (cached).** A large model on the tiny subset: train and validation curves diverge. A training-set-size slider replays precomputed runs.

**Act 4: Finale (browser only).**
- **Tech tree:** every revealed control plus the extras (optimizer, init, normalization, ...) unlock together as a "full control panel", bridging to the competition.
- **Rematch:** the same duel as Act 0 against the trained doodle model, which sees only the strokes drawn so far.
- **Close:** a session stats screen (runs, parameters trained, FLOPs this session), the booth link/QR, a one-line research pitch, then hand back to the PPT.

### 5.4 Submission form (`/build`)
- Fields in section 8.3. Core knobs visible; advanced knobs behind an "At your own risk" section with defaults.
- Live **parameter count** and **tier badge**: "your bot searches N moves ahead", updating as sliders move. Publish the tier table on the form; give no hints about which tier is best.
- Nickname and model name are public; contact (optional) is private. A consent checkbox covers contact and anonymized logging.
- On submit: save to `localStorage` first, then POST. Show the participant code and status. "Download backup" exports all local submissions as JSON.

### 5.5 Admin page (`/admin`)
Token-protected (entered once, kept in memory).
- Queue table: every job with status, nickname, config summary, stop reason.
- Per job: kill running, remove queued, redo (re-queue from scratch). A confirm dialog on each.
- Queue controls: pause/resume, Demo Mode.
- Live monitor: current job curves, GPU utilization, log tail.
- **Nuclear reset:** type `RESET` plus a hold-to-confirm button. The server snapshots the database to a timestamped file first.
- "Export everything" downloads all submissions and run stats as JSON/CSV.

---

## 6. Backend

### 6.1 Scheduler
- **One worker, FCFS queue.** Jobs: `demo_race` (priority) and `competition` (normal).
- **Demo preempts by killing.** When a demo race starts, the worker sets an abort flag checked between training steps. The running competition job stops within a step or two **without saving**, returns to the **front** of the queue, and later resumes from its last periodic checkpoint. Losing a few seconds of work is acceptable.
- **Checkpoints** every ~10 s of active GPU time (configurable): model, optimizer, schedule state, RNG state and `samples_seen`, so resumed runs do not double-count FLOPs.
- **Races** are one job group with up to 3 lanes interleaved in short slices (~100-300 ms) so wall-clock fairness is automatic. All lanes share data and seed.
- **Full-batch lanes** use gradient accumulation in chunks sized from available VRAM (math identical to a true full batch).
- **Demo Mode** (admin toggle) pauses competition jobs entirely.
- When the queue is empty the server idles and does **nothing** further: no ranking, no matches (idle cost accepted).

### 6.2 Training budget (competition)
- Stop at the **FLOPs budget** or the **active-GPU-time cap**, whichever comes first. FLOPs ≈ `6 × params × samples_seen` (count matmul weights; log exact too).
- The time cap is a safety net, roughly 3× the nominal time for the median config; it should rarely fire. Time is **active GPU seconds**, not wall clock.
- Calibrate `BUDGET_FLOPS` from the benchmark so the median config (~4 layers × 1024, batch 256) takes about 8-10 minutes uncontended on the benchmark GPU. Budgets are configurable in `config.py`.
- Record the stop reason: `budget`, `time`, `diverged` (NaN/inf), `killed`, `error`.
- LR schedules and warmup are defined over the **FLOPs budget**, so every config gets a complete schedule.

### 6.3 Persistence and restarts
- SQLite in WAL mode: `submissions`, `jobs`, `job_metrics`, `checkpoints`, `races`, `admin_events`.
- On start (including after the box was powered down), interrupted jobs return to the queue and resume from checkpoint. **A clean shutdown and a crash must be handled the same way.**
- Finished models go to `models/<submission_id>/` (section 8.4).

### 6.4 Device and precision
- Auto-detect the device; TF32 allowed; fp32 for evaluation. Handle OOM by shrinking chunk sizes; never crash the worker.
- A startup self-check prints device/GPU name, VRAM, which datasets loaded and their sizes, and runs a one-step training smoke test. On a CPU-only box it must still pass using tiny fallback sizes.

---

## 7. API contract

All JSON. Base `http://localhost:8000`. Admin routes need header `X-Admin-Token`. Errors: `{ "error": "code", "message": "..." }` with a proper HTTP status.

### Public
- `GET /api/health` → `{ ok, device, gpu_name, vram_gb, queue_length, demo_mode, paused, version }`
- `POST /api/demo/races` body `{ kind: "loss"|"lr"|"batch", lanes: [{ id, config }], max_seconds }` → `{ race_id }`
- `GET /api/demo/races/{race_id}/stream` (SSE). Events:
  - `tick`: `{ t, lanes: [{ id, step, samples_seen, updates, loss, acc, probe_preds: [16 ints] }] }` (~every 0.5 s)
  - `done`: `{ reason: "time"|"complete"|"aborted", final: [...] }`
  - `error`: `{ message }`
- `POST /api/demo/races/{race_id}/abort` → `{ ok }`
- `POST /api/submissions/preview` body: config → `{ valid, errors[], param_count, tier, search_depth_full_moves, est_flops_budget }`
- `POST /api/submissions` body `{ nickname, model_name, contact?, consent, config }` → `{ submission_id, participant_code, param_count, tier, status:"queued", queue_position }`
- `GET /api/submissions/{id}` → `{ status, queue_position?, progress?, stop_reason?, curves? }`
- `GET /api/leaderboard` → reserved; returns `[]`. No ranking logic on the server.

### Development
- `GET /api/dev/hello-stream?limit=&interval=` (SSE) → `tick` events `{ count, server_time }` once per `interval` seconds (default 1), then `done: { count }` if `limit` is set. For testing the tunnel.

### Admin
- `GET /admin/queue`; `POST /admin/jobs/{id}/kill`; `DELETE /admin/jobs/{id}`; `POST /admin/jobs/{id}/redo`
- `POST /admin/queue/pause`, `POST /admin/queue/resume`; `POST /admin/demo-mode` `{ enabled }`
- `POST /admin/reset` `{ confirm: "RESET" }` → snapshots the DB, then clears queue, results and stored runs
- `GET /admin/export`; `GET /admin/stream` (SSE live monitor); `GET /admin/logs?tail=200`

Define Pydantic models for every request and response and generate `shared/openapi.json` so frontend types come from one source.

---

## 8. Competition model spec (chess evaluation network)

### 8.1 Role
The net is **only a position evaluator**: encoded position in, a score for the side to move out. It never predicts moves. A fixed engine (section 10) does legal-move generation, search and game-over detection. **Pure net, no material term.** A bad net loses on its own merits.

### 8.2 Data use
Training reads the section 4.3 arrays: rows with `is_val == 0` for training, `is_val == 1` for the validation loss shown to participants. Convert `cp` (White's view) to a target for the side to move: with `perspective_flip` on, negate for Black to move. Convert to win probability with `p = sigmoid(cp / K)`; mates use `mate_clip` (a substitute centipawn value, sign from `mate`).

### 8.3 Architecture and config (all range-checked server-side)
MLP only. Total parameters, including the first layer, **≤ 30,000,000**. Choose `layers` and one `width`, with an optional per-layer width list that overrides them. The server computes parameter counts and rejects configs over the cap.

**Core knobs:** `layers`, `width` (or `layer_widths[]`); `activation` (the demo's functions with their parameters); `loss` (mse, huber with δ, bce on win probability, l1); `lr` (log scale); `batch_size`; `optimizer` (sgd, momentum, adam).

**Advanced knobs (defaults apply if untouched):**
- `target_type`: win probability (default) or clipped centipawns; `eval_squash_scale` (K); `mate_clip`
- `input_extras`: side to move and castling flags (**on by default**), en passant, material counts, simple attack maps (computed on the GPU from the board)
- `perspective_flip` (on/off); `color_flip_augmentation` (an exact chess symmetry: mirror ranks, swap colors, flip side, negate target)
- `data_slice`: `all` (default), `endgame` (`npc <= 10`), `balanced` (no mate and |cp| ≤ 100), `decisive` (mate or |cp| ≥ 300)
- `sampling`: uniform, or weighted toward larger |cp|
- `init`: xavier, he, small-normal
- `lr_schedule`: constant, cosine, linear decay; `warmup_frac`
- `grad_clip`; `ema` (on/off, decay)
- `normalization`: none, layernorm; `residual`: on/off
- `output_head`: linear, tanh, sigmoid
- `seed`

**Removed on purpose, do not add:** weight decay, dropout (no clear effect at this scale), game-result blending and rating-range filters (the data has neither).

Out-of-scope knobs take silent defaults. `DEFAULT_CONFIG` is a sensible but unremarkable config that yields a working, mediocre bot. **The House Net is not the default.**

### 8.4 Saved model format
`models/<submission_id>/`:
- `weights.safetensors` (fp32)
- `config.json` (full resolved config, param count, tier, search depth)
- `meta.json` (`stop_reason`, `flops_used`, `active_gpu_seconds`, `samples_seen`, `preemptions`, `val_loss`, timestamps)
- `curves.json` (train/val loss over time)

This is the **only interface** between training and the tournament. Document it in `shared/MODEL_FORMAT.md` with one loader used by both.

### 8.5 Tiers (search depth by size)
Depth is in **full moves** (one white move and one black reply); internal plies = 2 × full moves.

| Tier | Params | Search depth |
|---|---|---|
| Light | small | 3 full moves (6 plies) |
| Medium | middle | 2 full moves (4 plies) |
| Heavy | up to 30M | 1 full move (2 plies) |

Thresholds are **placeholders** in `tiers.json`; the benchmark sets them so each tier's cost per game is reasonable. If depth 3 is too slow even for light nets, shift every tier down (or cap at 4 plies). Mixed-depth games are normal. The House Net follows the same rules, budget and cap.

### 8.6 Default vs House Net
`DEFAULT_CONFIG` is separate from the House Net. The House Net is the strongest config the owner can find within the rules, trained by `train_house_net.py` with the **same pipeline and budget**, entered as "House Net". Beating it earns an extra chocolate.

---

## 9. Caching, fallbacks and recording

- `scripts/record_races.py` runs a **grid** of race configs on the real GPU and saves each SSE stream as JSON to `frontend/public/cache/races/<key>.json` (each loss option, a handful of learning rates, a handful of batch sizes).
- The frontend replays at original speed. For an uncached config, `S` replays the **nearest cached** one.
- Showcase models (doodle, raw-audio, log-mel audio) are trained once and exported by `export_weights.py` to `frontend/public/models/` with a manifest. Browser inference uses a small TS forward pass. **[verify]** inference speed.
- Act 3 steps 2-3 curves are precomputed JSON.
- **Replay mode** when the backend is unreachable: automatic, via `/api/health` (section 5.1).
- Fallback matrix: Acts 0, 1, 4 local; Act 2 forward pass bundled; Act 2 races cached; Act 3 mic → pre-recorded clips; Act 3 metrics and overfitting cached by design.

---

## 10. Tournament (separate CLI, manual, never auto-run)

Lives in `tournament/`. Run by the owner over SSH, e.g. `python -m tournament.run --rounds 6 --out results/`. **The API server never imports or starts it.**

### Engine
- `python-chess` for legal moves and all rules (check, checkmate, stalemate, repetition, 50-move, insufficient material).
- Alpha-beta with fixed move ordering. Depth from the model's tier. **No capture extension** unless the benchmark shows horizon blunders.
- **Deterministic:** exact ties break alphabetically by UCI string. Leaf evaluations run in **fixed-size fp32 batches**. No randomness anywhere.
- Guards: NaN/inf eval forfeits that game; games capped at ~150 plies and adjudicated by material. Optional per-move node cap if a tier is too slow **[verify]**.
- The tournament must convert a board into the model's input exactly as training does (same encoder code, imported from one place).

### Format
- **Swiss rounds with guaranteed games:** each round every model plays its opponent **twice from the same opening with colors swapped**; after N rounds everyone has exactly 2N games. No repeat pairings. A fixed suite of ~6-8 openings supplies variety. With an odd count, the unpaired model plays a reference bot.
- Reference bots (anchors): random mover, material-only evaluator, optionally shallow Stockfish.
- **Sanity gate:** before the tournament each model plays one quick game against the random mover; failures are listed and excluded unless overridden.
- **Safety:** refuses to run while the training queue is non-empty unless `--force`.
- Ranking: Bradley-Terry or Elo over all results, draws count half, bots as anchors. Outputs `ratings.json`, PGNs and a CSV.
- Finale: top-8 bracket precomputed so playback never waits (the frontend playback is a later task).
- Expected scale at ~35 entrants and 5-6 rounds: a few hundred games. **[verify]** games/hour per tier.

---

## 11. Benchmarks (the owner runs them on the GPU box and pastes the report back)

`scripts/benchmark.py` prints one report:
1. Device name, VRAM; throughput (samples/s) for tiny, median and ~30M-parameter MLPs at several batch sizes, including batch 1 and full-batch with accumulation.
2. Doodles: linear-only vs ReLU accuracy on the 10 classes; per-class confusion matrix (to swap too-easy or too-hard pairs).
3. Race drama: for each loss, learning rate and batch size option, accuracy vs time over the planned run length; flag configs whose curves do not separate or that finish within seconds.
4. Audio: raw-waveform vs log-mel accuracy.
5. Chess: search speed (nodes/s, seconds/move, estimated games/hour) per tier including a ~30M net with batched leaves; training throughput; default-config strength vs the random bot and the material baseline.
6. Calibrated `BUDGET_FLOPS` and time cap for the median config.
7. Browser inference time for each bundled model (a small HTML page or Node script).

The report states which **[verify]** items passed or need a design change.

---

## 12. Build phases (each ends with a "done when" check)

| # | Goal | Done when |
|---|---|---|
| 1 | Repo, backend skeleton, `fake_data.py`, `/api/health`, SSE hello, SQLite | Frontend button shows streamed numbers from the box through the tunnel |
| 2 | Loaders for the section 4 formats, rasterizer + audio features with golden tests on `data/samples/` | Golden tests pass in Python and TS; loaders pass on fake data, samples and real data |
| 3 | Training core: model builder, losses, optimizers, schedules, FLOPs accounting, checkpoints, race runner | CPU smoke tests pass; `benchmark.py` runs on GPU |
| 4 | Frontend shell: stage machine, gauge, keys, presenter page, offline behavior, Acts 0 and 1 | Linear collapse and ReLU bend both look right |
| 5 | Act 2 against the race API, cache pipeline, replay mode | Races finish in time; cutting the network falls back cleanly |
| 6 | Submission form, queue, scheduler with demo preemption, admin page | A demo race interrupts a competition job; kill, remove, redo, reset work with confirmations |
| 7 | Showcase models trained and bundled; forward-pass hero visual | Forward pass works offline in the browser |
| 8 | House Net and default config | Default is "meh", House Net is clearly better |
| 9 | Act 3, Act 4, 3D valley | Each act is independently skippable |
| 10 | Tournament CLI | Refuses to run on a busy queue; repeat runs give identical games |
| 11 | Mock demo and failure drills: network drop, server power-off and on, bad race config, mic failure | Owner approves; tag `demo-ready` |

Cut order if time runs short (the owner decides): live mic, 3D valley, Act 3 live parts, tournament frontend playback.

---

## 13. Security and hygiene

- Admin token from an env var; never committed; `.env` is gitignored.
- No public port; access is via the SSH tunnel; the backend binds `127.0.0.1`.
- Submissions are validated server-side against the schema and ranges (the owner handles any extra hygiene, such as profanity filtering).
- Do not store audio. Contact info is excluded from any exported or published dataset.
- Post-event: an anonymized export (configs, curves, final metrics) may go to a private Hugging Face repo; the owner decides about public release.

---

## 14. Credits (footer text, tiny)

- Quick, Draw! dataset, Google, CC BY 4.0, with a link to the dataset page.
- Speech Commands v0.02, Google, CC BY 4.0, with a link.
- Lichess open database of evaluated positions (CC0), with a link and a thank-you to Lichess.
