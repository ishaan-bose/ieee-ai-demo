# FINAL REPORT: phases 2 to 11 (one-shot build)

Branch: `claude/upbeat-gauss-3pb7rg` (restarted from `main` after the phase 1 merge; **nothing was pushed to `main`**).
Seven blocks, each committed and pushed when finished (`git log --oneline` shows them in order).

**In one paragraph.** Everything in SPEC sections 4 to 11 is built and tested here, on fake data, on the real sample files in
`data/samples/`, and in a real browser (Chromium). What is *not* proven is everything that needs the GPU, the 12M-row Lichess data,
the 1.09M doodles, a real microphone, your laptop, or your server's exact Python/torch install. Those are listed in section 3, and
section 4 gives the exact commands, in order, that turn each of them into a pasteable answer.

---

## 1. Feature status (Done / Partly / Not done)

Legend: **Done** = built and tested here. **Partly** = built, but something about it can only be finished/confirmed on the real box (noted).

### SPEC 0 and 0.1: rules and environment
| Item | Status | Notes |
|---|---|---|
| Never download datasets / no code that does | **Done** | nothing in the repo fetches data; `data_prep/` untouched |
| `fake_data.py` + golden tests on `data/samples/` | **Done** | all three datasets, exact SPEC 4 formats |
| Device-agnostic (cuda else cpu), sizes from VRAM | **Partly** | CPU paths fully tested; the CUDA paths (GPU-resident data, `mem_get_info` sizing, TF32, OOM micro-batching) are written but never ran on a GPU |
| `DATA_DIR` from env, `NUM_WORKERS` from env (never `os.cpu_count()`) | **Done** | tested (including a test that fakes 256 CPUs) |
| No torch in any requirements file; server's NVIDIA torch untouched | **Done** | `run_on_server.sh` also refuses if `pip --dry-run` wants torch/numpy |
| README with exact commands (setup, backend, tunnel, frontend, tests, benchmarks) | **Done** | `README.md` |
| Backend runs as a service at boot | **Partly** | `start_backend.sh` (tmux, idempotent) works; auto-start at boot needs to know your init system (NOTES.md) |
| Everything survives power-off/on | **Done** | WAL SQLite, jobs re-queue at the front and resume from checkpoint; tested at DB level and with a real preempt/resume |

### SPEC 3 and 4: architecture and data contract
| Item | Status | Notes |
|---|---|---|
| Vite proxy for `/api` and `/admin`, backend on 127.0.0.1 only | **Done** | `/admin` page vs API share a prefix: handled and tested |
| 4.1 Quick, Draw!: loaders, rasterizer spec (`shared/RASTERIZER.md`) in Python **and** TypeScript, golden fixtures | **Done** | max pixel difference 1 between the two implementations |
| `rasterize_quickdraw.py`: streaming, `NUM_WORKERS`, bounded memory, idempotent | **Partly** | works and is tested on fake data (ordered output, atomic writes). Speed on 1.09M real drawings is an estimate (~0.7 ms each) |
| 4.2 Speech: loaders, marvin ~2% subset, 300-clip overfit subset, `shared/AUDIO_FEATURES.md`, Python + TS + golden | **Done** | max difference < 1e-3 |
| 4.3 Lichess: loaders, data slices (endgame/balanced/decisive), mmap | **Done** | stored `mat` never used; the accepted real-data mismatches are statistics |
| Data contract as code (`python -m app.data.contract --real`) | **Done** | known imperfections reported as statistics, never failures |
| 4.4 Playground data generated in the browser | **Done** | clusters and spirals |

### SPEC 5: frontend
| Item | Status | Notes |
|---|---|---|
| 5.1 Stage machine, progressive disclosure, keys (→ ← Space S R G F), timer ring, accuracy gauge relative to chance, credits | **Done** | plus P, L, T, `,` `.` |
| Server offline is a normal state; races fall back to recordings; pending submissions upload on return; localStorage first | **Done** | tested with the mock server **and** by killing the real backend mid-race |
| 5.2 Routes `/`, `/presenter`, `/build`, `/admin` | **Done** | the presenter page follows the demo via BroadcastChannel or via the server (phone) |
| Act 0 duel | **Done** | |
| Act 1 (steps 1-6: line, knobs + least squares, spirals, stacked linear layers fuse, ReLU bends, activation gallery) | **Done** | Web Worker training; `hard_step` shows BROKE; `leaky_relu` at α=1 collapses; user-defined card removed as specified |
| Act 2.1 forward pass (glowing network, top-5 after every stroke) | **Partly** | works; until you export the trained doodle model it uses clearly-labelled random "demo weights" |
| Act 2.2 loss race, 2.3 backward pass + valley + learning-rate race, 2.4 batch race | **Done** | live (SSE) or recorded; the 3D valley is a react-three-fiber cartoon (2D fallback without WebGL) |
| Act 3.1 hearing (live mic push-to-talk, waveform → windows → FFT → mel → log, two models) | **Partly** | live mic tested only with Chromium's fake microphone; no-mic fallback to the 20 real clips tested; models are demo weights until exported |
| Act 3.2 metrics trap, 3.3 overfitting + size slider | **Done** | cached JSON, with demo data if the cache is missing |
| Act 4: tech tree, rematch, close (stats, QR, pitch) | **Done** | QR needs `VITE_BOOTH_URL` for a real address |
| Everything runs with no backend (`?mock=1`) | **Done** | complete mock server, `O` toggles it |
| 5.4 `/build` form (all 8.3 knobs, live param count and tier, tier table, consent, backup) | **Done** | parameter counts agree with Python for every tested config |
| 5.5 `/admin` (queue, per-job kill/remove/redo with confirm, pause/resume, Demo Mode, live monitor, nuclear reset with RESET + hold, export) | **Done** | GPU utilisation shows "n/a" without `pynvml` |

### SPEC 6 and 7: backend
| Item | Status | Notes |
|---|---|---|
| 6.1 One worker, FCFS, **demo race kills a running competition job, which returns to the front and resumes from its checkpoint** | **Done** | tested end to end: preempted at step 324, race ran, resumed from step 250, no double-counted samples/FLOPs |
| Checkpoints every ~10 s active time; races: up to 3 lanes in short slices; full-batch via gradient accumulation (identical math); Demo Mode | **Done** | accumulation tested equal to a true full batch |
| 6.2 FLOPs budget + active-GPU-time cap, stop reasons, schedules over FLOPs progress | **Done** | `BUDGET_FLOPS` still needs calibrating on the GPU (benchmark does it) |
| 6.3 SQLite WAL with the six tables, restart recovery, models folder | **Done** | DB migration v1 → v2 tested |
| 6.4 Device detection, startup self-check with smoke test | **Done** | |
| 7 Every endpoint, Pydantic models, errors as `{error, message}` | **Done** | plus documented extensions (config schema, presenter, idempotent `client_id`) in SPEC.md |
| `shared/openapi.json` so frontend types come from one source | **Partly** | the file is generated; the TypeScript types in `src/lib/types.ts` are hand-mirrored, not generated from it |

### SPEC 8 to 11
| Item | Status | Notes |
|---|---|---|
| 8.1-8.3 MLP evaluator with every core + advanced knob, ≤ 30M params, server-side range checks | **Done** | removed knobs (weight decay, dropout, ...) are ignored silently, as specified |
| 8.4 Saved model format + one shared loader + `shared/MODEL_FORMAT.md` | **Done** | |
| ONE shared position encoder for training and tournament | **Done** | verified against python-chess (attack maps, colour flip, material) |
| 8.5 Tiers | **Partly** | `shared/tiers.json` holds placeholders; the benchmark writes `tiers.suggested.json` |
| 8.6 Default config ≠ House Net; `train_house_net.py` (search + full budget + default baseline) | **Partly** | pipeline tested; the actual strong House Net needs a GPU run and your judgement |
| 9 Showcase models, `export_weights.py` + manifest, `record_races.py` grid, Act 3 curves, replay mode | **Partly** | all scripts tested on fake data; the real recordings/weights need your server run |
| 10 Tournament CLI (engine, Swiss with guaranteed games, Bradley-Terry, bots, sanity gate, refuse-if-busy, deterministic) | **Done** | repeat runs are byte-identical; bracket data precomputed. Frontend playback of the tournament is **not built** (SPEC's own first item to cut) |
| 11 `benchmark.py` sections 1-7 as one pasteable report | **Partly** | runs and is tested; only CPU numbers exist. Section 7 is a Node script (`npm run bench:inference`) |

### SPEC 12 to 14
| Item | Status | Notes |
|---|---|---|
| Phases 2-10 | **Done** | |
| Phase 11 mock demo and failure drills: network drop, bad race config, mic failure | **Done** | automated (UI walkthrough, real-backend cut test, mic-less browser) |
| Phase 11 server power-off and on | **Partly** | recovery logic is tested; the full drill with the GPU and a real power cycle is yours |
| Phase 11 owner approval, tag `demo-ready` | **Not done** | yours to do |
| 13 Security: admin token from env, `.env` ignored, no public port, server-side validation, no audio stored, contact excluded from exports | **Done** | `FRONTEND_HOST=0.0.0.0` (phone presenter) exposes the dev server on Wi-Fi: documented |
| 14 Credits footer | **Done** | |

---

## 2. What was tested and how

| Suite | Result | What it covers |
|---|---|---|
| Backend `pytest` (`cd backend && python3 -m pytest`) | **141 passed, 3 skipped (opt-in real-data tests)** | data contract on fake data **and** the real samples; rasterizer/audio golden; loaders; encoder vs python-chess; model/param counts vs torch; trainer (budget/time/divergence/abort/exact checkpoint resume); every knob end to end; races (fairness, batch 1, full-batch accumulation equals a true full batch, abort); scheduler + API integration (preemption and resume, kill/remove/redo, pause, Demo Mode, reset keeps the House Net and snapshots first, auth, export without contact, SSE streams, restart recovery, DB migration); every server script in quick mode; a real server process (binds 127.0.0.1, clean shutdown with an open stream) |
| Tournament `pytest` (`python3 -m pytest tournament/tests`) | **39 passed** | alpha-beta equals plain minimax; finds mates; alphabetical ties; node cap; NaN forfeits; Swiss properties for odd/even/large fields; circle-method fallback; Bradley-Terry recovery; exactly 2N games each; refuse-if-busy; sanity gate; byte-identical repeat runs; replayable bracket; the API never imports the tournament |
| Frontend unit (`npm test`) | **50 passed** | rasterizer + log-mel golden vs Python; TS forward pass vs PyTorch; the Act 1 MLP really learns (ReLU solves spirals, a deep linear stack cannot and collapses to one line, width 2 fails, hard step cannot learn); activation derivatives; config counts equal Python's |
| UI walkthrough (`npm run e2e`, mock server) | **18 checks passed** | duel, all Act 1 stages, gallery BROKE badge, drawing → network, live race, **server cut mid-race → recorded**, batch counters, mic (fake device) and **mic-less fallback**, trap, overfit, Act 4, G menu, presenter following, `/build` (live counts, validation, code, backup, **offline entry uploads by itself**), `/admin` (confirmations, hold-to-reset), 3D valley with WebGL, no console errors |
| Real frontend ↔ real backend (`e2e/real-backend-smoke.mjs`) | **5 of 5 passed** | health, live race streaming through the Vite proxy, form → participant code → status, admin login with the real token, wrong token rejected. Also killed the backend mid-race by hand: the page switched to the recorded stream and finished the race |
| `run_on_server.sh --quick` rehearsal on fake data | **all 10 stages OK** | deps, selfcheck, rasterize, benchmark, showcase, races, export, browser benchmark, House Net, pack |
| `npm run build` / `tsc` | clean | |

Bugs found *by* these tests and fixed: shutdown hang with an open SSE stream (phase 1); the page not noticing a dead backend through the
Vite proxy; two-schedulers-one-DB test mistake (test only); admin stream hang (test limit added); `markSent` overwriting the local
status with the server's `queued` (polling never started); flush race on reconnect; mock server re-seeding after reset;
python-chess dropping castling rights that the data contains (test adjusted); greedy Swiss pairing getting stuck near a full round
robin (replaced by a Dirac-safe rule + circle-method fallback).

## 3. What I could NOT test (needs the real GPU, real data, or a real browser/mic)

1. **Any CUDA code path**: GPU-resident datasets, VRAM-based chunk sizing, TF32 vs fp32 eval, the OOM micro-batching, `torch.cuda.synchronize` timing. All were exercised only on CPU.
2. **Real data at scale**: the exact real counts (`REAL_DATA=1` tests), `rasterize_quickdraw.py` on 1.09M drawings (time, memory), loading 12M Lichess rows (startup pause), mmap random-access speed on the CPU plan, the first competition job's load time.
3. **Training quality and timing**: doodle/audio accuracies, whether the race options "separate" (drama), the log-mel vs raw gap, chess strength of the default config vs the House Net, the calibrated `BUDGET_FLOPS`, search speed per tier on the GPU. The benchmark report answers each of these on your box.
4. **A real microphone and speech**: only Chromium's synthetic fake microphone. Real room noise, gain and your accent are untested.
5. **Browsers other than Chromium**, projector resolution/scaling (the layout is scaled from 1600x900), touch input, a Windows laptop (the PowerShell tunnel loop was written but not run).
6. **Your server's install**: Python 3.12.3 + NVIDIA torch 2.8 + `pip install safetensors python-chess` (the dry-run guard protects torch/numpy, but the install itself was only tried here), boot-time start, `pynvml` availability.
7. **The real SSH tunnel** and Wi-Fi drops (simulated with a dead/frozen backend and the mock outage).
8. **Load**: 35 to 60 people submitting at once, and a long-running queue over hours.
9. **Tournament at scale**: games/hour on the GPU, a 35-model event, optional Stockfish.

## 4. Exact commands, in order

Everything is in `README.md`; this is the short version. Paste the output blocks back where it says **PASTE**.

### On the server (`ssh e2e`)
```bash
# 0. get the code
cd ~ && git clone https://github.com/ishaan-bose/ieee-ai-demo.git 2>/dev/null; cd ~/ieee-ai-demo && git fetch && git checkout claude/upbeat-gauss-3pb7rg && git pull

# 1. install (the dry run must NOT mention torch or numpy)
python3 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
pip install --dry-run -r backend/requirements.txt -r tournament/requirements.txt
pip install -r backend/requirements.txt -r tournament/requirements.txt
python3 -c "import torch, numpy; print(torch.__version__, numpy.__version__)"        # unchanged?
cp backend/.env.example backend/.env && nano backend/.env                              # set ADMIN_TOKEN

# 2. rehearsal with tiny sizes, then the real run (inside tmux)
tmux new -s prep
cd ~/ieee-ai-demo/backend
./scripts/run_on_server.sh --quick          # PASTE the "PASTE THIS BACK" block
python3 -m app.data.contract --real --max-records 20000    # PASTE: PASS/FAIL per dataset + the Statistics lines
REAL_DATA=1 python3 -m pytest tests/test_real_data.py -v   # PASTE the last lines
./scripts/run_on_server.sh                  # PASTE the block, and: cat ~/demo/logs/benchmark_report.txt
cat ~/demo/state/calibration.json ~/demo/state/tiers.suggested.json      # PASTE (then edit shared/tiers.json if you agree)

# 3. run the backend
~/ieee-ai-demo/backend/start_backend.sh && sleep 8 && curl http://127.0.0.1:8000/api/health        # PASTE
```

### On the laptop
```bash
# 4. code + browser models
git clone https://github.com/ishaan-bose/ieee-ai-demo.git; cd ieee-ai-demo && git checkout claude/upbeat-gauss-3pb7rg
scp e2e:demo/artifacts.tgz . && tar xzf artifacts.tgz -C frontend/public
cd frontend && npm install && npm test && npm run bench:inference                          # PASTE the bench lines

# 5. rehearse with NO server, then for real (tunnel in one terminal, dev server in another)
npm run dev                                         # open http://localhost:5173/?mock=1  and click through every stage (G menu; O toggles the pretend server)
while true; do ssh -N -L 8000:localhost:8000 e2e; echo "dropped"; sleep 3; done      # Windows: see README section 4
npm run dev                                         # open http://localhost:5173/  /build  /admin  /presenter
```
Then the failure drills: start a race and pull the tunnel (the page must say "recorded" and finish), restart the backend with
`tmux send-keys -t backend C-c` + `start_backend.sh`, submit with the server down (the entry must appear as pending and upload by itself),
and in `/admin` turn Demo Mode on while a competition job is running and start a race (the job must yield, then resume).

### After the event
```bash
cd ~/ieee-ai-demo && python3 -m tournament.run --rounds 6 --out results/ --extra-model ~/demo/state/baselines/default-config
```

## 5. First things to look at when you paste results back
- `[verify]` lines at the end of `benchmark_report.txt` (each says PASS or NEEDS CHANGE and why).
- The **Statistics** lines of the contract report (the accepted imperfections: they should match what you saw).
- `default_over_house_val_mse` in the House Net summary (should be clearly above 1: the House Net must be clearly better than the default).
- `tiers.suggested.json` vs `shared/tiers.json`.
