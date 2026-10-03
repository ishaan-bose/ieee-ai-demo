# Build Your Own AI: IEEE CS @ PESU demo

`SPEC.md` is the single source of truth. Assumptions, decisions and open issues are in `NOTES.md`.
**`FINAL_REPORT.md` says, feature by feature, what is done, what was tested, and what only you can test on the real server.**

```
Laptop (frontend, localhost:5173)  --SSH tunnel-->  server (backend, 127.0.0.1:8000, GPU)
```

Every command below is copy-paste. Lines starting with `#` are comments.
"Server" is the rented box (`ssh e2e`). "Laptop" is the computer that runs the website.

**Try the whole demo with no server at all:** on the laptop run the frontend (section 5) and open
`http://localhost:5173/?mock=1`. Every stage works against a built-in mock server, and the **O** key switches the pretend
server off and on so you can watch the offline behaviour.

---

## 1. Server setup (once)

```bash
ssh e2e

# get the code. If the repo is private, git asks for your GitHub username and a personal access token.
cd ~
git clone https://github.com/ishaan-bose/ieee-ai-demo.git
cd ~/ieee-ai-demo
git checkout claude/upbeat-gauss-3pb7rg     # until this branch is merged into main; then use: git checkout main && git pull

# 1a. note the torch and numpy versions BEFORE installing anything
python3 -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# 1b. dry run: the output must NOT mention torch or numpy
pip install --dry-run -r backend/requirements.txt -r tournament/requirements.txt

# 1c. install for real (no virtualenv on this server, on purpose; see SPEC 0.1)
pip install -r backend/requirements.txt -r tournament/requirements.txt

# 1d. check torch and numpy did not change (compare with 1a)
python3 -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# 1e. settings file
cp backend/.env.example backend/.env
nano backend/.env        # set ADMIN_TOKEN to a long random string; Ctrl+O, Enter, Ctrl+X to save
```

If step 1b wants to install or upgrade **torch or numpy**, stop and tell Claude. Do not run 1c.
New packages in this version: `safetensors` (model files) and `python-chess` (tournament). Neither touches torch.

Update later with `cd ~/ieee-ai-demo && git pull`.

## 2. Run everything that prepares the event (server, once, in tmux)

One script runs every stage in order and **stops cleanly on the first failure**. Each stage is idempotent (finished work
is skipped), so after fixing a problem you run the same command again.

```bash
tmux new -s prep                      # long jobs live in tmux (Ctrl+B then D to leave, tmux attach -t prep to return)
cd ~/ieee-ai-demo/backend
./scripts/run_on_server.sh --quick    # REHEARSAL with tiny sizes (minutes): proves every stage works on this box
./scripts/run_on_server.sh            # the real thing (roughly 1-2 hours on the GPU box)
```

Stages, in order: `deps` (refuses to continue if pip wants to touch torch/numpy), `selfcheck`, `contract` (checks the real
data files against SPEC 4), `tests`, `rasterize` (1.09M doodles to image tensors), `benchmark` (sections 1-6), `showcase` (the three
bundled models + Act 3 curves), `races` (the recorded race grid), `export` (browser weights), `browserbench` (browser inference
timing; if the server has no Node it tells you to run `npm run bench:inference` on the laptop), `house` (House Net + default
baseline), `pack` (one `artifacts.tgz`).

Useful options: `--from showcase` (resume), `--only benchmark`, `--skip tests --skip contract`.
At the end it prints one **"PASTE THIS BACK"** block. Logs are in `~/demo/logs/`. The full benchmark report is
`~/demo/logs/benchmark_report.txt` (paste that separately).

The stages can also be run one by one: `python3 scripts/rasterize_quickdraw.py`, `scripts/benchmark.py`,
`scripts/train_showcase.py`, `scripts/record_races.py`, `scripts/export_weights.py`, `scripts/train_house_net.py`
(all from `backend/`, all accept `--help`; most accept `--quick` and `--force`).

**After the benchmark** (it writes `~/demo/state/calibration.json` with `BUDGET_FLOPS` and `TIME_CAP_SECONDS` for the median
config, about 9 minutes) restart the backend so it picks them up. It also writes `~/demo/state/tiers.suggested.json`:
look at it, and copy the depths into `shared/tiers.json` if you agree (SPEC 8.5).

**Copy the results to the laptop** (browser models and cached races are generated on the server):

```bash
# on the LAPTOP, from the repo folder:
scp e2e:demo/artifacts.tgz . && tar xzf artifacts.tgz -C frontend/public
```

## 3. Start the backend (server)

**Quick test, in the foreground** (stop it with Ctrl+C):

```bash
cd ~/ieee-ai-demo/backend
python3 -m app.main
```

It prints a **SELF-CHECK** block (device, GPU, VRAM, which data files exist, a one-step training smoke test) and then
`Uvicorn running on http://127.0.0.1:8000`. From a second SSH window:

```bash
curl http://127.0.0.1:8000/api/health
```

**Normal use, in tmux** (keeps running after you log out):

```bash
~/ieee-ai-demo/backend/start_backend.sh     # starts it (does nothing if already running)
tmux attach -t backend                      # watch it; detach with Ctrl+B then D
tmux send-keys -t backend C-c               # stop it
tail -f ~/demo/logs/backend.log             # or just read the log
```

After the server is powered off and on, run `start_backend.sh` again. Interrupted training jobs go back to the queue by
themselves and resume from their last checkpoint (see NOTES.md about starting it automatically at boot).

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `~/demo/data` | the datasets (SPEC section 4) |
| `STATE_DIR` | `~/demo/state` | database, models, checkpoints, calibration |
| `LOG_DIR` | `~/demo/logs` | backend and script logs |
| `NUM_WORKERS` | `4` | worker count. Never taken from `nproc`, which lies on this box |
| `ADMIN_TOKEN` | none | needed for `/admin` (admin is disabled without it) |
| `BUDGET_FLOPS`, `TIME_CAP_SECONDS` | calibration file, else `1e16`, `1620` | training budget per competition model |
| `CHECKPOINT_SECONDS`, `METRICS_SECONDS`, `VAL_ROWS`, `RACE_MAX_SECONDS` | `10`, `2`, `50000`, `120` | see `backend/.env.example` |
| `SELF_CHECK` | `1` | `0` skips the startup self-check |

The backend always binds `127.0.0.1:8000`: not reachable from the internet, only through the SSH tunnel.

## 4. The SSH tunnel (laptop)

**Once:** make sure `~/.ssh/config` on the laptop has keep-alives for `e2e` (keep your existing `HostName`, `User`, ...):

```
Host e2e
    ServerAliveInterval 15
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

The reconnect loop needs SSH **key** login. Test with `ssh e2e echo ok`.

**Every time**, in a terminal you leave open (it reconnects by itself when the link drops):

```bash
# Mac / Linux
while true; do ssh -N -L 8000:localhost:8000 e2e; echo "tunnel dropped, reconnecting in 3 s..."; sleep 3; done
```
```powershell
# Windows PowerShell
while ($true) { ssh -N -L 8000:localhost:8000 e2e; Write-Host "tunnel dropped, reconnecting in 3 s..."; Start-Sleep 3 }
```

Check it from another laptop terminal: `curl http://localhost:8000/api/health` (Windows: `curl.exe`).

## 5. Start the frontend (laptop)

Needs Node.js **22 LTS** (or 20.19+) from https://nodejs.org (`node --version`).

```bash
git clone https://github.com/ishaan-bose/ieee-ai-demo.git     # first time only
cd ieee-ai-demo && git checkout claude/upbeat-gauss-3pb7rg    # until merged into main
cd frontend
npm install                                                    # first time, and after package.json changes
npm run dev
```

Open **http://localhost:5173**. The dev server forwards `/api` and the admin API (`/admin/...`) to `http://127.0.0.1:8000`,
which is the tunnel. (`/admin` itself, opened in the browser, is the admin *page*.)

| Page | What |
|---|---|
| `/` | the demo (projected). Add `?mock=1` to run with no server |
| `/presenter` | the talk track for the current stage, for your phone |
| `/build` | the participant form (the booth, on your laptop) |
| `/admin` | admin panel: enter the `ADMIN_TOKEN` once. Not linked from anywhere |

**Demo keys:** `→` next · `←` back · `Space` train/run · `S` skip to the recorded result · `R` reset stage ·
`G` jump to any stage · `F` force recorded playback for this stage · `P` show the tiny "offline" marker on the big screen ·
`L` slow-mo for Act 1 training · `T` hold to speak (Act 3) · `,` `.` previous/next sample clip · `1`-`4` answer the duel.

**Phone as presenter notes:** run `FRONTEND_HOST=0.0.0.0 npm run dev`, then open `http://<laptop-ip>:5173/presenter` on the
phone (same Wi-Fi). The phone follows whichever stage the laptop is on. This exposes the dev server on your Wi-Fi: use it on a
trusted network and switch it back to plain `npm run dev` afterwards. In the same browser, a second tab on `/presenter` works too.

**Booth QR code:** the finale shows a QR code for `http://localhost:5173/build` by default. A phone cannot open `localhost`, so build
the real address in: `VITE_BOOTH_URL=http://<laptop-ip>:5173/build npm run dev`.

## 6. Tests

```bash
# backend (server or anywhere with Python 3.11+; fake data and the real samples in data/samples/)
cd ~/ieee-ai-demo/backend && pip install -r requirements-dev.txt && python3 -m pytest          # ~3 minutes; a few real-data tests are skipped
REAL_DATA=1 python3 -m pytest tests/test_real_data.py -v                                        # on the server: the REAL data (a few minutes)
python3 -m app.data.contract --real --max-records 20000                                         # the same checks as a readable report + statistics

# tournament
cd ~/ieee-ai-demo && python3 -m pytest tournament/tests -p no:warnings                          # ~2 minutes

# frontend (laptop)
cd frontend && npm test                                  # unit tests: rasterizer + audio golden tests, MLP, inference vs PyTorch, config
npm run build                                            # type check + production build
npm run dev                                              # (in another terminal) then:
npx playwright install chromium                          # once
npm run e2e                                              # 17 UI checks against the mock server, no backend needed
npm run bench:inference                                  # browser inference time of the bundled models (SPEC 11 item 7)
```

## 7. The tournament (server, by hand, after the event)

```bash
cd ~/ieee-ai-demo
python3 -m tournament.run --rounds 6 --out results/                # refuses to run while the training queue is not empty (--force overrides)
python3 -m tournament.run --rounds 6 --out results/ --extra-model ~/demo/state/baselines/default-config   # include the default config as an anchor
python3 -m tournament.run --help                                   # --stockfish PATH, --depth-cap-plies 4, --node-cap N, --override-sanity, ...
```

Each model plays each opponent twice from the same opening with colours swapped, so after N rounds everyone has exactly
2N games. Outputs in `results/`: `ratings.json`, `standings.csv`, `games.pgn`, `results.json`, `sanity.json`,
`bracket.json` (top-8 playoff, all moves precomputed), `run_config.json`. Repeat runs give byte-identical games.

## 8. Event-day checklist

1. Server: `~/ieee-ai-demo/backend/start_backend.sh`, then `curl http://127.0.0.1:8000/api/health`.
2. Laptop: tunnel loop (section 4), `npm run dev` (section 5). Open `/` and `/admin`. Switch **Demo Mode ON** in the admin page
   before the talk so races get the GPU; switch it off afterwards so competition jobs train.
3. Phone: `/presenter`.
4. Rehearse once: `?mock=1` shows every stage without the server; then walk the real thing, including pulling the tunnel
   (the race must switch to "recorded" by itself).
5. After the event: wait until the admin queue is empty, then run the tournament (section 7).

## Repo layout

```
SPEC.md  README.md  NOTES.md  FINAL_REPORT.md
data_prep/       the owner's prep scripts and logs. Reference only: not run, not edited.
data/samples/    small real samples of the three datasets (SPEC 4.5)
shared/          RASTERIZER.md, AUDIO_FEATURES.md, MODEL_FORMAT.md, tiers.json, openapi.json, config_defaults.json, golden/ fixtures
backend/
  app/           main.py, config.py, db.py, store.py, schemas.py, device.py, selfcheck.py
    api/         health, dev, demo (races), submissions, admin, presenter
    scheduler/   worker.py: one worker, FCFS, demo races preempt competition jobs
    training/    trainer.py, race.py, chess_data.py, losses.py, optim.py, classifier.py, audio_models.py, race_grid.py
    chess_net/   encoder.py (THE shared position encoder), model.py, config.py, model_io.py (THE shared loader)
    data/        contract.py (SPEC 4 as code), loaders.py, rasterizer.py, audio_features.py
  scripts/       run_on_server.sh and the stage scripts (rasterize, benchmark, showcase, races, export, house net)
  tests/         fake_data.py and the test suite
tournament/      engine/ (search, evaluators, play, openings), swiss.py, ratings.py, run.py (CLI), tests/
frontend/        Vite + React + TypeScript + Tailwind; src/acts, src/pages, src/lib, src/workers; e2e/; scripts/
```
