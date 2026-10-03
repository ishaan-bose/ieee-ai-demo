# Build Your Own AI — IEEE CS @ PESU demo

`SPEC.md` is the single source of truth. Assumptions and open issues are in `NOTES.md`.

**Status: phase 1** (repo skeleton, backend skeleton, fake data, `/api/health`, SSE hello stream,
SQLite schema, minimal frontend). Phase 1 is done when the frontend on your laptop shows numbers
streamed from the server through the SSH tunnel.

```
Laptop (frontend, localhost:5173)  --SSH tunnel-->  server (backend, 127.0.0.1:8000)
```

Every command below is copy-paste. Lines starting with `#` are comments.
"Server" means the rented box (`ssh e2e`). "Laptop" means the computer that runs the website.

---

## 1. Server setup (once)

```bash
ssh e2e

# get the code (first time only). If the repo is private, git asks for your GitHub
# username and a personal access token (not your password), or use the SSH URL
# git@github.com:ishaan-bose/ieee-ai-demo.git if this server has a GitHub SSH key.
cd ~
git clone https://github.com/ishaan-bose/ieee-ai-demo.git
cd ~/ieee-ai-demo
# the phase 1 code is on this branch until it is merged into main:
git checkout claude/upbeat-gauss-3pb7rg

# 1a. note the torch and numpy versions BEFORE installing anything
python3 -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# 1b. dry run: the output must NOT mention torch, and must not change numpy
pip install --dry-run -r backend/requirements.txt

# 1c. install for real (no virtualenv on this server, on purpose; see SPEC 0.1)
pip install -r backend/requirements.txt

# 1d. check torch and numpy did not change (compare with 1a)
python3 -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# 1e. settings file (optional; these are the defaults)
cp backend/.env.example backend/.env
nano backend/.env        # set ADMIN_TOKEN to a long random string; Ctrl+O, Enter, Ctrl+X to save
```

If step 1b wants to install or upgrade **torch or numpy**, stop and tell Claude. Don't run 1c.

To update the code later:

```bash
cd ~/ieee-ai-demo && git pull
```

## 2. Start the backend (server)

**Quick test, in the foreground** (stop it with Ctrl+C):

```bash
cd ~/ieee-ai-demo/backend
python3 -m app.main
```

At startup it prints a **SELF-CHECK** block: device, GPU name, VRAM, which dataset files it found
and their sizes, and a one-step training smoke test. It should end with `result : OK`.
Then you'll see `Uvicorn running on http://127.0.0.1:8000`.

From a second SSH window on the server you can check it:

```bash
curl http://127.0.0.1:8000/api/health
curl -N http://127.0.0.1:8000/api/dev/hello-stream      # a new tick every second; Ctrl+C to stop
```

**Normal use, in tmux** (keeps running after you log out):

```bash
~/ieee-ai-demo/backend/start_backend.sh     # starts it (does nothing if already running)
tmux attach -t backend                      # watch it; detach with Ctrl+B then D
tmux send-keys -t backend C-c               # stop it
tail -f ~/demo/backend.log                  # or just read the log
```

After the server is powered off and on, run `start_backend.sh` again (see NOTES.md about starting
it automatically at boot).

Settings come from environment variables (or `backend/.env`):

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `~/demo/data` | the datasets (SPEC section 4) |
| `STATE_DIR` | `~/demo/state` | SQLite database, later also models and checkpoints |
| `NUM_WORKERS` | `4` | worker/thread count. Never taken from `nproc`, which lies on this box |
| `ADMIN_TOKEN` | — | for `/admin` routes (from phase 6) |
| `SELF_CHECK` | `1` | set to `0` to skip the startup self-check |

The backend always binds `127.0.0.1:8000`, so it is not reachable from the internet. Only the SSH tunnel reaches it.

## 3. The SSH tunnel (laptop)

**Once:** make sure your laptop's `~/.ssh/config` has keep-alives for the `e2e` host. Open it
(Mac/Linux: `nano ~/.ssh/config`; Windows: `notepad $HOME\.ssh\config`) and make sure the `e2e`
entry has these three lines. Keep your existing `HostName`, `User`, `IdentityFile` and so on.

```
Host e2e
    ServerAliveInterval 15
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

The reconnect loop needs SSH **key** login (no password prompt). Test it with `ssh e2e echo ok`.

**Every time:** open a terminal on the laptop and leave it running. If the connection drops,
it reconnects by itself.

Mac / Linux (bash or zsh):

```bash
while true; do ssh -N -L 8000:localhost:8000 e2e; echo "tunnel dropped, reconnecting in 3 s..."; sleep 3; done
```

Windows (PowerShell):

```powershell
while ($true) { ssh -N -L 8000:localhost:8000 e2e; Write-Host "tunnel dropped, reconnecting in 3 s..."; Start-Sleep 3 }
```

Stop it with Ctrl+C (press it twice quickly to break out of the loop). Check it from another
laptop terminal: `curl http://localhost:8000/api/health` (in Windows PowerShell use `curl.exe`).

## 4. Start the frontend (laptop)

**Once:** install Node.js **22 LTS** (needs Node `20.19+` or `22.12+`) from https://nodejs.org.
Check the version with `node --version`.

```bash
git clone https://github.com/ishaan-bose/ieee-ai-demo.git     # first time only
cd ieee-ai-demo
git checkout claude/upbeat-gauss-3pb7rg                       # until merged into main
cd frontend
npm install                                                   # first time, and after package.json changes
npm run dev
```

Open **http://localhost:5173** in the browser:

- **Check /api/health** shows the server's device, GPU and queue state.
- **Live stream** counts up once per second, streamed from the server through the tunnel.
  If the tunnel or the backend goes down, it shows "reconnecting…" (orange) and resumes by itself.

The dev server forwards `/api` and `/admin` to `http://127.0.0.1:8000`, which is the tunnel.
To point it elsewhere: `BACKEND_URL=http://127.0.0.1:9000 npm run dev`.

## 5. Run the tests

On the server (or anywhere with Python 3.11+):

```bash
cd ~/ieee-ai-demo/backend
pip install -r requirements-dev.txt          # once: adds pytest and httpx (no torch)
python3 -m pytest                            # fake data + the real samples in data/samples/
```

Expected: everything passes, with 3 tests **skipped**. Those are the real-data checks, which you
opt into. Run them on the server, where the real data lives (takes a few minutes, safe on the
16 GB CPU plan):

```bash
cd ~/ieee-ai-demo/backend
REAL_DATA=1 python3 -m pytest tests/test_real_data.py -v
# or the same checks as a readable report:
python3 -m app.data.contract --real --max-records 20000
```

Other useful commands (from `backend/`):

```bash
python3 -m app.selfcheck                              # just the startup self-check
python3 -m tests.fake_data --out /tmp/fake --n 200    # write fake datasets (SPEC 4 formats)
DATA_DIR=/tmp/fake python3 -m app.main                # run the backend on fake data
python3 -m app.export_openapi                         # regenerate shared/openapi.json
```

Frontend type check and build (laptop, from `frontend/`): `npm run build`.

## 6. Benchmarks

Not yet: `backend/scripts/benchmark.py` comes in phase 3.

---

## Repo layout

```
SPEC.md  README.md  NOTES.md
data_prep/       owner's finished prep scripts + logs. Reference only; don't run or edit.
data/samples/    small real samples of the three datasets (SPEC 4.5)
shared/          openapi.json now; rasterizer/audio specs and golden fixtures from phase 2
backend/
  app/           main.py, config.py, db.py, device.py, selfcheck.py, sse.py
    api/         health.py, dev.py (routes)
    data/        contract.py: executable version of the SPEC 4 data contract
    scheduler/ training/   (later phases)
  scripts/       one-off scripts (later phases)
  tests/         fake_data.py + tests
tournament/      separate CLI (phase 10)
frontend/        Vite + React + TypeScript + Tailwind
```
