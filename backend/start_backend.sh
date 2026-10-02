#!/usr/bin/env bash
# Start the backend inside a tmux session called "backend" (does nothing if it is
# already running). Safe to run at any time, including right after a reboot.
#   attach to see logs:   tmux attach -t backend      (detach again: Ctrl+B then D)
#   stop:                 tmux send-keys -t backend C-c
set -euo pipefail
cd "$(dirname "$0")"
if tmux has-session -t backend 2>/dev/null; then
  echo "backend is already running (tmux attach -t backend to see it)"
  exit 0
fi
tmux new-session -d -s backend "python3 -m app.main 2>&1 | tee -a ~/demo/backend.log"
echo "backend started in tmux session 'backend'; log: ~/demo/backend.log"
