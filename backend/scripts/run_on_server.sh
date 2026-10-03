#!/usr/bin/env bash
# Run every server-side stage in order; stop cleanly on the FIRST failure; print one summary to paste back.
#
#   cd ~/ieee-ai-demo/backend
#   ./scripts/run_on_server.sh                 # everything (GPU box: roughly 1-2 hours; run it inside tmux!)
#   ./scripts/run_on_server.sh --quick         # tiny sizes: a rehearsal on the CPU plan (minutes)
#   ./scripts/run_on_server.sh --from showcase # resume from a stage (earlier stages are skipped)
#   ./scripts/run_on_server.sh --only benchmark
#   ./scripts/run_on_server.sh --skip tests --skip contract
#
# Stages, in order: deps selfcheck contract tests rasterize benchmark showcase races export house pack
# Every stage is idempotent (finished work is skipped), so after a failure fix the problem and run the same command again.
# Logs: ~/demo/logs/<stage>.log ; summary lines: ~/demo/logs/run_on_server_summary.txt
set -u -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

STAGES=(deps selfcheck contract tests rasterize benchmark showcase races export house pack)
QUICK=""; FROM=""; ONLY=""; SKIP=()
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) QUICK="--quick" ;;
    --from) FROM="$2"; shift ;;
    --only) ONLY="$2"; shift ;;
    --skip) SKIP+=("$2"); shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 2 ;;
  esac
  shift
done

LOG_DIR="${LOG_DIR:-$HOME/demo/logs}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/run_on_server_summary.txt"
: > "$SUMMARY"
T_START=$(date +%s)
STARTED=0; [ -z "$FROM" ] && STARTED=1

run_stage() {  # name, command...
  local name="$1"; shift
  local log="$LOG_DIR/stage_${name}.log"
  local t0=$(date +%s)
  echo; echo "===== STAGE: $name  ($(date +%H:%M:%S)) ====="
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local dt=$(( $(date +%s) - t0 ))
  grep -h '^SUMMARY' "$log" >> "$SUMMARY" 2>/dev/null
  if [ $rc -ne 0 ]; then
    echo "STAGE $name FAILED (exit $rc after ${dt}s). Full log: $log"
    echo "STAGE $name FAILED (exit $rc)" >> "$SUMMARY"
    print_summary
    echo "Fix the problem, then re-run: ./scripts/run_on_server.sh --from $name${QUICK:+ --quick}"
    exit 1
  fi
  echo "STAGE $name OK (${dt}s)"
  echo "STAGE $name OK ${dt}s" >> "$SUMMARY"
}

print_summary() {
  echo; echo "================= PASTE THIS BACK ================="
  echo "run_on_server $(date '+%F %T') quick=${QUICK:-no} host=$(hostname) total=$(( $(date +%s) - T_START ))s"
  python3 - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only")
PY
  cat "$SUMMARY"
  [ -f "$LOG_DIR/benchmark_report.txt" ] && echo "(full benchmark report: $LOG_DIR/benchmark_report.txt  -> paste that separately)"
  echo "==================================================="
}

stage_deps() {
  # Never let pip touch the NVIDIA torch (SPEC 0.1). Dry-run first and refuse if torch/numpy would change.
  local plan; plan=$(pip install --dry-run -r requirements.txt -r ../tournament/requirements.txt 2>&1)
  echo "$plan" | tail -5
  if echo "$plan" | grep -E "Would install" | grep -Eiq "(^|[ ])(torch|numpy)-[0-9]"; then
    echo "pip wants to install/replace torch or numpy: STOP. Flag this to Claude; do not install."; return 1
  fi
  pip install -r requirements.txt -r ../tournament/requirements.txt
}
stage_selfcheck() { python3 -m app.selfcheck; }
stage_contract() { python3 -m app.data.contract --real --max-records 20000; }
stage_tests() {
  python3 -c "import pytest" 2>/dev/null || pip install -r requirements-dev.txt
  python3 -m pytest -x -p no:warnings -o addopts="" -q
}
stage_rasterize() { python3 scripts/rasterize_quickdraw.py; }
stage_benchmark() { python3 scripts/benchmark.py $QUICK; }
stage_showcase() { python3 scripts/train_showcase.py $QUICK; }
stage_races() { python3 scripts/record_races.py $QUICK; }
stage_export() { python3 scripts/export_weights.py; }
stage_house() { python3 scripts/train_house_net.py $QUICK; }
stage_pack() {
  local out="${ARTIFACTS:-$HOME/demo/artifacts.tgz}"
  local root; root="$(cd .. && pwd)"
  (cd "$root/frontend/public" && tar czf "$out" models cache 2>/dev/null) || { echo "nothing to pack yet"; return 1; }
  ls -la "$out"
  echo "SUMMARY pack: {\"artifacts\": \"$out\", \"bytes\": $(stat -c %s "$out")}"
  echo "On your LAPTOP, from the repo folder:  scp e2e:demo/artifacts.tgz . && tar xzf artifacts.tgz -C frontend/public"
}

for st in "${STAGES[@]}"; do
  [ -n "$ONLY" ] && [ "$st" != "$ONLY" ] && continue
  [ -n "$FROM" ] && [ "$st" = "$FROM" ] && STARTED=1
  [ $STARTED -eq 0 ] && continue
  skip=0; for s in "${SKIP[@]:-}"; do [ "$s" = "$st" ] && skip=1; done
  [ $skip -eq 1 ] && { echo "skipping $st"; continue; }
  run_stage "$st" "stage_$st"
done
print_summary
