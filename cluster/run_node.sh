#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
CFG=${1:-study/study.yaml}
CORES=${CORES:-24}
THREADS=${THREADS:-4}
JOBS=$(( CORES / THREADS > 0 ? CORES / THREADS : 1 ))
run() { python study/run.py "$CFG" stage "$@"; }
python study/run.py "$CFG" plan
echo "using $CORES cores: $JOBS tasks x $THREADS threads${GPUS:+, GPUs $GPUS}"
run calib --jobs "$JOBS" --threads "$THREADS"
run data --jobs "$JOBS" --threads "$THREADS"
run manifest
run train --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"}
run eval --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"}
run summary
