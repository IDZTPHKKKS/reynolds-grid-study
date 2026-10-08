#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
CFG=${1:-study/study.yaml}
CORES=${CORES:-24}
THREADS=${THREADS:-4}
JOBS=$(( CORES / THREADS > 0 ? CORES / THREADS : 1 ))
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
if [ -n "${GPUS:-}" ]; then export SOLVER_DEVICE=${SOLVER_DEVICE:-cuda}; fi
run() { python study/run.py "$CFG" stage "$@"; }
python study/run.py "$CFG" plan
echo "using $CORES cores: $JOBS tasks x $THREADS threads${GPUS:+, GPUs $GPUS}, solver on ${SOLVER_DEVICE:-cpu}"
run calib --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"}
run data --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"}
run manifest
run train --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"} ${GPU_JOBS:+--gpu-jobs "$GPU_JOBS"}
run eval --jobs "$JOBS" --threads "$THREADS" ${GPUS:+--gpus "$GPUS"} ${GPU_JOBS:+--gpu-jobs "$GPU_JOBS"}
run summary
