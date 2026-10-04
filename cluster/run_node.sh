#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
CFG=${1:-study/study.yaml}
python study/run.py "$CFG" plan
python study/run.py "$CFG" all --jobs "${JOBS:-$(nproc)}" --threads "${THREADS:-1}" ${GPUS:+--gpus "$GPUS"}
