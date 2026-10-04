#!/usr/bin/env bash
#PBS -N regrid
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=48:00:00
#PBS -j oe
set -euo pipefail
cd "$PBS_O_WORKDIR"
[ -f .venv/bin/activate ] && . .venv/bin/activate
export OMP_NUM_THREADS=4 FFT_WORKERS=4 TORCH_NUM_THREADS=4
python study/run.py "$CFG" task "$STAGE" "${PBS_ARRAY_INDEX:-0}"
