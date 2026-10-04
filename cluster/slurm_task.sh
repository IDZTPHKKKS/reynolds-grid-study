#!/usr/bin/env bash
#SBATCH --job-name=regrid
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=48:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f .venv/bin/activate ] && . .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK FFT_WORKERS=$SLURM_CPUS_PER_TASK TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
python study/run.py "$CFG" task "$STAGE" "${SLURM_ARRAY_TASK_ID:-0}"
