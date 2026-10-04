#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
CFG=${1:-study/study.yaml}
GPU_OPTS=${GPU_OPTS:---gres=gpu:1}
mkdir -p slurm_logs
dep=""
for stage in calib data manifest train eval summary; do
  n=$(python study/run.py "$CFG" count "$stage")
  opts=""
  case $stage in train|eval) opts="$GPU_OPTS --mem=48G";; esac
  id=$(sbatch --parsable $opts ${dep:+--dependency=afterok:$dep} --array=0-$((n - 1)) \
       --job-name="regrid_$stage" --export=ALL,STAGE=$stage,CFG=$CFG cluster/slurm_task.sh)
  echo "$stage: $n tasks, job $id"
  dep=$id
done
