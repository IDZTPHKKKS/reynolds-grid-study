#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
CFG=${1:-study/study.yaml}
GPU_SELECT=${GPU_SELECT:-select=1:ncpus=4:ngpus=1:mem=48gb}
dep=""
for stage in calib data manifest train eval summary; do
  n=$(python study/run.py "$CFG" count "$stage")
  opts=""
  case $stage in train|eval) opts="-l $GPU_SELECT";; esac
  [ "$n" -gt 1 ] && opts="$opts -J 0-$((n - 1))"
  id=$(qsub $opts ${dep:+-W depend=afterok:$dep} -N "regrid_$stage" -v STAGE=$stage,CFG=$CFG cluster/pbs_task.sh)
  echo "$stage: $n tasks, job $id"
  dep=$id
done
