# Reynolds x grid study

Forced 2D turbulence (Kolmogorov flow) on a table of Reynolds numbers and grids, and neural
surrogates (CNN, U-Net, FNO; each on its native grid and on a fixed 128² grid) trained on every
cell and tested on every other cell. Each pair is labelled by what changes between training and
test: Reynolds number only, grid only, or both.

Default table (`study/study.yaml`): Re_f = 10, 20, 40, 80, 160, 320, 640 on 64², 128², 256²,
512², keeping the cells each grid resolves (19 cells), 8 trajectories per cell (2 held out for
testing), 6 surrogates, 5 seeds.

## Setup

```bash
git clone https://github.com/IDZTPHKKKS/reynolds-grid-study.git
cd reynolds-grid-study
# module load python cuda      # if the cluster uses environment modules
bash cluster/setup.sh          # venv, packages, tests, 20 s smoke study
```

The repository is private: on the cluster use `gh auth login`, an SSH key added to GitHub, or a
personal access token as the password when cloning.

## Run

See the plan first:

```bash
python study/run.py study/study.yaml plan
```

One machine (all cores, optional GPUs). Use `tmux` or `nohup` so it survives logging out:

```bash
tmux new -s study
. .venv/bin/activate
JOBS=48 THREADS=1 GPUS=0,1 cluster/run_node.sh study/study.yaml
```

`JOBS` is the number of parallel tasks, `THREADS` the CPU threads per task (FFT and torch),
`GPUS` the GPU ids for training and evaluation (two tasks per GPU at a time; change with
`--gpu-jobs`). A good split for the solver is `JOBS x THREADS` = number of cores.

Single stages, for example to give the 512² data more threads:

```bash
python study/run.py study/study.yaml stage calib --jobs 7 --threads 4
python study/run.py study/study.yaml stage data --jobs 12 --threads 4
python study/run.py study/study.yaml stage manifest
python study/run.py study/study.yaml stage train --gpus 0,1,2,3
python study/run.py study/study.yaml stage eval --gpus 0,1,2,3
python study/run.py study/study.yaml stage summary
```

Batch schedulers, one array job per stage, each waiting for the previous one (check the
partition, queue and GPU options at the top of the scripts first):

```bash
GPU_OPTS="--partition=gpu --gres=gpu:1" bash cluster/slurm_submit.sh study/study.yaml   # SLURM
GPU_SELECT="select=1:ncpus=4:ngpus=1:mem=48gb" bash cluster/pbs_submit.sh study/study.yaml  # PBS
```

Every task skips work that is already on disk, so a stopped or failed run is resumed by running
the same command again. Logs are in `runs/<name>/logs/<stage>_<task>.log`.

## Output

`runs/full/` (or `$STUDY_OUT`):

| file | |
|---|---|
| `cells.csv` | per cell: measured Re_f, k_max/k_d (resolution), max CFL, trajectories |
| `summary.csv` | per surrogate, seed, train cell and test cell: one-step, mean rollout and final-step relative L2 error |
| `summary_mean.csv` | the same averaged over seeds, with standard deviations |
| `data/`, `checkpoints/`, `results/` | trajectories, trained models, raw evaluation files |

The `summary` stage also prints, per surrogate, the rollout error relative to a model trained on
the test cell, for Re-only, grid-only and combined changes. The CSV files are small; copy them
back with

```bash
rsync -av --include="*.csv" --exclude="*" user@cluster:reynolds-grid-study/runs/full/ results_full/
```

## Settings

`study/study.yaml`:

| key | |
|---|---|
| `reynolds`, `grids`, `max_re` | the table; a cell is run if Re <= `max_re[grid]` |
| `trajectories`, `test_trajectories` | trajectories per cell and how many are held out |
| `t_total`, `dt_sample` | length of each trajectory and the surrogate time step (same for all cells) |
| `dt_times_n` | solver time step x grid size (0.32 keeps the CFL number near 0.5) |
| `variants`, `seeds` | surrogates and training seeds |
| `train`, `eval` | training and rollout settings |
| `solver_overrides` | any key of the solver config (`configs/re_low.yaml` is the template) |

The forcing amplitude is calibrated once per Reynolds number (target u_rms = 1) on the
coarsest grid of the table that resolves it, and reused on the finer grids.

## Other entry points

```bash
python scripts/verify_solver.py                              # solver checks
python scripts/generate_dataset.py --config configs/re_high.yaml --outdir data
python burgers/run_e5.py                                     # 1D Burgers experiment
python -m pytest -q tests
```

Equations and conventions: `docs/equations.md`; solver checks: `docs/solver_validation.md`.
