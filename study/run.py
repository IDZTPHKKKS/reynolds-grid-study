#!/usr/bin/env python3
"""Reynolds x grid study: data, training and evaluation split into independent tasks.

    python3 study/run.py study/study.yaml plan
    python3 study/run.py study/study.yaml all --jobs 32 --gpus 0,1
    python3 study/run.py study/study.yaml stage data --jobs 48
    python3 study/run.py study/study.yaml count train
    python3 study/run.py study/study.yaml task train 17
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "neural"))

STAGES = ["calib", "data", "manifest", "train", "eval", "summary"]


def load(path):
    c = yaml.safe_load(open(path))
    c["path"] = os.path.abspath(path)
    c["out"] = os.path.abspath(os.environ.get("STUDY_OUT", os.path.join(ROOT, "runs", c["name"])))
    return c


def cells(c):
    return [(f"re{re:g}_n{n}", float(re), int(n))
            for n in c["grids"] for re in c["reynolds"] if re <= c["max_re"][n]]


def paths(c):
    p = {k: os.path.join(c["out"], k) for k in ("configs", "calib", "data", "checkpoints", "results", "logs")}
    for d in p.values():
        os.makedirs(d, exist_ok=True)
    return p


def tasks(c, stage):
    n_cells = len(cells(c))
    if stage == "calib":
        return [(re,) for re in sorted({re for _, re, _ in cells(c)})]
    if stage == "data":
        return [(i, t) for i in range(n_cells) for t in range(c["trajectories"])]
    if stage in ("train", "eval"):
        order = sorted(range(n_cells), key=lambda i: -cells(c)[i][2])
        return [(v, i, s) for i in order for v in c["variants"] for s in c["seeds"]]
    return [()]


def cell_config(c, i):
    name, re, n = cells(c)[i]
    raw = yaml.safe_load(open(os.path.join(ROOT, "configs", "re_low.yaml")))
    raw["name"] = name
    raw["description"] = f"Kolmogorov flow, target Re_f = {re:g}, {n}^2 grid"
    raw["grid"] = {"Nx": n, "Ny": n}
    raw["forcing"]["amplitude"] = round(2.8 * re ** -0.367, 4)
    raw["viscosity"]["target_re_f"] = re
    raw["time"].update(dt=c["dt_times_n"] / n, t_total=c["t_total"], dt_sample=c["dt_sample"])
    raw["initial_condition"]["seed"] = 100000 * (i + 1)
    raw["output"].update(n_traj=c["trajectories"], seed_offset=50000)
    for key, sub in (c.get("solver_overrides") or {}).items():
        raw[key].update(sub)
    path = os.path.join(paths(c)["configs"], f"{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(raw, fh, sort_keys=False)
    return path


def split(c):
    n, k = c["trajectories"], c["test_trajectories"]
    return list(range(n - k)), list(range(n - k, n))


def calib_file(c, re):
    return os.path.join(paths(c)["calib"], f"re{re:g}.json")


def run_calib(c, re):
    from generate_dataset import calibrate_amplitude
    from solver.config import load_config
    out = calib_file(c, re)
    if os.path.exists(out):
        return
    i = min((k for k, (_, r, _) in enumerate(cells(c)) if r == re), key=lambda k: cells(c)[k][2])
    cfg = load_config(cell_config(c, i))
    method = "iteration"
    try:
        A, hist = calibrate_amplitude(cfg)
    except RuntimeError as e:
        hist = getattr(e, "history", None)
        if not hist:
            raise
        A, method = fit_amplitude(hist, cfg.calibration.target_u_rms), "fit"
        print(f"no convergence; amplitude from a power-law fit of {len(hist)} measurements: A = {A:.4f}", flush=True)
    with open(out, "w") as fh:
        json.dump({"re": re, "calibrated_on": cells(c)[i][0], "amplitude": A, "method": method,
                   "history": hist}, fh, indent=2)


def fit_amplitude(hist, target):
    a = np.log([h["A"] for h in hist])
    u = np.log([h["u_rms_measured"] for h in hist])
    if np.ptp(a) < 1e-6:
        return float(np.exp(a.mean()) * target / np.exp(u.mean()))
    slope, icept = np.polyfit(a, u, 1)
    if not 0.2 <= slope <= 2.0:
        slope = 0.75
        icept = u.mean() - slope * a.mean()
    return float(np.exp((np.log(target) - icept) / slope))


def run_data(c, i, t):
    from generate_dataset import generate_trajectory
    from solver.config import load_config
    p = paths(c)
    name, re, _ = cells(c)[i]
    if os.path.exists(os.path.join(p["data"], f"{name}_traj{t}.json")):
        return
    A = json.load(open(calib_file(c, re)))["amplitude"]
    generate_trajectory(load_config(cell_config(c, i)), A, t, p["data"])


def run_manifest(c):
    p = paths(c)
    metas = [json.load(open(f)) for f in glob.glob(os.path.join(p["data"], "*_traj*.json"))]
    metas.sort(key=lambda m: (m["name"], m["trajectory"]))
    with open(os.path.join(p["data"], "manifest.json"), "w") as fh:
        json.dump(metas, fh, indent=2)
    by = defaultdict(list)
    for m in metas:
        by[m["name"]].append(m)
    rows = []
    for name, re, n in cells(c):
        ms = by.get(name, [])
        rows.append({"cell": name, "re_target": re, "n": n, "trajectories": len(ms),
                     "re_f": np.mean([m["measured"]["Re_f"] for m in ms]) if ms else "",
                     "kmax_over_kd": np.mean([m["measured"]["k_max_over_k_d"] for m in ms]) if ms else "",
                     "max_courant": max([m["measured"]["max_courant"] for m in ms]) if ms else ""})
    write_csv(os.path.join(c["out"], "cells.csv"), rows)
    for r in rows:
        print(f"{r['cell']:>12}  trajectories {r['trajectories']}/{c['trajectories']}  "
              f"Re_f {fmt(r['re_f'])}  kmax/kd {fmt(r['kmax_over_kd'])}  CFL {fmt(r['max_courant'])}")


def run_train(c, v, i, s):
    from train import train_one
    p = paths(c)
    name, _, n = cells(c)[i]
    tag = f"_s{s}"
    if os.path.exists(os.path.join(p["checkpoints"], f"{v}_{name}{tag}_history.json")):
        return
    T = c["train"]
    accum = max(1, n // 64) if (n >= T["accum_above_n"] and not v.endswith("_rs")) else 1
    train_one(v, name, T["epochs"], T["batch_size"], T["lr"], T["max_pairs_per_traj"], T["width"],
              T["n_blocks"], seed=s, accum_steps=accum, data_dir=p["data"], tag=tag,
              traj_ids=split(c)[0], ckpt_dir=p["checkpoints"])


def run_eval(c, v, i, s):
    from evaluate import evaluate, load_model
    from train import get_device
    p = paths(c)
    a = cells(c)[i][0]
    tag = f"_s{s}"
    todo = [b for b, _, _ in cells(c)
            if not os.path.exists(os.path.join(p["results"], f"{v}_train{a}_test{b}{tag}.json"))]
    if not todo:
        return
    E = c["eval"]
    model = load_model(v, a, get_device(), tag, p["checkpoints"])
    for b in todo:
        evaluate(v, a, b, E["rollout_steps"], c["test_trajectories"], E["starts_per_traj"], seed=0,
                 data_dir=p["data"], tag=tag, traj_ids=split(c)[1], out_dir=p["results"],
                 ckpt_dir=p["checkpoints"], model=model)


def category(a, b):
    if a[0] == b[0]:
        return "same"
    if a[2] == b[2]:
        return "re_only"
    if a[1] == b[1]:
        return "grid_only"
    return "re_and_grid"


def run_summary(c):
    p = paths(c)
    info = {name: (name, re, n) for name, re, n in cells(c)}
    rows = []
    for f in sorted(glob.glob(os.path.join(p["results"], "*.json"))):
        r = json.load(open(f))
        a, b = r["train_regime"], r["test_regime"]
        if a not in info or b not in info or not r["mean_rel_l2_curve"]:
            continue
        seed = int(os.path.basename(f)[:-5].rsplit("_s", 1)[1])
        rows.append({"variant": r["variant"], "seed": seed, "train": a, "train_re": info[a][1],
                     "train_n": info[a][2], "test": b, "test_re": info[b][1], "test_n": info[b][2],
                     "category": category(info[a], info[b]), "step1": r["mean_rel_l2_curve"][0],
                     "rollout": r["mean_rel_l2_averaged_over_rollout"],
                     "final": r["mean_rel_l2_final_step"], "n_rollouts": r["n_rollouts"]})
    write_csv(os.path.join(c["out"], "summary.csv"), rows)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["train"], r["test"])].append(r)
    mean_rows = []
    for (v, a, b), rs in sorted(groups.items()):
        base = rs[0]
        row = {k: base[k] for k in ("variant", "train", "train_re", "train_n", "test", "test_re", "test_n",
                                    "category")}
        row["seeds"] = len(rs)
        for m in ("step1", "rollout"):
            for s, f in (("mean", np.mean), ("std", np.std)):
                row[f"{m}_{s}"] = f(np.array([r[m] for r in rs]))
        mean_rows.append(row)
    write_csv(os.path.join(c["out"], "summary_mean.csv"), mean_rows)
    own = {(r["variant"], r["seed"], r["test"]): r["rollout"] for r in rows if r["category"] == "same"}
    excess = defaultdict(list)
    for r in rows:
        ref = own.get((r["variant"], r["seed"], r["test"]))
        if ref and r["category"] != "same":
            excess[(r["variant"], r["category"])].append(r["rollout"] / ref)
    print(f"{len(rows)} results -> {os.path.join(c['out'], 'summary.csv')}, summary_mean.csv")
    print("rollout error relative to a model trained on the test cell (median, IQR):")
    for v in c["variants"]:
        parts = []
        for cat in ("re_only", "grid_only", "re_and_grid"):
            x = np.array(excess.get((v, cat), []))
            parts.append(f"{cat} {np.median(x):.2f} [{np.percentile(x, 25):.2f}, {np.percentile(x, 75):.2f}] n={len(x)}"
                         if len(x) else f"{cat} -")
        print(f"  {v:>12}: " + "   ".join(parts))


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (float, np.floating)) else "-"


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_task(c, stage, args):
    fn = {"calib": run_calib, "data": run_data, "manifest": run_manifest, "train": run_train,
          "eval": run_eval, "summary": run_summary}[stage]
    fn(c, *args)


def run_stage(c, stage, jobs, gpus, threads):
    n = len(tasks(c, stage))
    logs = paths(c)["logs"]
    failed = []

    def one(k):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads), TORCH_NUM_THREADS=str(threads),
                   FFT_WORKERS=str(threads))
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[k % len(gpus)]
        log = os.path.join(logs, f"{stage}_{k:04d}.log")
        with open(log, "a") as fh:
            r = subprocess.run([sys.executable, os.path.abspath(__file__), c["path"], "task", stage, str(k)],
                               stdout=fh, stderr=subprocess.STDOUT, env=env)
        if r.returncode:
            failed.append(k)
        print(f"[{time.strftime('%H:%M:%S')}] {stage} {k + 1}/{n} {'FAILED, see ' + log if r.returncode else 'done'}",
              flush=True)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        list(ex.map(one, range(n)))
    if failed:
        print(f"{stage}: {len(failed)} of {n} tasks failed: {failed[:20]}", flush=True)
    return not failed


def plan(c):
    C = cells(c)
    snaps = int(round(c["t_total"] / c["dt_sample"])) + 1
    gb = sum(n * n * 4 * snaps for _, _, n in C) * c["trajectories"] / 1e9
    print(f"output: {c['out']}")
    print(f"{len(C)} cells:")
    for n in c["grids"]:
        print(f"  {n:>4}^2: " + ", ".join(f"Re {re:g}" for _, re, m in C if m == n))
    print(f"fixed grid for the *_rs surrogates: {c.get('fixed_grid', 128)}^2")
    print(f"data: {c['trajectories']} trajectories per cell ({c['test_trajectories']} held out), "
          f"{snaps} snapshots each, about {gb:.0f} GB")
    for s in STAGES:
        print(f"  {s:>8}: {len(tasks(c, s))} tasks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("command", choices=["plan", "count", "task", "stage", "all"])
    ap.add_argument("stage", nargs="?", choices=STAGES)
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    help="comma-separated GPU ids for train/eval, e.g. 0,1,2,3")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per task")
    ap.add_argument("--gpu-jobs", type=int, default=None, help="parallel train/eval tasks (default: 2 per GPU)")
    a = ap.parse_args()
    c = load(a.config)
    os.environ["FIXED_GRID"] = str(c.get("fixed_grid", 128))
    if a.command == "plan":
        return plan(c)
    if a.command == "count":
        return print(len(tasks(c, a.stage)))
    if a.command == "task":
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", 1)))
        return run_task(c, a.stage, tasks(c, a.stage)[a.index])
    gpus = [g for g in a.gpus.split(",") if g.strip()]
    stages = STAGES if a.command == "all" else [a.stage]
    for s in stages:
        jobs = a.jobs
        if s in ("train", "eval") and gpus:
            jobs = a.gpu_jobs or 2 * len(gpus)
        if s in ("manifest", "summary"):
            print(f"=== {s}", flush=True)
            run_task(c, s, ())
            continue
        print(f"=== {s}: {len(tasks(c, s))} tasks, {jobs} at a time", flush=True)
        if not run_stage(c, s, jobs, gpus if s in ("train", "eval") else [], a.threads):
            sys.exit(f"stopping: {s} had failures; fix and rerun, finished tasks are skipped")


if __name__ == "__main__":
    main()
