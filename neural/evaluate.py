#!/usr/bin/env python3
"""Autoregressive rollout evaluation within and across Reynolds regimes."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from solver.spectral import SpectralGrid2D  # noqa: E402
from dataset import load_regime_trajectories  # noqa: E402
from features import make_input, invert_prediction  # noqa: E402
from model import VorticityCNN, FNO2d, UNet2d  # noqa: E402
from train import CKPT_DIR, get_device  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_model(variant: str, train_regime: str, device, tag: str = "", ckpt_dir: str = None):
    ckpt = torch.load(os.path.join(ckpt_dir or CKPT_DIR, f"{variant}_{train_regime}{tag}.pt"),
                      map_location=device, weights_only=False)
    if ckpt.get("arch", "cnn") == "fno":
        model = FNO2d(ckpt["in_channels"], width=ckpt["width"], n_layers=ckpt["n_blocks"])
    elif ckpt.get("arch") == "unet":
        model = UNet2d(ckpt["in_channels"], width=ckpt["width"])
    else:
        model = VorticityCNN(ckpt["in_channels"], width=ckpt["width"], n_blocks=ckpt["n_blocks"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


def rollout_one_trajectory(model, variant, grid, nu, omega_traj, n_start, rollout_steps, device):
    """Autoregressive rollout starting from omega_traj[n_start], compared against the
    TRUE trajectory at each subsequent stored step."""
    errors = []
    omega_current = omega_traj[n_start].copy()
    for step in range(1, rollout_steps + 1):
        x = make_input(variant, grid, nu, omega_current)
        x_t = torch.from_numpy(x[None]).to(device)
        with torch.no_grad():
            pred = model(x_t).cpu().numpy()[0]
        omega_pred = invert_prediction(variant, grid, nu, pred, omega_current)
        true_idx = n_start + step
        if true_idx >= omega_traj.shape[0]:
            break
        omega_true = omega_traj[true_idx]
        rel_l2 = float(np.linalg.norm(omega_pred - omega_true) / (np.linalg.norm(omega_true) + 1e-12))
        errors.append(rel_l2)
        omega_current = omega_pred  # autoregressive: feed prediction back in
    return errors


def evaluate(variant, train_regime, test_regime, rollout_steps, n_test_trajectories,
            n_starts_per_traj, seed=0, data_dir=None, tag="", traj_ids=None, out_dir=None,
            ckpt_dir=None, model=None):
    device = get_device()
    if model is None:
        model = load_model(variant, train_regime, device, tag, ckpt_dir)

    trajs = load_regime_trajectories(test_regime, data_dir, traj_ids)[:n_test_trajectories]
    meta0 = trajs[0]["meta"]
    grid = SpectralGrid2D(meta0["grid"]["Nx"], meta0["grid"]["Ny"],
                          meta0["domain"]["Lx"], meta0["domain"]["Ly"])
    nu = meta0["physics"]["nu"]

    rng = np.random.default_rng(seed)
    all_errors = []  # list of (rollout_steps,) arrays
    for traj in trajs:
        omega_traj = traj["omega"]
        n = omega_traj.shape[0]
        max_start = n - rollout_steps - 1
        if max_start <= 0:
            continue
        starts = rng.choice(max_start, size=min(n_starts_per_traj, max_start), replace=False)
        for s in starts:
            errs = rollout_one_trajectory(model, variant, grid, nu, omega_traj, int(s),
                                          rollout_steps, device)
            if len(errs) == rollout_steps:
                all_errors.append(errs)

    all_errors = np.array(all_errors)  # (n_rollouts, rollout_steps)
    mean_curve = all_errors.mean(axis=0).tolist()
    std_curve = all_errors.std(axis=0).tolist()

    result = {
        "variant": variant, "train_regime": train_regime, "test_regime": test_regime,
        "cross_re": train_regime != test_regime,
        "rollout_steps": rollout_steps, "n_rollouts": int(all_errors.shape[0]),
        "mean_rel_l2_curve": mean_curve, "std_rel_l2_curve": std_curve,
        "mean_rel_l2_final_step": mean_curve[-1] if mean_curve else None,
        "mean_rel_l2_averaged_over_rollout": float(np.mean(all_errors)) if all_errors.size else None,
    }
    out_dir = out_dir or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{variant}_train{train_regime}_test{test_regime}{tag}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[{variant}] train={train_regime} test={test_regime} (cross_re={result['cross_re']}) "
          f"n_rollouts={result['n_rollouts']} "
          f"mean_rel_l2(final step)={result['mean_rel_l2_final_step']:.4f} "
          f"mean_rel_l2(avg over rollout)={result['mean_rel_l2_averaged_over_rollout']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--train-regime", required=True)
    ap.add_argument("--test-regime", required=True)
    ap.add_argument("--rollout-steps", type=int, default=20)
    ap.add_argument("--n-test-trajectories", type=int, default=2)
    ap.add_argument("--n-starts-per-traj", type=int, default=20)
    args = ap.parse_args()
    evaluate(args.variant, args.train_regime, args.test_regime, args.rollout_steps,
            args.n_test_trajectories, args.n_starts_per_traj)


if __name__ == "__main__":
    main()
