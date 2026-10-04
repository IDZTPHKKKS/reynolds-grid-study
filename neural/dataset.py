"""Dataset construction for the neural cross-Re generalization experiment."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from solver.spectral import SpectralGrid2D  # noqa: E402
from features import make_input, make_target  # noqa: E402


def load_regime_trajectories(regime: str, data_dir: str = None, traj_ids=None):
    import json
    data_dir = data_dir or os.path.join(_REPO_ROOT, "data")
    with open(os.path.join(data_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    metas = [m for m in manifest if m["name"] == regime]
    if traj_ids is not None:
        metas = [m for m in metas if m["trajectory"] in set(traj_ids)]
    metas.sort(key=lambda m: m["trajectory"])
    trajs = []
    for meta in metas:
        d = np.load(os.path.join(data_dir, meta["file"]))
        trajs.append({"omega": d["omega"].astype(np.float32), "meta": meta})
    return trajs


class NextStepDataset(Dataset):
    """(input, target) pairs at lag=1 stored snapshot, for one regime, one
    preprocessing variant, restricted to a temporal split."""

    def __init__(self, regime: str, variant: str, split: str,
                val_fraction: float = 0.15, data_dir: str = None,
                max_pairs_per_traj: int = None, traj_ids=None):
        self.regime = regime
        self.variant = variant
        trajs = load_regime_trajectories(regime, data_dir, traj_ids)
        assert len(trajs) > 0, f"no trajectories found for regime {regime}"
        meta0 = trajs[0]["meta"]
        self.grid = SpectralGrid2D(meta0["grid"]["Nx"], meta0["grid"]["Ny"],
                                   meta0["domain"]["Lx"], meta0["domain"]["Ly"])
        self.nu = meta0["physics"]["nu"]

        self.pairs = []  # list of (traj_idx, t_idx)
        for ti, traj in enumerate(trajs):
            n = traj["omega"].shape[0]
            n_val = max(1, int(round(val_fraction * n)))
            if split == "train":
                idx_range = range(0, n - n_val - 1)
            elif split == "val":
                idx_range = range(n - n_val - 1, n - 1)
            elif split == "all":
                idx_range = range(0, n - 1)
            else:
                raise ValueError(split)
            idx_list = list(idx_range)
            if max_pairs_per_traj is not None and len(idx_list) > max_pairs_per_traj:
                idx_list = list(np.linspace(idx_list[0], idx_list[-1],
                                            max_pairs_per_traj).round().astype(int))
            for t in idx_list:
                self.pairs.append((ti, t))
        self.trajs = trajs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ti, t = self.pairs[i]
        omega_now = self.trajs[ti]["omega"][t]
        omega_next = self.trajs[ti]["omega"][t + 1]
        x = make_input(self.variant, self.grid, self.nu, omega_now)
        y = make_target(self.variant, self.grid, self.nu, omega_next, omega_now)
        return torch.from_numpy(x), torch.from_numpy(y)
