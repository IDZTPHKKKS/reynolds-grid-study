#!/usr/bin/env python3
"""Train one (variant, source_regime) model."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NextStepDataset
from features import IN_CHANNELS
from model import VorticityCNN, FNO2d, UNet2d

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one(variant: str, regime: str, epochs: int, batch_size: int, lr: float,
             max_pairs_per_traj: int, width: int, n_blocks: int, seed: int = 0,
             accum_steps: int = 1, data_dir: str = None, tag: str = "", traj_ids=None,
             ckpt_dir: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = get_device()

    ckpt_dir = ckpt_dir or CKPT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    train_ds = NextStepDataset(regime, variant, "train", max_pairs_per_traj=max_pairs_per_traj,
                               data_dir=data_dir, traj_ids=traj_ids)
    val_ds = NextStepDataset(regime, variant, "val", max_pairs_per_traj=max_pairs_per_traj // 4
                             if max_pairs_per_traj else None, data_dir=data_dir,
                             traj_ids=traj_ids)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    arch = "fno" if variant.startswith("fno") else "unet" if variant.startswith("unet") else "cnn"
    if arch == "fno":
        model = FNO2d(IN_CHANNELS[variant], width=width, n_layers=n_blocks)
    elif arch == "unet":
        model = UNet2d(IN_CHANNELS[variant], width=width)
    else:
        model = VorticityCNN(IN_CHANNELS[variant], width=width, n_blocks=n_blocks)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"[{variant}/{regime}] device={device} params={model.n_params()} "
          f"train_pairs={len(train_ds)} val_pairs={len(val_ds)}", flush=True)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            opt.zero_grad()
            loss_total = 0.0
            for xm, ym in zip(x.chunk(accum_steps), y.chunk(accum_steps)):
                xm, ym = xm.to(device), ym.to(device)
                loss = torch.mean((model(xm) - ym) ** 2) * (xm.shape[0] / x.shape[0])
                loss.backward()
                loss_total += loss.item()
            opt.step()
            train_losses.append(loss_total)
        sched.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                val_losses.append(sum(
                    torch.mean((model(xm.to(device)) - ym.to(device)) ** 2).item() * xm.shape[0] / x.shape[0]
                    for xm, ym in zip(x.chunk(accum_steps), y.chunk(accum_steps))))

        tr, va = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_mse": tr, "val_mse": va,
                        "lr": sched.get_last_lr()[0], "wall_s": time.time() - t0})
        print(f"  epoch {epoch:3d}  train_mse={tr:.6e}  val_mse={va:.6e}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    ckpt_path = os.path.join(ckpt_dir, f"{variant}_{regime}{tag}.pt")
    torch.save({"model_state": model.state_dict(), "variant": variant, "regime": regime,
               "width": width, "n_blocks": n_blocks, "in_channels": IN_CHANNELS[variant],
               "arch": arch},
              ckpt_path)
    with open(os.path.join(ckpt_dir, f"{variant}_{regime}{tag}_history.json"), "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"  saved {ckpt_path}")
    return ckpt_path, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline", "divisive", "conditioned",
                                                              "quantile", "baseline_rs", "quantile_rs",
                                                              "fno", "fno_rs", "unet", "unet_rs"])
    ap.add_argument("--regime", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-pairs-per-traj", type=int, default=800)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="split each batch into this many micro-batches (same gradient, less memory)")
    args = ap.parse_args()
    train_one(args.variant, args.regime, args.epochs, args.batch_size, args.lr,
             args.max_pairs_per_traj, args.width, args.n_blocks, args.seed, args.accum_steps)


if __name__ == "__main__":
    main()
