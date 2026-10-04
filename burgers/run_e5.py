#!/usr/bin/env python3
"""Phase 6 E5: cross-Re test on decaying 1D Burgers, native vs fixed grid, CNN and FNO (seed 0).

Writes burgers/_results/e5.json. Data are generated into burgers/_data/ if missing.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from solver import resample, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "_data"), os.path.join(HERE, "_results")
REGIMES = {"low": (0.1, 128), "mid": (0.03, 256), "high": (0.01, 1024)}
N_FIX, DT, N_SAMPLES, N_TRAIN, N_TEST = 256, 0.05, 80, 64, 16
EPOCHS, BATCH, LR, STEPS = 18, 32, 3e-4, 20
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0


class ResBlock1d(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv1d(w, w, 3, padding=1, padding_mode="circular")
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.c2(self.act(self.c1(x)))


class CNN1d(nn.Module):
    def __init__(self, w=32, n_blocks=4):
        super().__init__()
        self.stem = nn.Conv1d(1, w, 3, padding=1, padding_mode="circular")
        self.blocks = nn.Sequential(*[ResBlock1d(w) for _ in range(n_blocks)])
        self.head = nn.Conv1d(w, 1, 3, padding=1, padding_mode="circular")
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.head(self.blocks(self.act(self.stem(x))))


class Spectral1d(nn.Module):
    def __init__(self, w, modes):
        super().__init__()
        self.modes = modes
        self.w = nn.Parameter(torch.randn(w, w, modes, dtype=torch.cfloat) / (w * w))

    def forward(self, x):
        xh = torch.fft.rfft(x)
        out = torch.zeros(x.shape[0], x.shape[1], xh.shape[-1], dtype=torch.cfloat)
        out[..., :self.modes] = torch.einsum("bix,iox->box", xh[..., :self.modes], self.w)
        return torch.fft.irfft(out, n=x.shape[-1])


class FNO1d(nn.Module):
    def __init__(self, w=32, n_layers=4, modes=16):
        super().__init__()
        self.lift = nn.Conv1d(1, w, 1)
        self.spec = nn.ModuleList([Spectral1d(w, modes) for _ in range(n_layers)])
        self.point = nn.ModuleList([nn.Conv1d(w, w, 1) for _ in range(n_layers)])
        self.proj = nn.Sequential(nn.Conv1d(w, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.act = nn.GELU()

    def forward(self, x):
        h = self.lift(x)
        for i, (s, p) in enumerate(zip(self.spec, self.point)):
            h = s(h) + p(h)
            if i < len(self.spec) - 1:
                h = self.act(h)
        return x + self.proj(h)


def data(name):
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, f"{name}.npz")
    if not os.path.exists(path):
        nu, n = REGIMES[name]
        seed = {"low": 11, "mid": 22, "high": 33}[name]
        np.savez(path, train=simulate(n, nu, N_TRAIN, N_SAMPLES, DT, seed).astype(np.float32),
                 test=simulate(n, nu, N_TEST, N_SAMPLES, DT, seed + 100).astype(np.float32))
    d = np.load(path)
    return d["train"], d["test"]


def train(arch, fixed, name):
    torch.manual_seed(SEED)
    tr, _ = data(name)
    if fixed:
        tr = resample(tr.astype(np.float64), N_FIX).astype(np.float32)
    n_val = 8
    xs = torch.from_numpy(tr[:-n_val, :-1].reshape(-1, 1, tr.shape[-1]))
    ys = torch.from_numpy(tr[:-n_val, 1:].reshape(-1, 1, tr.shape[-1]))
    model = CNN1d() if arch == "cnn" else FNO1d()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(EPOCHS):
        perm = torch.randperm(len(xs), generator=g)
        for i in range(0, len(xs), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = torch.mean((model(xs[idx]) - ys[idx]) ** 2)
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


def evaluate(model, fixed, test_name):
    _, te = data(test_name)
    te = te.astype(np.float64)
    n = te.shape[-1]
    errs = []
    for traj in te:
        for s in range(0, N_SAMPLES - STEPS + 1, 10):
            u = traj[s]
            e = []
            for k in range(1, STEPS + 1):
                x = resample(u, N_FIX) if fixed else u
                with torch.no_grad():
                    y = model(torch.from_numpy(x[None, None].astype(np.float32)))[0, 0].double().numpy()
                u = resample(y, n) if fixed else y
                e.append(np.linalg.norm(u - traj[s + k]) / np.linalg.norm(traj[s + k]))
            errs.append(e)
    errs = np.array(errs)
    return {"rollout": float(errs.mean()), "step1": float(errs[:, 0].mean())}


def main():
    torch.set_num_threads(4)
    os.makedirs(RES, exist_ok=True)
    out = {}
    for arch, fixed in itertools.product(("cnn", "fno"), (False, True)):
        key = f"{arch}{'_rs' if fixed else ''}"
        for a in REGIMES:
            model = train(arch, fixed, a)
            os.makedirs(os.path.join(RES, "models"), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(RES, "models", f"{key}_{a}_s{SEED}.pt"))
            for b in REGIMES:
                out[f"{key}|{a}|{b}"] = r = evaluate(model, fixed, b)
                print(f"{key:7s} {a:>4s}->{b:<4s} rollout {r['rollout']:.4f} step1 {r['step1']:.4f}", flush=True)
            name = "e5.json" if SEED == 0 else f"e5_s{SEED}.json"
            json.dump(out, open(os.path.join(RES, name), "w"), indent=2)
    print("e5 done", flush=True)


if __name__ == "__main__":
    main()
