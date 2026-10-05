#!/usr/bin/env python3
"""Compare the torch (GPU) solver with the numpy solver from the same state, and time both.

    python3 scripts/check_gpu_solver.py --device cuda --n 256
    python3 scripts/check_gpu_solver.py --device torch:cpu --n 64 --steps 100
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from solver.config import load_config  # noqa: E402
from solver.ns2d import NS2D  # noqa: E402


def advance(cfg, device, state, steps):
    os.environ["SOLVER_DEVICE"] = device
    sim = NS2D(cfg, forcing_amplitude=0.6)
    sim.omega_hat = state.copy()
    if sim._accelerator() is not None:
        sim._accelerator()
    t = time.time()
    res = sim.run(steps * cfg.time.dt, diagnostics_interval=steps * cfg.time.dt / 4)
    return sim.omega_hat, (time.time() - t) / steps, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--re", type=float, default=160.0)
    ap.add_argument("--steps", type=int, default=400)
    a = ap.parse_args()
    cfg = load_config(os.path.join(ROOT, "configs", "re_mid.yaml"),
                      {"grid": {"Nx": a.n, "Ny": a.n}, "time": {"dt": 0.32 / a.n},
                       "viscosity": {"target_re_f": a.re}})
    os.environ["SOLVER_DEVICE"] = "cpu"
    sim = NS2D(cfg, forcing_amplitude=0.6)
    sim.set_initial_state(seed=1)
    sim.run(200 * cfg.time.dt)
    state = sim.omega_hat.copy()
    ref, t_cpu, r_cpu = advance(cfg, "cpu", state, a.steps)
    advance(cfg, a.device, state, 20)
    out, t_dev, r_dev = advance(cfg, a.device, state, a.steps)
    rel = np.linalg.norm(out - ref) / np.linalg.norm(ref)
    dE = max(abs(x - y) / abs(x) for x, y in zip(r_cpu.diagnostics["E"], r_dev.diagnostics["E"]))
    print(f"{a.n}^2, Re {a.re:g}, {a.steps} steps")
    print(f"  relative difference in the final state: {rel:.2e}")
    print(f"  largest relative difference in energy:   {dE:.2e}")
    print(f"  numpy {t_cpu * 1e3:.2f} ms/step, {a.device} {t_dev * 1e3:.2f} ms/step ({t_cpu / t_dev:.1f}x)")
    print("  OK" if rel < 1e-9 and dE < 1e-9 else "  MISMATCH")
    sys.exit(0 if rel < 1e-9 and dE < 1e-9 else 1)


if __name__ == "__main__":
    main()
