#!/usr/bin/env python3
"""Generate statistically stationary Kolmogorov-flow (forced 2D NS) trajectories."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.config import CONVENTIONS_VERSION, load_config
from solver.diagnostics import reynolds_numbers, state_diagnostics
from solver.forcing import KolmogorovForcing
from solver.ns2d import NS2D


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calibrate_amplitude(cfg, verbose: bool = True):
    """Adjust A until the measured stationary u_rms matches the target."""
    target = cfg.calibration.target_u_rms
    A = float(cfg.forcing.amplitude)
    sim = NS2D(cfg, forcing_amplitude=A)
    sim.set_initial_state(seed=cfg.initial_condition.seed + 5000)
    log("  calibration: initial equilibration (stationarity criteria) ...")
    burn0, _ = sim.run_until_stationary(verbose=False)
    log(f"  calibration: equilibrated after {burn0:.0f} time units")
    history = []
    for it in range(cfg.calibration.max_iterations):
        res = sim.run(cfg.calibration.pilot_measure, diagnostics_interval=0.5)
        u_rms = float(np.mean(res.diagnostics["u_rms"]))
        rel = abs(u_rms - target) / target
        history.append({"iteration": it, "A": A, "u_rms_measured": u_rms, "rel_error": rel})
        if verbose:
            log(f"  calibration iter {it}: A={A:.4f} -> u_rms={u_rms:.4f} (target {target}, rel err {rel:.3f})")
        if rel < cfg.calibration.tolerance:
            return A, history
        if len(history) == 1:
            A_new = A * (target / u_rms) ** (1.0 / 0.75) if u_rms > 0 else 2 * A
        else:
            # secant in (log A, log u)
            A0 = history[-2]["A"]; u0 = history[-2]["u_rms_measured"]
            A1 = history[-1]["A"]; u1 = history[-1]["u_rms_measured"]
            if u0 == u1:  # pragma: no cover
                A_new = A1 * (target / u1) ** (1.0 / 0.75)
            else:
                slope = (np.log(u1) - np.log(u0)) / (np.log(A1) - np.log(A0))
                if not np.isfinite(slope) or not (0.4 <= abs(slope) <= 1.2):
                    slope = 0.75
                A_new = A1 * (target / u1) ** (1.0 / slope)
        A_undamped = float(np.clip(0.9 * A_new + 0.1 * A, 1e-8, 1e8))
        max_step = 1.35
        A = float(np.clip(A_undamped, A / max_step, A * max_step))
        sim.forcing = KolmogorovForcing(A, cfg.forcing.mode, sim.grid)
        sim.run(cfg.calibration.pilot_reequilibrate,
                diagnostics_interval=cfg.calibration.pilot_reequilibrate)
    err = RuntimeError(f"Amplitude calibration did not converge: {history}")
    err.history = history
    raise err


def generate_trajectory(cfg, amplitude: float, traj_index: int, outdir: str):
    sim = NS2D(cfg, forcing_amplitude=amplitude)
    seed = cfg.initial_condition.seed + cfg.output.seed_offset + 1000 * traj_index
    sim.set_initial_state(seed=seed)
    log(f"  traj {traj_index}: IC seed {seed}, burning in ...")
    t0 = time.time()
    burn, report = sim.run_until_stationary(verbose=False)
    log(f"  traj {traj_index}: burn-in {burn:.0f} time units "
        f"({report['drift_E']:.4f}/{report['drift_enstrophy']:.4f}/{report['production_dissipation_mismatch']:.4f})")

    res = sim.run(cfg.time.t_total, sample_interval=cfg.time.dt_sample,
                  diagnostics_interval=min(cfg.time.dt_sample, 0.5), progress=True)
    omega = np.stack(res.samples).astype(cfg.output.dtype)
    d_avg = {k: float(np.mean(v)) for k, v in res.diagnostics.items()
             if k not in ("time", "courant")}
    # Reynolds numbers from the *time-averaged* stationary state.
    re = reynolds_numbers(sim.grid, sim.forcing, sim.nu, d_avg, drag=sim.drag)

    fname = f"{cfg.name}_traj{traj_index}.npz"
    np.savez_compressed(
        os.path.join(outdir, fname),
        omega=omega,
        times=np.asarray(res.sample_times),
        diag_times=np.asarray(res.times),
        diag_names=np.array(sorted(res.diagnostics.keys())),
        **{f"diag_{k}": np.asarray(v) for k, v in res.diagnostics.items()},
    )
    meta = {
        "name": cfg.name,
        "trajectory": traj_index,
        "seed": seed,
        "file": fname,
        "n_snapshots": int(omega.shape[0]),
        "grid": {"Nx": cfg.grid.Nx, "Ny": cfg.grid.Ny},
        "domain": {"Lx": cfg.domain.Lx, "Ly": cfg.domain.Ly},
        "physics": {
            "nu": sim.nu, "drag": sim.drag,
            "forcing_amplitude": amplitude, "forcing_mode": cfg.forcing.mode,
            "k_f": sim.forcing.k_f,
        },
        "time": {"dt": cfg.time.dt, "dt_sample": cfg.time.dt_sample,
                 "t_total": cfg.time.t_total, "burn_in": burn},
        "numerics": {"integrator": cfg.numerics.integrator, "dealias": cfg.numerics.dealias,
                     "k_max": sim.grid.k_max, "nx_max": sim.grid.nx_max, "ny_max": sim.grid.ny_max},
        "measured": {
            "u_rms_mean": d_avg["u_rms"], "E_mean": d_avg["E"],
            "enstrophy_mean": d_avg["enstrophy"],
            "P_in_mean": d_avg["P_in"], "eps_mean": d_avg["eps"],
            "max_courant": res.max_courant,
            **{k: (float(v) if np.isfinite(v) else None) for k, v in re.items()},
        },
        "stationarity": report,
        "conventions_version": CONVENTIONS_VERSION,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(outdir, f"{fname[:-4]}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    log(f"  traj {traj_index}: saved {fname}: {omega.shape[0]} snapshots, "
        f"Re_f={re['Re_f']:.1f}, k_max/k_d={re['k_max_over_k_d']:.2f}, "
        f"CFL_max={res.max_courant:.2f}, wall {meta['wall_time_s']:.0f}s")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--trajs", type=int, default=None, help="override output.n_traj")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    manifest_path = os.path.join(args.outdir, "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)

    for path in args.config:
        cfg = load_config(path)
        if args.trajs is not None:
            cfg.output.n_traj = args.trajs
        log(f"=== {cfg.name}: N={cfg.grid.Nx}, nu={cfg.nu:.4e}, drag={cfg.viscosity.drag}, "
            f"target Re_f={cfg.viscosity.target_re_f} ===")
        A, hist = calibrate_amplitude(cfg) if cfg.calibration.enabled else (cfg.forcing.amplitude, [])
        log(f"  calibrated amplitude A = {A:.4f}")
        for traj in range(cfg.output.n_traj):
            meta = generate_trajectory(cfg, A, traj, args.outdir)
            meta["calibration"] = hist
            meta["config_path"] = path
            manifest.append(meta)
            with open(manifest_path, "w") as fh:
                json.dump(manifest, fh, indent=2)
    log(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
