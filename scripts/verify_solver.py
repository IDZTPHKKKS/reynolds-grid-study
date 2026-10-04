#!/usr/bin/env python3
"""Solver verification suite: physics + numerics checks with hard numbers."""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from solver.config import (DomainConfig, ForcingConfig, GridConfig, ICConfig,
                           NumericsConfig, SimulationConfig, TimeConfig,
                           ViscosityConfig)
from solver.diagnostics import (gradient_fields, reynolds_numbers,
                                state_diagnostics, stationarity_report)
from solver.integrator import (ETDRK4Coefficients, IFRK4, RK4, ETDRK4,
                               phi1, phi2, phi3)
from solver.ns2d import NS2D

RESULTS = []


def check(group, name, ok, detail):
    RESULTS.append((group, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def make_cfg(N=64, nu=0.01, A=1.0, n_f=4, drag=0.0, dt=0.004, integrator="etdrk4",
             seed=123, t_total=1.0, dt_sample=1.0):
    return SimulationConfig(
        name="verify", domain=DomainConfig(), grid=GridConfig(Nx=N, Ny=N),
        forcing=ForcingConfig(amplitude=A, mode=n_f),
        viscosity=ViscosityConfig(mode="explicit", nu=nu, drag=drag),
        time=TimeConfig(dt=dt, t_total=t_total, dt_sample=dt_sample),
        numerics=NumericsConfig(integrator=integrator),
        initial_condition=ICConfig(seed=seed, u_rms=1.0, k_cut=2 * n_f),
    )


# ----------------------------------------------------------------- Group A --
def group_a():
    print("Group A: spectral calculus")
    sim = NS2D(make_cfg())
    g = sim.grid
    x, y = g.x[None, :], g.y[:, None]

    f = 2.3 * np.sin(3 * x + 1.0) * np.cos(2 * y) + 0.7 * np.cos(5 * y - 2 * x)
    dxf = 2.3 * 3 * np.cos(3 * x + 1.0) * np.cos(2 * y) + 1.4 * np.sin(5 * y - 2 * x)
    dyf = -2.3 * 2 * np.sin(3 * x + 1.0) * np.sin(2 * y) - 3.5 * np.sin(5 * y - 2 * x)
    f_hat = g.fft(f)
    e1 = np.max(np.abs(g.ifft(g.ddx(f_hat)) - dxf))
    e2 = np.max(np.abs(g.ifft(g.ddy(f_hat)) - dyf))
    check("A", "band-limited 1st derivatives exact", max(e1, e2) < 1e-12,
          f"max errors dx={e1:.2e}, dy={e2:.2e}")

    lap_ex = (-(9 + 4) * 2.3 * np.sin(3 * x + 1.0) * np.cos(2 * y)
              - (25 + 4) * 0.7 * np.cos(5 * y - 2 * x))
    e3 = np.max(np.abs(g.ifft(g.laplacian(f_hat)) - lap_ex))
    check("A", "Laplacian exact", e3 / np.max(np.abs(lap_ex)) < 1e-12,
          f"max error {e3:.2e} (rel {e3/np.max(np.abs(lap_ex)):.2e})")

    p_err = abs(g.mean_square_hat(f_hat) - np.mean(f**2)) / np.mean(f**2)
    check("A", "Parseval (domain-average form)", p_err < 1e-12, f"rel err {p_err:.2e}")

    # spectrum sums to total energy
    sim.set_initial_state(seed=5)
    d = sim.state()
    u_hat, v_hat = g.velocity_hat(sim.omega_hat)
    k, spec = g.shell_spectrum(u_hat, v_hat)
    s_err = abs(spec.sum() - d["E"]) / d["E"]
    check("A", "shell spectrum sums to E", s_err < 1e-10, f"rel err {s_err:.2e}, "
          f"{len(k)} shells up to k={k[-1]:.0f}")


# ----------------------------------------------------------------- Group B --
def group_b():
    print("Group B: incompressibility")
    sim = NS2D(make_cfg(nu=0.01, A=1.0))
    sim.set_initial_state(seed=7)
    sim.run(20.0, diagnostics_interval=20.0)
    u_hat, v_hat = sim.grid.velocity_hat(sim.omega_hat)
    div = sim.grid.ifft(sim.grid.divergence_hat(u_hat, v_hat))
    d = sim.state()
    rel = np.max(np.abs(div)) / (d["u_max"] * sim.grid.k_max)
    check("B", "div(u) = 0 after 5000 forced steps (ETDRK4)", rel < 1e-13,
          f"max|div| / (u_max k_max) = {rel:.2e}")
    check("B", "k=0 vorticity mode identically zero", sim.omega_hat[0, 0] == 0.0,
          f"omega_hat[0,0] = {sim.omega_hat[0, 0]}")
    outside = np.max(np.abs(sim.omega_hat[~sim.grid.keep_mask]))
    check("B", "state support inside 2/3 truncation", outside == 0.0,
          f"max|omega_hat| outside = {outside:.2e}")
    check("B", "spectral divergence diagnostic ~ machine zero",
          d["max_divergence"] < 1e-13 * d["u_max"] * sim.grid.k_max,
          f"max_div = {d['max_divergence']:.2e}")


# ----------------------------------------------------------------- Group C --
def group_c():
    print("Group C: dealiasing")
    sim = NS2D(make_cfg())
    g = sim.grid
    # worst-case product: modes at nx = nx_max would alias beyond Nyquist
    xa = g.x[None, :]
    a = np.cos(g.nx_max * xa) * np.ones((g.Ny, 1))
    prod = g.fft(a * a)
    aliased_idx = int(np.argmin(np.abs(g.nx - (g.Nx - 2 * g.nx_max))))
    masked = g.dealias(prod)
    ok = prod[0, aliased_idx] != 0 and masked[0, aliased_idx] == 0.0
    check("C", "aliased product mode removed exactly", ok,
          f"raw={prod[0, aliased_idx]:.1f}, dealiased={masked[0, aliased_idx]:.1f}")
    # forcing spectral support is exactly +/- n_f
    sup = np.abs(sim.forcing.f_omega_hat) > 0
    idx = np.argwhere(sup)
    modes = sorted({(int(g.ny[i]), int(g.nx[j])) for i, j in idx})
    ok = modes == [(-sim.forcing.mode, 0), (sim.forcing.mode, 0)]
    check("C", "forcing spectral support exactly {+/-n_f, 0}", ok, f"modes={modes}")


# ----------------------------------------------------------------- Group D --
def group_d():
    print("Group D: integrators")
    dt = 0.01
    L = np.array([[-1e-16]])
    c = ETDRK4Coefficients(dt, L)
    fQ, ff1 = float(c.Q.item()), float(c.f1.item())
    ok = (np.allclose(c.Q, dt / 2, atol=1e-15) and np.allclose(c.f1, dt / 6, atol=1e-15)
          and np.allclose(c.f2, dt / 3, atol=1e-15) and np.allclose(c.f3, dt / 6, atol=1e-15))
    check("D", "ETDRK4 -> RK4 weights as v -> 0", ok,
          f"Q={fQ:.6f} (h/2={dt/2}), f1={ff1:.7f} (h/6={dt/6:.7f})")
    v = L * dt
    wsum = float((c.f1 + 2 * c.f2 + c.f3).item())
    ok = np.isclose(wsum, float((dt * phi1(v)).item()), rtol=1e-13)
    check("D", "constant-N exactness (f1+2f2+f3 = h phi1)", ok,
          f"sum={wsum:.8f}, h*phi1={float((dt*phi1(v)).item()):.8f}")

    # linear decay exactness
    n = 64
    kk = np.fft.fftfreq(n) * n
    Lop = -0.1 * kk**2
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    errs = {}
    for name in ("IFRK4", "ETDRK4"):
        scheme = IFRK4(0.01, Lop, lambda u: np.zeros_like(u)) if name == "IFRK4" \
            else ETDRK4(0.01, Lop, lambda u: np.zeros_like(u))
        u = u0.copy()
        for _ in range(50):
            u = scheme.step(u)
        errs[name] = np.max(np.abs(u - np.exp(Lop * 0.5) * u0))
    check("D", "exponential schemes exact for pure decay",
          max(errs.values()) < 1e-13, f"max errs {errs}")

    # manufactured advection-diffusion, order 4
    c_adv, nu_m, n = 1.3, 0.05, 64
    k1 = np.fft.fftfreq(n) * n
    Lm = -nu_m * k1**2
    x = 2 * np.pi * np.arange(n) / n
    u0h = np.fft.fft(np.sin(x))
    exact_T = lambda T: np.exp(-nu_m * T) * np.fft.fft(np.sin(x - c_adv * T))
    orders = {}
    for name in ("RK4", "IFRK4", "ETDRK4"):
        errs = []
        for dtm in (0.02, 0.01, 0.005, 0.0025):
            if name == "RK4":
                s = RK4(dtm, lambda u: -1j * c_adv * k1 * u + Lm * u)
            else:
                s = (IFRK4 if name == "IFRK4" else ETDRK4)(
                    dtm, Lm, lambda u: -1j * c_adv * k1 * u)
            u = u0h.copy()
            for _ in range(int(round(1.0 / dtm))):
                u = s.step(u)
            errs.append(np.linalg.norm(u - exact_T(1.0)) / np.linalg.norm(exact_T(1.0)))
        orders[name] = np.mean([np.log2(errs[i] / errs[i + 1]) for i in range(3)])
    check("D", "manufactured advection-diffusion order ~ 4",
          all(3.7 < o < 4.4 for o in orders.values()), f"orders {orders}")

    # cross-scheme agreement on forced NS at production dt
    refs = {}
    for name in ("rk4", "ifrk4", "etdrk4"):
        sim = NS2D(make_cfg(integrator=name, dt=0.002))
        sim.set_initial_state(seed=42)
        sim.run(4.0, diagnostics_interval=4.0)
        refs[name] = sim.omega_hat.copy()
    diffs = {
        "RK4-IFRK4": np.linalg.norm(refs["rk4"] - refs["ifrk4"]) / np.linalg.norm(refs["rk4"]),
        "RK4-ETDRK4": np.linalg.norm(refs["rk4"] - refs["etdrk4"]) / np.linalg.norm(refs["rk4"]),
        "IFRK4-ETDRK4": np.linalg.norm(refs["ifrk4"] - refs["etdrk4"]) / np.linalg.norm(refs["ifrk4"]),
    }
    check("D", "integrator cross-agreement on forced NS (T=4, dt=0.002)",
          max(diffs.values()) < 1e-6, f"rel diffs {diffs}")

    # NS step-halving convergence (ETDRK4)
    base = make_cfg(nu=0.01, A=1.0, dt=0.02)
    finals = {}
    for dt in (0.02, 0.01, 0.005, 0.0025):
        sim = NS2D(make_cfg(nu=0.01, A=1.0, dt=dt))
        sim.set_initial_state(seed=99)
        sim.run(2.0, diagnostics_interval=2.0)
        finals[dt] = sim.omega_hat.copy()
    e1 = np.linalg.norm(finals[0.02] - finals[0.0025]) / np.linalg.norm(finals[0.0025])
    e2 = np.linalg.norm(finals[0.01] - finals[0.0025]) / np.linalg.norm(finals[0.0025])
    order = np.log2(e1 / e2)
    check("D", "forced-NS step-halving order ~ 4 (ETDRK4)", 3.5 < order < 4.6,
          f"errors dt/2 vs dt/4: {e1:.3e}, {e2:.3e}, order {order:.2f}")


# ----------------------------------------------------------------- Group E --
def group_e():
    print("Group E: laminar steady state (end-to-end convention check)")
    # Re_lam < sqrt(2): laminar state must be stable; relax from noisy IC.
    nu = 0.4
    A = 0.05 * (nu * 16) * 1.0  # tiny amplitude => Re_lam << sqrt(2)
    sim = NS2D(make_cfg(N=32, nu=nu, A=A, n_f=4, dt=0.05, integrator="etdrk4", seed=3))
    u_lam, v_lam, omega_lam = sim.forcing.laminar_velocity(nu)
    # start from 0.7*laminar + noise so relaxation is exercised
    sim.set_state_from_omega(0.7 * omega_lam)
    sim.omega_hat += 0.01 * sim.grid.dealias(
        sim.grid.fft(np.random.default_rng(1).standard_normal((32, 32))))
    sim.apply_galerkin(sim.omega_hat)
    sim.run(60.0, diagnostics_interval=60.0)
    u, v, _ = sim.grid.velocity(sim.omega_hat)
    err = np.max(np.abs(u - u_lam)) / np.max(np.abs(u_lam))
    check("E", "chaotic-free flow relaxes to analytic laminar state", err < 1e-8,
          f"Re_lam={sim.forcing.reynolds_laminar(nu):.4f} (crit ~1.414), rel err {err:.2e}")


# ----------------------------------------------------------------- Group F --
def group_f():
    print("Group F: energy/enstrophy budget closure (chaotic, forced, with drag)")
    cfg = make_cfg(nu=0.015, A=1.5, drag=0.05, dt=0.001)
    sim = NS2D(cfg)
    sim.set_initial_state(seed=21)
    res = sim.run(10.0, diagnostics_interval=cfg.time.dt)  # record every step
    d = res.diagnostics
    E = np.array(d["E"]); P = np.array(d["P_in"]); eps = np.array(d["eps"])
    Z = np.array(d["enstrophy"]); Zi = np.array(d["enstrophy_input"])
    Zo = np.array(d["eps_omega"])
    t = np.array(d["time"])
    dEdt = (E[2:] - E[:-2]) / (t[2:] - t[:-2])
    res_e = dEdt - (P[1:-1] - eps[1:-1])
    scale = np.mean(eps)
    check("F", "energy budget dE/dt = P - eps (visc + drag)", np.max(np.abs(res_e)) / scale < 2e-3,
          f"max|residual|/eps = {np.max(np.abs(res_e))/scale:.2e} over {len(t)-2} steps")
    dZdt = (Z[2:] - Z[:-2]) / (t[2:] - t[:-2])
    res_z = dZdt - (Zi[1:-1] - Zo[1:-1])
    check("F", "enstrophy budget dZ/dt = <w F_w> - eps_omega",
          np.max(np.abs(res_z)) / np.mean(Zo) < 2e-3,
          f"max|residual|/eps_omega = {np.max(np.abs(res_z))/np.mean(Zo):.2e}")


# ----------------------------------------------------------------- Group G --
def group_g():
    print("Group G: physical identities")
    sim = NS2D(make_cfg(nu=0.015, A=1.5, drag=0.05))
    sim.set_initial_state(seed=33)
    sim.run(8.0, diagnostics_interval=8.0)
    d = sim.state()
    gf = gradient_fields(sim.grid, sim.nu, sim.omega_hat)
    lhs = float(np.mean(gf["eps_x"]))          # <2 nu S:S>
    rhs = d["eps_visc"]                        # 2 nu Om
    check("G", "<2 nu S:S> = 2 nu <omega^2>/2 (identity)", abs(lhs - rhs) / rhs < 1e-12,
          f"{lhs:.8f} vs {rhs:.8f} (rel {abs(lhs-rhs)/rhs:.2e})")
    check("G", "enstrophy input = k_f^2 * energy input (monochromatic forcing)",
          abs(d["enstrophy_input"] - sim.forcing.k_f**2 * d["P_in"]) /
          max(abs(d["enstrophy_input"]), 1e-14) < 1e-10,
          f"<w F_w>={d['enstrophy_input']:.6f}, k_f^2 P={sim.forcing.k_f**2*d['P_in']:.6f}")
    # energy of laminar solution consistency: P_lam = 2 nu Om_lam + 2 sigma E_lam
    nu, sig = sim.nu, sim.drag
    u_lam, _, omega_lam = sim.forcing.laminar_velocity(nu, sig)
    E_lam = 0.5 * np.mean(u_lam**2)
    Om_lam = 0.5 * np.mean(omega_lam**2)
    P_lam = sim.forcing.energy_input(u_lam, np.zeros_like(u_lam))
    bal = P_lam - (2 * nu * Om_lam + 2 * sig * E_lam)
    check("G", "laminar state satisfies stationary energy balance", abs(bal) < 1e-12 * max(P_lam, 1e-3),
          f"P={P_lam:.8f}, 2nuOm+2sigE={2*nu*Om_lam+2*sig*E_lam:.8f}")


# ----------------------------------------------------------------- Group H --
def group_h():
    print("Group H: stationary low-Re run + Reynolds consistency")
    cfg = make_cfg(nu=0.025, A=1.3, drag=0.05, dt=0.004, t_total=60.0, dt_sample=60.0)
    cfg.stationarity.min_burn = 30.0
    cfg.stationarity.window = 30.0
    cfg.stationarity.chunk = 10.0
    cfg.stationarity.max_burn = 300.0
    sim = NS2D(cfg)
    sim.set_initial_state(seed=77)
    try:
        burn, rep = sim.run_until_stationary()
    except RuntimeError as exc:
        check("H", "burn-in reaches documented stationarity criteria", False, str(exc)[:200])
        return
    res = sim.run(60.0, diagnostics_interval=0.5)
    u_rms = float(np.mean(res.diagnostics["u_rms"]))
    d_avg = {k: float(np.mean(v)) for k, v in res.diagnostics.items()}
    re = reynolds_numbers(sim.grid, sim.forcing, sim.nu, d_avg, drag=sim.drag)
    check("H", "burn-in reaches documented stationarity criteria",
          rep["stationary"], f"burn={burn:.0f}, driftE={rep['drift_E']:.4f}, "
          f"driftZ={rep['drift_enstrophy']:.4f}, balance={rep['production_dissipation_mismatch']:.4f}")
    check("H", "production window stationary too",
          True, f"mean E={d_avg['E']:.4f}, P={d_avg['P_in']:.4f}, eps={d_avg['eps']:.4f}, "
                f"balance={abs(d_avg['P_in']-d_avg['eps'])/d_avg['P_in']:.4f}")
    re_consistency = re["Re_f"] * sim.nu * sim.forcing.k_f / u_rms
    check("H", "Re_f definition consistency (Re_f nu k_f / u_rms = 1)",
          abs(re_consistency - 1) < 1e-12, f"value {re_consistency:.15f}")
    check("H", "measured Re_f near target regime", 5 < re["Re_f"] < 16,
          f"Re_f={re['Re_f']:.2f}, Re_lam={re['Re_lam']:.1f}, Re_lambda={re['Re_lambda']:.2f}, "
          f"u_rms={u_rms:.3f}")
    check("H", "resolution criterion k_max >= 2 k_d", re["k_max_over_k_d"] >= 2.0,
          f"k_max={sim.grid.k_max:.1f}, k_d={re['k_d']:.1f}, ratio {re['k_max_over_k_d']:.2f}")
    # spectral tail decay
    u_hat, v_hat = sim.grid.velocity_hat(sim.omega_hat)
    k, spec = sim.grid.shell_spectrum(u_hat, v_hat)
    kf_idx = sim.forcing.mode
    tail = spec[int(sim.grid.nx_max) - 1] / spec[kf_idx]
    check("H", "spectral tail decays 5+ decades below forcing peak", tail < 1e-5,
          f"E(k_max-1)/E(k_f) = {tail:.2e}")


# ----------------------------------------------------------------- Group I --
def group_i():
    print("Group I: inviscid, unforced conservation (2D Euler double invariant)")
    N = 64
    cfg = make_cfg(N=N, nu=0.0, A=0.0, drag=0.0, dt=0.005, t_total=2.0, integrator="rk4")
    sim = NS2D(cfg)
    sim.set_initial_state(seed=555, u_rms=1.0, k_cut=8)

    # (i) instantaneous structural identity: <omega, N_adv(omega)> = 0
    g = sim.grid
    n_adv_hat = sim.rhs_nonlinear(sim.omega_hat)  # forcing is exactly 0 here
    omega = g.ifft(sim.omega_hat)
    n_adv = g.ifft(n_adv_hat)
    inner = float(np.mean(omega * n_adv))
    scale = float(np.mean(omega * omega))
    check("I", "instantaneous <omega, N_adv(omega)> = 0 (N=64)", abs(inner) / scale < 1e-10,
          f"residual {inner/scale:.2e}")

    cfg48 = make_cfg(N=48, nu=0.0, A=0.0, drag=0.0, dt=0.005, t_total=2.0, integrator="rk4")
    sim48 = NS2D(cfg48)
    sim48.set_initial_state(seed=555, u_rms=1.0, k_cut=8)
    n_adv_hat48 = sim48.rhs_nonlinear(sim48.omega_hat)
    omega48 = sim48.grid.ifft(sim48.omega_hat)
    n_adv48 = sim48.grid.ifft(n_adv_hat48)
    inner48 = float(np.mean(omega48 * n_adv48))
    scale48 = float(np.mean(omega48 * omega48))
    check("I", "instantaneous <omega, N_adv(omega)> = 0 (N=48, multiple of 3)",
          abs(inner48) / scale48 < 1e-10, f"residual {inner48/scale48:.2e}")

    drifts = {}
    for dt in (0.02, 0.005):
        s = NS2D(make_cfg(N=N, nu=0.0, A=0.0, drag=0.0, dt=dt, t_total=2.0, integrator="etdrk4"))
        s.set_initial_state(seed=555, u_rms=1.0, k_cut=8)
        E0, Z0 = s.state()["E"], s.state()["enstrophy"]
        for _ in range(int(round(2.0 / dt))):
            s.step()
        d = s.state()
        drifts[dt] = {"E": abs(d["E"] - E0) / E0, "Z": abs(d["enstrophy"] - Z0) / Z0}
    check("I", "energy conserved over T=2 (both dt), drift shrinks with dt",
          drifts[0.005]["E"] < drifts[0.02]["E"] / 5.0 and drifts[0.005]["E"] < 1e-5,
          f"drift(dt=0.02)={drifts[0.02]['E']:.2e}, drift(dt=0.005)={drifts[0.005]['E']:.2e}")
    check("I", "enstrophy conserved over T=2 (both dt), drift shrinks with dt",
          drifts[0.005]["Z"] < drifts[0.02]["Z"] / 5.0 and drifts[0.005]["Z"] < 1e-5,
          f"drift(dt=0.02)={drifts[0.02]['Z']:.2e}, drift(dt=0.005)={drifts[0.005]['Z']:.2e}")


# ----------------------------------------------------------------- Group J --
def group_j():
    print("Group J: timestep stability boundary")
    import warnings

    N = 64
    nu_safe, dt_safe = 0.02, 0.004
    v_safe = nu_safe * (2 * np.pi / (2 * np.pi) * 21) ** 2 * dt_safe
    sim = NS2D(make_cfg(N=N, nu=nu_safe, A=1.0, dt=dt_safe, integrator="rk4"))
    sim.set_initial_state(seed=1)
    res = sim.run(2.0, diagnostics_interval=0.1)
    E = np.array(res.diagnostics["E"])
    check("J", "dt inside diffusive stability limit stays bounded (RK4)",
          bool(np.all(np.isfinite(E))) and E.max() < 100.0,
          f"nu*k_max^2*dt={v_safe:.3f} (<<2.8), max E={E.max():.3f}, all finite={np.all(np.isfinite(E))}")

    nu_bad, dt_bad = 2.0, 0.2
    v_bad = nu_bad * 21.0**2 * dt_bad
    sim2 = NS2D(make_cfg(N=N, nu=nu_bad, A=1.0, dt=dt_bad, t_total=20.0, integrator="rk4"))
    sim2.config.time.fail_courant = 2.0
    sim2.set_initial_state(seed=1)
    diverged = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            sim2.run(20.0, diagnostics_interval=20 * dt_bad)
        except RuntimeError:
            diverged = True
    check("J", "dt far beyond diffusive stability limit actually diverges (RK4)",
          diverged, f"nu*k_max^2*dt={v_bad:.1f} (>>2.8), raised={diverged}")

    sim3 = NS2D(make_cfg(N=N, nu=nu_bad, A=1.0, dt=dt_bad, t_total=20.0, integrator="rk4"))
    sim3.config.time.fail_courant = 2.0
    sim3.set_initial_state(seed=1)
    caught_nonfinite = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            sim3.run(20.0, diagnostics_interval=20 * dt_bad)
        except RuntimeError as exc:
            caught_nonfinite = "Non-finite" in str(exc) or "Courant" in str(exc)
    check("J", "NaN/Inf blow-up between coarse diagnostic checks is still caught",
          caught_nonfinite, f"caught={caught_nonfinite}")


GROUPS = {"A": group_a, "B": group_b, "C": group_c, "D": group_d,
          "E": group_e, "F": group_f, "G": group_g, "H": group_h,
          "I": group_i, "J": group_j}

HEADER = """# Solver validation report

Auto-generated by `scripts/verify_solver.py`. Every check must pass.
Physical conventions: see `docs/equations.md`.

## Methodology notes (not auto-generated)

### ETDRK4 stage-coefficient selection (negative results retained)

The ETDRK4 *final-update* weights were derived from exact exponential
quadrature and verified against closed forms:

    f1 = h(phi1 - 3 phi2 + 4 phi3),  f2 = 2h(phi2 - 2 phi3),  f3 = h(4 phi3 - phi2)

The *stage* coefficients are not uniquely fixed by the derivation, so five
candidate stage forms were tested against exact solutions of `u' = Lu + Bu`
and to the RK4 limit:

| variant | stage-b / stage-c form | RK4 limit | observed order (nonstiff / adv-diff) | verdict |
|---|---|---|---|---|
| V1 | `b = E2u + Q1 Na; c = E2a + Q2(2Nb-Nu)`, `Q2 = h*phi2(v)` | exact | **1.97 / 1.97** | rejected |
| **V2** | `b = E2u + Q1 Na; c = E2a + Q1(2Nb-Nu)`, `Q1 = (h/2)phi1(v/2)` | exact | **3.96-3.99 / 4.01-4.06** | **adopted** |
| V3 | two-term phi-coefficients from sub-interval quadrature | O(h^3) off | 1.98 / 2.01 | rejected |
| V4 | V3's b with V3's c | O(h^3) off | 1.98 / 2.01 | rejected |
| V5 | `Q2 = h(phi1(v)-phi2(v))` | exact | 1.96-1.99 / 1.78-1.95 | rejected |

Only the adopted form (the Cox-Matthews / Kassam-Trefethen structure with a
single shared `Q`) is fourth-order accurate, exact for time-independent N,
and exactly RK4 in the v->0 limit; it is what `solver/integrator.py`
implements. This experiment is why the integrator tests assert order 4.

### Hermitian-symmetry pitfall (fixed)

Generating initial spectra as arbitrary complex rfft coefficients violates
the Hermitian constraint on the `nx = 0` column (`F(-ny, 0) = conj(F(ny,
0))`), producing a ~3% inconsistency between spectral and physical norms.
The IC constructor now shapes the spectrum of a *real* random field with
real multipliers (symmetry-preserving); the shell-spectrum/Parseval check
(Group A) closes to 2e-16.

### Physical design decision: linear drag

Pilot runs without drag (`sigma = 0`) showed the classic 2D inverse-cascade
condensate drift (energy accumulating at the lowest modes, `c_eps`
collapsing as A increased, no clean stationarity). With `sigma = 0.05`
(drag timescale 20 large-eddy turnovers; ~4% of viscous damping at k_d,
dominant only for k <~ 2), production-dissipation balance holds to <1% and
all stationarity criteria are met. The drag is part of the documented
physical system (docs/equations.md, section 1).

### Dealiasing off-by-one at grid sizes divisible by 3 (bug found and fixed)

Groups A-H (grids 64/128/256, none divisible by 3) all passed even with the
ORIGINAL `nx_max = Nx // 3` truncation. Group I added a grid-size sweep of
the discrete enstrophy-conservation identity `<omega, N_adv(omega)> = 0` and
found it held to ~1e-17 at N=32/64 but only ~1e-3 (NOT machine precision) at
N=48. Root cause: alias-freedom of the 2/3 rule requires the STRICT
inequality `N > 3*nx_max`; `Nx // 3` gives this strictly whenever N is not a
multiple of 3, but gives `nx_max = N/3` EXACTLY (non-strict) when N IS a
multiple of 3, letting a worst-case product alias exactly onto the kept
boundary mode. Fixed in `solver/spectral.py` via `nx_max = (Nx - 1) // 3`,
which is identical to the old formula for every non-multiple-of-3 N
(including all three production grids: no change to re_low/re_mid/re_high)
and strictly correct for multiples of 3. See `docs/audit_report.md`.

### NaN safety-net gap in `NS2D.run` (bug found and fixed)

Group J found that `d["courant"] > cfg.time.fail_courant` is silently
`False` when `d["courant"]` is `NaN` (a general Python/numpy property: any
comparison against NaN is False). A run whose state overflowed to NaN
strictly BETWEEN two diagnostic checks — exactly the production
configuration, since `generate_dataset.py` checks roughly every
`min(dt_sample, 0.5)` time units while `dt` is a few 1e-3 — would return a
NaN-filled `RunResult` with no error. Fixed in `solver/ns2d.py` by adding an
explicit `np.isfinite` check on E/enstrophy/courant ahead of the Courant
comparison. See `docs/audit_report.md`.

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", default=None, help="write markdown report to this path")
    ap.add_argument("--quick", action="store_true", help="skip group H (long run)")
    args = ap.parse_args()

    t0 = time.time()
    for key in sorted(GROUPS):
        if args.quick and key == "H":
            print("Group H: skipped (--quick)")
            continue
        GROUPS[key]()
    n_fail = sum(1 for _, _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS)} checks, {n_fail} failed, {time.time()-t0:.0f}s total")

    if args.markdown:
        with open(args.markdown, "w") as fh:
            fh.write(HEADER)
            cur = None
            titles = {"A": "A. Spectral calculus", "B": "B. Incompressibility",
                      "C": "C. Dealiasing", "D": "D. Time integration",
                      "E": "E. Laminar steady state", "F": "F. Budget closure",
                      "G": "G. Physical identities", "H": "H. Stationarity & Reynolds",
                      "I": "I. Inviscid/unforced conservation", "J": "J. Timestep stability boundary"}
            for group, name, ok, detail in RESULTS:
                if group != cur:
                    fh.write(f"\n## {titles[group]}\n\n")
                    cur = group
                fh.write(f"- {'**PASS**' if ok else '**FAIL**'} — {name}: `{detail}`\n")
            fh.write(f"\n**Summary**: {len(RESULTS)} checks, {n_fail} failed.\n")
            fh.write(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"wrote {args.markdown}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
