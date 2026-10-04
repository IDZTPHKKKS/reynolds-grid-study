"""Unit tests: inviscid, unforced (2D Euler) conservation."""

from __future__ import annotations

import numpy as np
import pytest

from solver.config import (DomainConfig, ForcingConfig, GridConfig, ICConfig,
                           NumericsConfig, SimulationConfig, TimeConfig,
                           ViscosityConfig)
from solver.ns2d import NS2D


def _euler_cfg(dt: float, N: int = 48) -> SimulationConfig:
    return SimulationConfig(
        name="euler_conservation",
        domain=DomainConfig(), grid=GridConfig(Nx=N, Ny=N),
        forcing=ForcingConfig(amplitude=0.0, mode=4),      # UNFORCED
        viscosity=ViscosityConfig(mode="explicit", nu=0.0, drag=0.0),  # INVISCID
        time=TimeConfig(dt=dt, t_total=1.0, dt_sample=1.0),
        numerics=NumericsConfig(integrator="etdrk4"),
        initial_condition=ICConfig(seed=555, u_rms=1.0, k_cut=8),
    )


def test_zero_amplitude_forcing_is_constructible():
    """amplitude=0 must be accepted (it is a physically valid, trivial forcing) and
    must produce exactly zero forcing fields/coefficients."""
    sim = NS2D(_euler_cfg(dt=0.005))
    f = sim.forcing
    assert f.amplitude == 0.0
    assert np.all(f.f_x == 0.0) and np.all(f.f_y == 0.0) and np.all(f.f_omega == 0.0)
    assert np.all(f.f_omega_hat == 0.0)
    with pytest.raises(ValueError):
        from solver.forcing import KolmogorovForcing
        KolmogorovForcing(-1.0, 4, sim.grid)  # still must reject negative


@pytest.mark.parametrize("dt", [0.01, 0.005])
def test_energy_and_enstrophy_conserved_inviscid_unforced(dt):
    """E(t) and Omega(t) must be conserved by the inviscid, unforced, dealiased
    advection term, up to time-integrator truncation error."""
    cfg = _euler_cfg(dt=dt)
    sim = NS2D(cfg)
    sim.set_initial_state(seed=555)
    d0 = sim.state()
    E0, Z0 = d0["E"], d0["enstrophy"]
    assert E0 > 0 and Z0 > 0

    n_steps = int(round(2.0 / dt))   # integrate 2 (large-eddy) time units
    E_hist, Z_hist = [E0], [Z0]
    for _ in range(n_steps):
        sim.step()
        d = sim.state()
        E_hist.append(d["E"])
        Z_hist.append(d["enstrophy"])

    E_hist, Z_hist = np.array(E_hist), np.array(Z_hist)
    drift_E = np.max(np.abs(E_hist - E0)) / E0
    drift_Z = np.max(np.abs(Z_hist - Z0)) / Z0
    assert drift_E < 1e-3, f"energy drift {drift_E:.2e} at dt={dt}"
    assert drift_Z < 1e-3, f"enstrophy drift {drift_Z:.2e} at dt={dt}"


def test_conservation_drift_shrinks_with_smaller_dt():
    """Inviscid conservation drift shrinks with dt (time-stepping error, not a leak)."""
    drifts = {}
    for dt in (0.02, 0.005):
        cfg = _euler_cfg(dt=dt)
        sim = NS2D(cfg)
        sim.set_initial_state(seed=555)
        E0 = sim.state()["E"]
        for _ in range(int(round(1.0 / dt))):
            sim.step()
        E1 = sim.state()["E"]
        drifts[dt] = abs(E1 - E0) / E0
    assert drifts[0.005] < drifts[0.02] / 5.0, drifts


def test_nonlinear_term_is_exactly_skew_orthogonal_to_state():
    """The dealiased advection term conserves enstrophy: <omega, N(omega)> = 0."""
    cfg = _euler_cfg(dt=0.005, N=32)
    sim = NS2D(cfg)
    sim.set_initial_state(seed=9)
    omega_hat = sim.omega_hat
    n_adv_hat = sim.rhs_nonlinear(omega_hat)   # forcing is exactly zero here
    g = sim.grid
    omega = g.ifft(omega_hat)
    n_adv = g.ifft(n_adv_hat)
    inner = float(np.mean(omega * n_adv))      # <omega, N_adv(omega)>
    scale = float(np.mean(omega * omega))
    assert abs(inner) / scale < 1e-10, f"enstrophy-conservation residual {inner:.3e} (scale {scale:.3e})"
