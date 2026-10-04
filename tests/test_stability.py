"""Unit tests: timestep stability boundary and the NaN/Inf safety net."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from solver.config import (DomainConfig, ForcingConfig, GridConfig, ICConfig,
                           NumericsConfig, SimulationConfig, TimeConfig,
                           ViscosityConfig)
from solver.ns2d import NS2D


def _cfg(dt, nu, integrator="rk4", N=64, fail_courant=2.0, t_total=5.0):
    return SimulationConfig(
        name="stability", domain=DomainConfig(), grid=GridConfig(Nx=N, Ny=N),
        forcing=ForcingConfig(amplitude=1.0, mode=4),
        viscosity=ViscosityConfig(mode="explicit", nu=nu, drag=0.0),
        time=TimeConfig(dt=dt, t_total=t_total, dt_sample=1.0, fail_courant=fail_courant),
        numerics=NumericsConfig(integrator=integrator),
        initial_condition=ICConfig(seed=1, u_rms=1.0, k_cut=8),
    )


def test_dt_inside_diffusive_limit_is_stable_rk4():
    """RK4 stays bounded below the diffusive stability limit."""
    N = 64
    nu = 0.02
    dt = 0.004
    sim = NS2D(_cfg(dt, nu, integrator="rk4", N=N, t_total=2.0))
    sim.set_initial_state(seed=1)
    res = sim.run(2.0, diagnostics_interval=0.1)
    E = np.array(res.diagnostics["E"])
    assert np.all(np.isfinite(E))
    assert E.max() < 100.0   # generously bounded; no blow-up


def test_dt_beyond_diffusive_limit_diverges_rk4():
    """RK4 diverges beyond the diffusive stability limit."""
    N = 64
    nu = 2.0
    dt = 0.2
    sim = NS2D(_cfg(dt, nu, integrator="rk4", N=N, t_total=20.0, fail_courant=2.0))
    sim.set_initial_state(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # expected overflow
        with pytest.raises(RuntimeError):
            sim.run(20.0, diagnostics_interval=20 * dt)  # coarse: checks span the blow-up


def test_etdrk4_linear_part_unconditionally_stable_even_at_large_dt():
    """ETDRK4 stays stable at a time step where RK4 diverges."""
    N = 64
    nu = 2.0
    dt = 0.2
    cfg = _cfg(dt, nu, integrator="etdrk4", N=N, t_total=2.0)
    cfg.forcing.amplitude = 0.0
    sim = NS2D(cfg)
    sim.set_initial_state(seed=1, u_rms=0.1)  # small IC so advective CFL isn't the limiter here
    res = sim.run(2.0, diagnostics_interval=0.2)
    E = np.array(res.diagnostics["E"])
    assert np.all(np.isfinite(E))
    # strongly damped (nu=2.0 is huge): energy must have decayed, not grown
    assert E[-1] < E[0]


def test_nan_state_is_caught_even_with_coarse_diagnostics_interval():
    """A NaN that appears between diagnostic checks is still caught by run()."""
    N = 64
    nu = 2.0
    dt = 0.2
    sim = NS2D(_cfg(dt, nu, integrator="rk4", N=N, t_total=20.0, fail_courant=2.0))
    sim.set_initial_state(seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with pytest.raises(RuntimeError, match="Non-finite diagnostics"):
            sim.run(20.0, diagnostics_interval=20 * dt)


def test_fail_courant_still_triggers_on_large_but_finite_courant():
    """The finite-Courant safety check still triggers."""
    N = 64
    nu = 0.5
    dt = 0.05
    sim = NS2D(_cfg(dt, nu, integrator="rk4", N=N, t_total=5.0, fail_courant=2.0))
    sim.set_initial_state(seed=1)
    with pytest.raises(RuntimeError, match="Courant number"):
        sim.run(5.0, diagnostics_interval=dt)
