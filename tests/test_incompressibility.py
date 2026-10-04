"""Tests: the reconstructed velocity field is divergence-free."""

import numpy as np
import pytest

from solver.config import load_config
from solver.ns2d import NS2D


@pytest.fixture(scope="module")
def small_config(tmp_path_factory):
    import yaml
    cfg = {
        "name": "test_incompress",
        "domain": {"Lx": 2 * np.pi, "Ly": 2 * np.pi},
        "grid": {"Nx": 32, "Ny": 32},
        "forcing": {"type": "kolmogorov", "amplitude": 0.1, "mode": 4},
        "viscosity": {"mode": "explicit", "nu": 0.02},
        "time": {"dt": 0.002, "t_total": 1.0, "dt_sample": 1.0},
        "numerics": {"dealias": "2/3", "integrator": "etdrk4"},
        "initial_condition": {"type": "band_limited_random", "u_rms": 0.5, "k_cut": 6, "seed": 7},
        "calibration": {"enabled": False},
        "stationarity": {"min_burn": 1, "window": 1, "chunk": 1, "max_burn": 2},
        "output": {"n_traj": 1, "dtype": "float64", "directory": "data"},
    }
    p = tmp_path_factory.mktemp("cfg") / "test_incompress.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return load_config(str(p))


@pytest.mark.parametrize("integrator", ["etdrk4", "ifrk4", "rk4"])
def test_divergence_free_state(small_config, integrator):
    """Velocity reconstructed from a random band-limited vorticity field is divergence-
    free to machine precision."""
    cfg = small_config
    cfg.numerics.integrator = integrator
    sim = NS2D(cfg)
    sim.set_initial_state(seed=11)
    u_hat, v_hat = sim.grid.velocity_hat(sim.omega_hat)
    div_hat = sim.grid.divergence_hat(u_hat, v_hat)
    max_div = np.max(np.abs(sim.grid.ifft(div_hat)))
    scale = np.max(np.abs(u_hat)) * sim.grid.k_max
    assert max_div < 1e-13 * max(scale, 1.0)


@pytest.mark.parametrize("integrator", ["etdrk4", "ifrk4", "rk4"])
def test_incompressibility_preserved_under_evolution(small_config, integrator):
    """After 300 forced steps, div(u) remains at machine zero and the k=0 vorticity
    mode remains exactly zero (mean-vorticity conservation)."""
    cfg = small_config
    cfg.numerics.integrator = integrator
    sim = NS2D(cfg)
    sim.set_initial_state(seed=12)
    u_scale = sim.state()["u_max"]
    for _ in range(300):
        sim.step()
    u_hat, v_hat = sim.grid.velocity_hat(sim.omega_hat)
    max_div = np.max(np.abs(sim.grid.ifft(sim.grid.divergence_hat(u_hat, v_hat))))
    assert max_div < 1e-13 * max(u_scale * sim.grid.k_max, 1.0)
    assert sim.omega_hat[0, 0] == 0.0
    # vorticity is still exactly inside the Galerkin truncation
    assert np.all(sim.omega_hat[~sim.grid.keep_mask] == 0.0)


def test_streamfunction_solves_poisson_exactly(small_config):
    """omega + lap(psi) = 0 at machine precision for the evolved state."""
    sim = NS2D(small_config)
    sim.set_initial_state(seed=13)
    for _ in range(50):
        sim.step()
    omega_hat = sim.omega_hat
    psi_hat = sim.grid.streamfunction_hat(omega_hat)
    residual = omega_hat + sim.grid.laplacian(psi_hat)
    residual[0, 0] = 0.0
    assert np.max(np.abs(residual)) < 1e-13 * max(np.max(np.abs(omega_hat)), 1.0)


def test_nonlinear_term_has_zero_mean(small_config):
    """The conservative-form nonlinear term has exactly zero k=0 mode (mean vorticity
    is not forced), even though the forcing is present."""
    sim = NS2D(small_config)
    sim.set_initial_state(seed=14)
    nl = sim.rhs_nonlinear(sim.omega_hat)
    assert nl[0, 0] == 0.0
    # and the pure (non-forcing) part also has zero mean
    nl_adv = nl - sim.forcing.f_omega_hat
    assert nl_adv[0, 0] == 0.0


def test_nonlinear_term_support_inside_truncation(small_config):
    sim = NS2D(small_config)
    sim.set_initial_state(seed=15)
    nl = sim.rhs_nonlinear(sim.omega_hat)
    assert np.all(nl[~sim.grid.keep_mask] == 0.0)
