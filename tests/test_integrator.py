"""Unit tests: phi functions and integrator consistency/order."""

import numpy as np
import pytest

from solver.integrator import RK4, IFRK4, ETDRK4, ETDRK4Coefficients, phi1, phi2, phi3


def test_phi_limits_at_zero():
    v = np.zeros(5)
    assert np.allclose(phi1(v), 1.0)
    assert np.allclose(phi2(v), 0.5)
    assert np.allclose(phi3(v), 1.0 / 6.0)


def test_phi_values_at_one():
    # reference values from the exact closed forms with e = 2.718281828...
    e = np.e
    assert np.isclose(phi1(1.0), e - 1.0)
    assert np.isclose(phi2(1.0), e - 2.0)
    assert np.isclose(phi3(1.0), e - 2.5)


def test_phi_continuity_across_threshold():
    """Taylor branch (|v| < 0.5) and direct branch agree at the seam."""
    from solver.integrator import _phi_taylor
    for j, phi in [(1, phi1), (2, phi2), (3, phi3)]:
        for v0 in (0.5, -0.5, 0.5000001, -0.5000001):
            taylor_val = _phi_taylor(np.array([v0]), j)[0]
            val = phi(np.array([v0]))[0]
            rel = abs(val - taylor_val) / max(abs(val), 1e-3)
            # the two branches must agree to near machine precision at the seam
            assert rel < 1e-13, f"phi{j} branch mismatch at v={v0}: {rel:.2e}"


def test_phi_recurrence_identity():
    """phi_{j+1}(v) = (phi_j(v) - 1/j!)/v  (away from 0)."""
    v = np.linspace(0.7, 3.0, 11)
    assert np.allclose((phi1(v) - 1.0) / v, phi2(v), rtol=1e-12)
    assert np.allclose((phi2(v) - 0.5) / v, phi3(v), rtol=1e-12)


def test_etdrk4_coefficients_reduce_to_rk4():
    """As v = L dt -> 0 the ETDRK4 update becomes classical RK4: Q -> dt/2; f1, f3 ->
    dt/6; f2 -> dt/3."""
    dt = 0.01
    L = np.full((4, 4), -1e-18)          # essentially zero linear operator
    c = ETDRK4Coefficients(dt, L)
    assert np.allclose(c.E, 1.0)
    assert np.allclose(c.Q, dt / 2, rtol=0, atol=1e-14)
    assert np.allclose(c.f1, dt / 6, rtol=0, atol=1e-14)
    assert np.allclose(c.f2, dt / 3, rtol=0, atol=1e-14)
    assert np.allclose(c.f3, dt / 6, rtol=0, atol=1e-14)


def test_etdrk4_weights_sum_matches_constant_forcing_exactness():
    """ETDRK4 is exact for constant forcing: f1 + 2 f2 + f3 = h phi1(v)."""
    dt = 0.1
    for lam in (-0.3, -2.0, -30.0):
        L = np.array([[lam]])
        c = ETDRK4Coefficients(dt, L)
        total = c.f1 + 2 * c.f2 + c.f3
        assert np.isclose(total[0], dt * phi1(np.array([lam * dt]))[0], rtol=1e-12)


@pytest.mark.parametrize("Scheme", [IFRK4, ETDRK4])
def test_linear_decay_is_exact(Scheme):
    """With N = 0, exponential schemes reproduce e^{L t} to machine precision."""
    n = 32
    k = np.fft.fftfreq(n) * n
    L = -0.05 * k**2
    rng = np.random.default_rng(0)
    u0 = rng.standard_normal(n)
    dt, steps = 0.01, 50
    scheme = Scheme(dt, L, lambda u: np.zeros_like(u))
    u = u0.copy()
    for _ in range(steps):
        u = scheme.step(u)
    exact = np.exp(L * dt * steps) * u0
    assert np.allclose(u, exact, rtol=1e-13, atol=1e-14)


def _make_advdiff_ops(n, c, nu):
    k = np.fft.fftfreq(n) * n  # integer wavenumbers on 2*pi-periodic domain
    L = -nu * k**2

    def rhs_n(u_hat):
        return -c * (1j * k * u_hat)      # spectral d/dx

    def rhs_total(u_hat):
        return rhs_n(u_hat) + L * u_hat

    return L, rhs_n, rhs_total


@pytest.mark.parametrize("Scheme", [RK4, IFRK4, ETDRK4])
def test_fourth_order_convergence_advdiff(Scheme):
    """Observed order ~ 4 on the manufactured advection-diffusion problem."""
    n, c, nu = 64, 1.3, 0.02
    L, rhs_n, rhs_total = _make_advdiff_ops(n, c, nu)
    T = 1.0
    x = 2 * np.pi * np.arange(n) / n
    u0_hat = np.fft.fft(np.sin(x))
    exact_hat = np.exp(-nu * T) * np.fft.fft(np.sin(x - c * T))

    dts = [0.02, 0.01, 0.005, 0.0025]
    errs = []
    for dt in dts:
        if Scheme is RK4:
            scheme = Scheme(dt, rhs_total)
        else:
            scheme = Scheme(dt, L, rhs_n)
        u = u0_hat.copy()
        for _ in range(int(round(T / dt))):
            u = scheme.step(u)
        errs.append(np.linalg.norm(u - exact_hat) / np.linalg.norm(exact_hat))
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    mean_order = np.mean(orders)
    assert mean_order > 3.7, f"{Scheme.__name__} order {mean_order:.2f} (errs={errs})"


@pytest.mark.parametrize("SchemeA, SchemeB", [(RK4, IFRK4), (RK4, ETDRK4), (IFRK4, ETDRK4)])
def test_schemes_agree_in_quasi_nonstiff_limit(SchemeA, SchemeB):
    """For |v| = nu k_max^2 dt << 1 (production regime), all schemes must agree closely
    on the nonlinear manufactured problem."""
    n, c, nu = 64, 1.0, 0.02
    L, rhs_n, rhs_total = _make_advdiff_ops(n, c, nu)
    T, dt = 0.5, 0.005
    x = 2 * np.pi * np.arange(n) / n
    u0_hat = np.fft.fft(np.sin(x) + 0.3 * np.cos(3 * x))

    def run(Scheme):
        s = Scheme(dt, rhs_total) if Scheme is RK4 else Scheme(dt, L, rhs_n)
        u = u0_hat.copy()
        for _ in range(int(round(T / dt))):
            u = s.step(u)
        return u

    ua, ub = run(SchemeA), run(SchemeB)
    rel = np.linalg.norm(ua - ub) / np.linalg.norm(ua)
    assert rel < 1e-8, f"{SchemeA.__name__} vs {SchemeB.__name__}: rel diff {rel:.2e}"
