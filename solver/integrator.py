"""Time integrators for the semi-discrete vorticity equation."""

from __future__ import annotations

import numpy as np

__all__ = ["phi1", "phi2", "phi3", "RK4", "IFRK4", "ETDRK4", "ETDRK4Coefficients"]


# --------------------------------------------------------------------- phi --
def _phi_taylor(v: np.ndarray, j: int, nterms: int = 20) -> np.ndarray:
    """phi_j(v) = sum_{m=0}^{nterms} v^m / (j+m)!"""
    from math import factorial

    s = np.full_like(v, 1.0 / factorial(j + nterms), dtype=v.dtype)
    for m in range(nterms - 1, -1, -1):
        s = 1.0 / factorial(j + m) + v * s
    return s


def _phi_direct_and_taylor(v, j, direct_fn):
    v = np.asarray(v, dtype=float)
    out = _phi_taylor(v, j)
    mask = np.abs(v) >= 0.5
    if np.any(mask):
        with np.errstate(divide="ignore", invalid="ignore"):
            direct = direct_fn(v)
        out = np.where(mask, direct, out)
    return out


def phi1(v):
    """phi_1(v) = (e^v - 1)/v,  phi_1(0) = 1."""
    v = np.asarray(v, dtype=float)
    direct = lambda z: np.expm1(z) / z
    return _phi_direct_and_taylor(v, 1, direct)


def phi2(v):
    """phi_2(v) = (e^v - 1 - v)/v^2,  phi_2(0) = 1/2."""
    v = np.asarray(v, dtype=float)
    direct = lambda z: (np.expm1(z) - z) / z**2
    return _phi_direct_and_taylor(v, 2, direct)


def phi3(v):
    """phi_3(v) = (e^v - 1 - v - v^2/2)/v^3,  phi_3(0) = 1/6."""
    v = np.asarray(v, dtype=float)
    direct = lambda z: (np.expm1(z) - z - z**2 / 2.0) / z**3
    return _phi_direct_and_taylor(v, 3, direct)


# ------------------------------------------------------------------- RK4 ----
class RK4:
    """Classical explicit RK4 on the full RHS (linear part explicit)."""

    def __init__(self, dt: float, rhs):
        self.dt = float(dt)
        self.rhs = rhs
        self.n_rhs_evals = 0

    def _f(self, u):
        self.n_rhs_evals += 1
        return self.rhs(u)

    def step(self, u: np.ndarray) -> np.ndarray:
        dt = self.dt
        k1 = self._f(u)
        k2 = self._f(u + (dt / 2) * k1)
        k3 = self._f(u + (dt / 2) * k2)
        k4 = self._f(u + dt * k3)
        return u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------------------------------------------- IFRK4 ----
class IFRK4:
    """Integrating-factor RK4: exact linear propagation + RK4 in w = e^{-Lt}u."""

    def __init__(self, dt: float, L_hat: np.ndarray, rhs_N):
        self.dt = float(dt)
        self.L = L_hat
        self.rhs_N = rhs_N
        self.Eh = np.exp(L_hat * dt)
        self.Eh2 = np.exp(L_hat * dt / 2)
        self.n_rhs_evals = 0

    def _n(self, u):
        self.n_rhs_evals += 1
        return self.rhs_N(u)

    def step(self, u: np.ndarray) -> np.ndarray:
        dt, E, E2 = self.dt, self.Eh, self.Eh2
        a1 = self._n(u)
        u2 = E2 * (u + 0.5 * dt * a1)
        a2 = self._n(u2)
        u3 = E2 * u + 0.5 * dt * a2
        a3 = self._n(u3)
        u4 = E * u + dt * E2 * a3
        a4 = self._n(u4)
        return E * u + (dt / 6) * (E * a1 + 2 * E2 * a2 + 2 * E2 * a3 + a4)


# ---------------------------------------------------------------- ETDRK4 ----
class ETDRK4Coefficients:
    """Precomputed ETDRK4 coefficient arrays for a diagonal operator."""

    def __init__(self, dt: float, L_hat: np.ndarray):
        h = float(dt)
        v = h * L_hat
        self.dt, self.v = h, v
        self.E = np.exp(v)
        self.E2 = np.exp(v / 2)
        p1, p2, p3 = phi1(v), phi2(v), phi3(v)
        self.Q = 0.5 * h * phi1(v / 2)
        self.f1 = h * (p1 - 3 * p2 + 4 * p3)
        self.f2 = 2 * h * (p2 - 2 * p3)
        self.f3 = h * (4 * p3 - p2)


class ETDRK4:
    """ETDRK4 (Cox & Matthews 2002; Kassam & Trefethen 2005 form)."""

    def __init__(self, dt: float, L_hat: np.ndarray, rhs_N):
        self.dt = float(dt)
        self.rhs_N = rhs_N
        self.coeffs = ETDRK4Coefficients(dt, L_hat)
        self.n_rhs_evals = 0

    def _n(self, u):
        self.n_rhs_evals += 1
        return self.rhs_N(u)

    def step(self, u: np.ndarray) -> np.ndarray:
        c = self.coeffs
        E, E2, Q, f1, f2, f3 = c.E, c.E2, c.Q, c.f1, c.f2, c.f3
        nu = self._n(u)
        a = E2 * u + Q * nu
        na = self._n(a)
        b = E2 * u + Q * na
        nb = self._n(b)
        cc = E2 * a + Q * (2 * nb - nu)
        nc = self._n(cc)
        return E * u + f1 * nu + f2 * (na + nb) + f3 * nc


INTEGRATORS = {"rk4": RK4, "ifrk4": IFRK4, "etdrk4": ETDRK4}


def make_integrator(name: str, dt: float, L_hat: np.ndarray, rhs_N, rhs_total=None):
    """Factory."""
    name = name.lower()
    if name not in INTEGRATORS:
        raise ValueError(f"Unknown integrator '{name}'. Options: {sorted(INTEGRATORS)}.")
    if name == "rk4":
        if rhs_total is None:
            raise ValueError("RK4 requires rhs_total (full RHS including linear part).")
        return RK4(dt, rhs_total)
    return INTEGRATORS[name](dt, L_hat, rhs_N)
