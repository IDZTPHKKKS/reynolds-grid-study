"""Diagnostics for 2D forced turbulence: integral quantities, budgets, Reynolds
numbers, spectra, gradient fields and stationarity testing."""

from __future__ import annotations

import numpy as np

from .forcing import KolmogorovForcing
from .spectral import SpectralGrid2D

__all__ = [
    "state_diagnostics",
    "gradient_fields",
    "reynolds_numbers",
    "courant_number",
    "stationarity_report",
]


# ------------------------------------------------------------ state level --
def state_diagnostics(grid: SpectralGrid2D, forcing: KolmogorovForcing, nu: float,
                      omega_hat: np.ndarray, drag: float = 0.0) -> dict:
    """Scalar diagnostics of the current state (all per unit area)."""
    u, v, psi = grid.velocity(omega_hat)
    omega = grid.ifft(omega_hat)
    u_hat, v_hat = grid.velocity_hat(omega_hat)

    E = 0.5 * grid.mean(u * u + v * v)
    Om = 0.5 * grid.mean(omega * omega)
    P_in = forcing.energy_input(u, v)
    Pin_om = forcing.enstrophy_input(omega)
    pal = 0.5 * grid.mean_gradient_magnitude_sq(omega_hat)
    div_hat = grid.divergence_hat(u_hat, v_hat)
    max_div = float(np.max(np.abs(grid.ifft(div_hat))))

    eps_visc = 2.0 * nu * Om
    eps_drag = 2.0 * drag * E
    eps_om_visc = 2.0 * nu * pal
    eps_om_drag = 2.0 * drag * Om

    return {
        "E": E,
        "enstrophy": Om,
        "P_in": P_in,
        "enstrophy_input": Pin_om,
        "eps": eps_visc + eps_drag,          # total mean energy dissipation rate
        "eps_visc": eps_visc,
        "eps_drag": eps_drag,
        "eps_omega": eps_om_visc + eps_om_drag,   # total enstrophy dissipation
        "eps_omega_visc": eps_om_visc,
        "palinstrophy": pal,
        "u_rms": float(np.sqrt(2.0 * E)),
        "u_max": float(np.max(np.sqrt(u * u + v * v))),
        "max_divergence": max_div,
        "omega_rms": float(np.sqrt(2.0 * Om)),
    }


# ------------------------------------------------------- gradient fields --
def gradient_fields(grid: SpectralGrid2D, nu: float, omega_hat: np.ndarray) -> dict:
    """Pointwise gradient-tensor quantities needed by the (future) local similarity
    analysis and by the Fukami-et-al-style collapse diagnostics."""
    u_hat, v_hat = grid.velocity_hat(omega_hat)
    dux, duy = grid.ifft(grid.ddx(u_hat)), grid.ifft(grid.ddy(u_hat))
    dvx, dvy = grid.ifft(grid.ddx(v_hat)), grid.ifft(grid.ddy(v_hat))

    Sxx, Syy = dux, dvy
    Sxy = 0.5 * (duy + dvx)
    S_colon_S = Sxx**2 + Syy**2 + 2.0 * Sxy**2
    omega = grid.ifft(omega_hat)

    dox_hat = grid.ddx(omega_hat)
    doy_hat = grid.ddy(omega_hat)
    grad_omega_sq = grid.ifft(dox_hat) ** 2 + grid.ifft(doy_hat) ** 2

    Q = 0.25 * omega**2 - 0.5 * S_colon_S
    R = -(dux * dvy - duy * dvx)

    eps_x = 2.0 * nu * S_colon_S
    zeta_x = nu * grad_omega_sq
    eps_floor = 1e-12 * max(eps_x.max(), 1e-300)
    zeta_floor = 1e-12 * max(zeta_x.max(), 1e-300)
    eta_eps = (nu**3 / np.maximum(eps_x, eps_floor)) ** 0.25
    eta_zeta = (nu**3 / np.maximum(zeta_x, zeta_floor)) ** (1.0 / 6.0)

    return {
        "dux": dux, "duy": duy, "dvx": dvx, "dvy": dvy,
        "Sxx": Sxx, "Syy": Syy, "Sxy": Sxy,
        "Q": Q, "R": R,
        "omega": omega,
        "eps_x": eps_x, "zeta_x": zeta_x,
        "eta_eps": eta_eps, "eta_zeta": eta_zeta,
        "grad_omega_sq": grad_omega_sq,
    }


# -------------------------------------------------------- reynolds numbers --
def reynolds_numbers(grid: SpectralGrid2D, forcing: KolmogorovForcing, nu: float,
                     diag: dict, drag: float = 0.0) -> dict:
    """Reynolds numbers and turbulence scales, *measured* from the state."""
    E, Om, eps, eps_om = diag["E"], diag["enstrophy"], diag["eps"], diag["eps_omega"]
    u_rms = diag["u_rms"]
    k_f = forcing.k_f
    L = np.sqrt(grid.Lx * grid.Ly)

    lam = np.sqrt(E / Om) if Om > 0 else np.nan
    k_d = (eps_om / nu**3) ** (1.0 / 6.0) if eps_om > 0 else np.nan

    return {
        "Re_f": u_rms / (nu * k_f),
        "Re_lam": forcing.reynolds_laminar(nu, drag),
        "Re_L": u_rms * L / nu,
        "Re_lambda": u_rms * lam / nu,
        "taylor_lambda": lam,
        "k_d": k_d,
        "eta_d": 1.0 / k_d if k_d > 0 else np.nan,
        "k_max_over_k_d": grid.k_max / k_d if k_d > 0 else np.nan,
        "k_max_times_eta": grid.k_max / k_d if k_d > 0 else np.nan,
        "c_eps": eps / (u_rms**3 * k_f) if u_rms > 0 else np.nan,
    }


def courant_number(u: np.ndarray, v: np.ndarray, kmax_x: float, kmax_y: float,
                   dt: float) -> float:
    """Advective spectral Courant number ``C = dt * max_{x,y} ( |u| k_max,x + |v|
    k_max,y )``."""
    return float(dt * np.max(np.abs(u) * kmax_x + np.abs(v) * kmax_y))


# ------------------------------------------------------------ stationarity --
def stationarity_report(t: np.ndarray, energy: np.ndarray, enstrophy: np.ndarray,
                        power_in: np.ndarray, eps: np.ndarray,
                        drift_e: float, drift_z: float, balance: float) -> dict:
    """Assess statistical stationarity of a diagnostic time series."""
    def rel_drift(x):
        h = len(x) // 2
        m = np.mean(x)
        return abs(np.mean(x[h:]) - np.mean(x[:h])) / m if m > 0 else np.inf

    dE = rel_drift(energy)
    dZ = rel_drift(enstrophy)
    mP, mEps = np.mean(power_in), np.mean(eps)
    bal = abs(mP - mEps) / mP if mP > 0 else np.inf
    return {
        "drift_E": float(dE), "drift_enstrophy": float(dZ),
        "production_dissipation_mismatch": float(bal),
        "mean_E": float(np.mean(energy)), "mean_enstrophy": float(np.mean(enstrophy)),
        "mean_P_in": float(mP), "mean_eps": float(mEps),
        "stationary": bool(dE < drift_e and dZ < drift_z and bal < balance),
        "thresholds": {"drift_e": drift_e, "drift_z": drift_z, "balance": balance},
    }
