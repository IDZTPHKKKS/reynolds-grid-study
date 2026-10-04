"""Feature construction for the neural cross-Re generalization experiment (Phase 4)."""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from solver.spectral import SpectralGrid2D  # noqa: E402

VARIANTS = ["baseline", "divisive", "conditioned", "quantile", "baseline_rs", "quantile_rs",
            "fno", "fno_rs", "unet", "unet_rs"]
IN_CHANNELS = {"baseline": 1, "divisive": 1, "conditioned": 2,
               "quantile": 1, "baseline_rs": 1, "quantile_rs": 1,
               "fno": 1, "fno_rs": 1, "unet": 1, "unet_rs": 1}

N_REF = int(os.environ.get("FIXED_GRID", 128))
N_QUANTILES = 1024


def spectral_resample(f: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a real periodic (N, N) field to (n_out, n_out) in Fourier space: exact
    zero-padding when upsampling, truncation to |n| < n_out/2 when downsampling."""
    n_in = f.shape[0]
    if n_in == n_out:
        return f
    f_hat = np.fft.rfft2(f)
    m = min(n_in, n_out) // 2  # keep |ny|, nx < m
    out = np.zeros((n_out, n_out // 2 + 1), dtype=complex)
    out[:m, :m] = f_hat[:m, :m]
    out[-(m - 1):, :m] = f_hat[-(m - 1):, :m]
    # backward-normalized rfft2: coefficients scale with the point count
    return np.fft.irfft2(out * (n_out / n_in) ** 2, s=(n_out, n_out))


def _interp_extrap(x, xp, fp):
    """np.interp with linear extrapolation from the end segments."""
    y = np.interp(x, xp, fp)
    lo, hi = x < xp[0], x > xp[-1]
    y[lo] = fp[0] + (x[lo] - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0])
    y[hi] = fp[-1] + (x[hi] - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    return y


def snapshot_quantiles(omega: np.ndarray):
    """(values, gaussian_scores): the snapshot's own quantile function, sampled at
    N_QUANTILES points."""
    from scipy.special import ndtri
    p = (np.arange(N_QUANTILES) + 0.5) / N_QUANTILES
    vals = np.quantile(omega.ravel(), p)
    vals, idx = np.unique(vals, return_index=True)
    return vals, ndtri(p[idx])


def to_scores(field, q):
    return _interp_extrap(field.ravel(), q[0], q[1]).reshape(field.shape)


def from_scores(z, q):
    return _interp_extrap(z.ravel(), q[1], q[0]).reshape(z.shape)


def _split_variant(variant: str):
    """'quantile_rs' -> ('quantile', True); 'baseline' -> ('baseline', False)."""
    return (variant[:-3], True) if variant.endswith("_rs") else (variant, False)


def local_zeta_and_tau(grid: SpectralGrid2D, nu: float, omega: np.ndarray):
    """Local enstrophy dissipation zeta = nu |grad omega|^2 and its time scale zeta^(-1/3)."""
    omega_hat = grid.fft(omega)
    dox = grid.ifft(grid.ddx(omega_hat))
    doy = grid.ifft(grid.ddy(omega_hat))
    grad_omega_sq = dox**2 + doy**2
    zeta = nu * grad_omega_sq
    zeta_floor = 1e-8 * max(float(zeta.max()), 1e-300)
    zeta_safe = np.maximum(zeta, zeta_floor)
    tau = zeta_safe ** (-1.0 / 3.0)
    return zeta, tau


def make_input(variant: str, grid: SpectralGrid2D, nu: float, omega: np.ndarray) -> np.ndarray:
    """Return the network's input array, shape (IN_CHANNELS[variant], Ny, Nx)."""
    base, rs = _split_variant(variant)
    if base in ("fno", "unet") and not rs:
        variant = "baseline"
    if base == "quantile" or rs:
        w = spectral_resample(omega, N_REF) if rs else omega
        if base == "quantile":
            w = to_scores(w, snapshot_quantiles(w))
        return w[None, :, :].astype(np.float32)
    if variant == "baseline":
        return omega[None, :, :].astype(np.float32)
    zeta, tau = local_zeta_and_tau(grid, nu, omega)
    if variant == "divisive":
        return (omega * tau)[None, :, :].astype(np.float32)
    if variant == "conditioned":
        zeta_ch = np.log1p(zeta).astype(np.float32)
        return np.stack([omega.astype(np.float32), zeta_ch], axis=0)
    raise ValueError(variant)


def make_target(variant: str, grid: SpectralGrid2D, nu: float, omega_next: np.ndarray,
                omega_now: np.ndarray) -> np.ndarray:
    """Training target in the same units as the input variant."""
    base, rs = _split_variant(variant)
    if base in ("fno", "unet") and not rs:
        variant = "baseline"
    if base == "quantile" or rs:
        now = spectral_resample(omega_now, N_REF) if rs else omega_now
        nxt = spectral_resample(omega_next, N_REF) if rs else omega_next
        if base == "quantile":
            nxt = to_scores(nxt, snapshot_quantiles(now))  # CDF of the INPUT state
        return nxt[None, :, :].astype(np.float32)
    if variant in ("baseline", "conditioned"):
        return omega_next[None, :, :].astype(np.float32)
    if variant == "divisive":
        _, tau = local_zeta_and_tau(grid, nu, omega_now)
        return (omega_next * tau)[None, :, :].astype(np.float32)
    raise ValueError(variant)


def invert_prediction(variant: str, grid: SpectralGrid2D, nu: float,
                      pred: np.ndarray, omega_now: np.ndarray) -> np.ndarray:
    """Map a network prediction (in whatever units `make_target` used) back to physical
    vorticity units, for rollout and evaluation."""
    base, rs = _split_variant(variant)
    if base in ("fno", "unet") and not rs:
        variant = "baseline"
    if base == "quantile" or rs:
        out = pred[0].astype(np.float64)
        if base == "quantile":
            now = spectral_resample(omega_now, N_REF) if rs else omega_now
            out = from_scores(out, snapshot_quantiles(now))
        return spectral_resample(out, grid.Nx) if rs else out
    if variant in ("baseline", "conditioned"):
        return pred[0]
    if variant == "divisive":
        _, tau = local_zeta_and_tau(grid, nu, omega_now)
        return pred[0] / tau
    raise ValueError(variant)
