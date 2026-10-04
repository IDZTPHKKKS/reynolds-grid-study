"""Decaying 1D viscous Burgers equation on [0, 2pi): pseudo-spectral, 2/3 dealiasing, ETDRK4."""
from __future__ import annotations

import numpy as np

L = 2 * np.pi


def etdrk4_coeffs(lin, dt, m=32):
    """Kassam-Trefethen ETDRK4 coefficients for a diagonal linear operator."""
    e, e2 = np.exp(dt * lin), np.exp(dt * lin / 2)
    r = np.exp(1j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
    lr = dt * lin[:, None] + r[None, :]
    q = dt * np.real(np.mean((np.exp(lr / 2) - 1) / lr, axis=1))
    f1 = dt * np.real(np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr ** 2)) / lr ** 3, axis=1))
    f2 = dt * np.real(np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr ** 3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * lr - lr ** 2 + np.exp(lr) * (4 - lr)) / lr ** 3, axis=1))
    return e, e2, q, f1, f2, f3


class Burgers1D:
    def __init__(self, n: int, nu: float, dt: float):
        self.n, self.nu, self.dt = n, nu, dt
        self.k = np.fft.rfftfreq(n, d=1.0 / n)
        self.keep = self.k <= (n - 1) // 3
        self.coef = etdrk4_coeffs(-nu * self.k ** 2, dt)

    def nonlinear(self, uh):
        u = np.fft.irfft(uh, n=self.n)
        return -0.5j * self.k * np.fft.rfft(u * u) * self.keep

    def step(self, uh):
        e, e2, q, f1, f2, f3 = self.coef
        nv = self.nonlinear(uh)
        a = e2 * uh + q * nv
        na = self.nonlinear(a)
        b = e2 * uh + q * na
        nb = self.nonlinear(b)
        c = e2 * a + q * (2 * nb - nv)
        nc = self.nonlinear(c)
        return (e * uh + f1 * nv + 2 * f2 * (na + nb) + f3 * nc) * self.keep


def random_ic(n_traj: int, n: int, rng, k_max: int = 4):
    x = np.arange(n) * L / n
    u = np.zeros((n_traj, n))
    for k in range(1, k_max + 1):
        amp = rng.standard_normal((n_traj, 1)) / k
        u += amp * np.cos(k * x[None, :] + rng.uniform(0, 2 * np.pi, (n_traj, 1)))
    return u / u.std(axis=1, keepdims=True)


def simulate(n: int, nu: float, n_traj: int, n_samples: int, dt_sample: float, seed: int):
    """Trajectories of shape (n_traj, n_samples + 1, n), sampled every dt_sample."""
    dt = min(0.2 * (L / n) / 3.0, dt_sample / 10)
    sub = int(np.ceil(dt_sample / dt))
    solver = Burgers1D(n, nu, dt_sample / sub)
    uh = np.fft.rfft(random_ic(n_traj, n, np.random.default_rng(seed)), axis=-1) * solver.keep
    out = np.empty((n_traj, n_samples + 1, n))
    out[:, 0] = np.fft.irfft(uh, n=n)
    for i in range(1, n_samples + 1):
        for _ in range(sub):
            uh = solver.step(uh)
        out[:, i] = np.fft.irfft(uh, n=n)
    if not np.isfinite(out).all():
        raise RuntimeError("Burgers simulation diverged")
    return out


def resample(u, n_out):
    """Fourier resampling of real periodic fields along the last axis."""
    n_in = u.shape[-1]
    if n_in == n_out:
        return u
    uh = np.fft.rfft(u, axis=-1)
    m = min(n_in, n_out) // 2
    out = np.zeros(u.shape[:-1] + (n_out // 2 + 1,), dtype=complex)
    out[..., :m] = uh[..., :m]
    return np.fft.irfft(out * (n_out / n_in), n=n_out, axis=-1)
