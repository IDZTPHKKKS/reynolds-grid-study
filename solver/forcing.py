"""Kolmogorov forcing for 2D Navier-Stokes."""

from __future__ import annotations

import numpy as np

from .spectral import SpectralGrid2D

__all__ = ["KolmogorovForcing"]


class KolmogorovForcing:
    """Monochromatic Kolmogorov forcing ``f = (A sin(k_f y), 0)``."""

    kind = "kolmogorov"

    def __init__(self, amplitude: float, mode: int, grid: SpectralGrid2D):
        if amplitude < 0:
            raise ValueError("Forcing amplitude must be >= 0.")
        if int(mode) != mode or mode < 1:
            raise ValueError("Forcing mode n_f must be a positive integer.")
        if mode > grid.ny_max:
            raise ValueError(
                f"Forcing mode n_f={mode} lies outside the 2/3-rule truncation "
                f"(ny_max={grid.ny_max}); increase Ny or lower n_f."
            )
        self.amplitude = float(amplitude)
        self.mode = int(mode)
        self.grid = grid
        self.k_f = (2.0 * np.pi / grid.Ly) * self.mode

        # Physical-space forcing fields (y-dependence only; broadcast over x).
        self.f_x = self.amplitude * np.sin(self.k_f * grid.y)[:, None] * np.ones((1, grid.Nx))
        self.f_y = np.zeros((grid.Ny, grid.Nx))
        self.f_omega = -self.amplitude * self.k_f * np.cos(self.k_f * grid.y)[:, None] * np.ones(
            (1, grid.Nx)
        )

        shape = (grid.Ny, grid.Nx // 2 + 1)
        self.f_x_hat = np.zeros(shape, dtype=np.complex128)
        self.f_y_hat = np.zeros(shape, dtype=np.complex128)
        self.f_omega_hat = np.zeros(shape, dtype=np.complex128)
        i_pos = np.where(grid.ny == self.mode)[0][0]
        i_neg = np.where(grid.ny == -self.mode)[0][0]
        norm = grid.Ny * grid.Nx / 2.0
        # sin(k_f y): F(k) = -i*N/2 at ny=+n_f, +i*N/2 at ny=-n_f
        self.f_x_hat[i_pos, 0] = -1j * self.amplitude * norm
        self.f_x_hat[i_neg, 0] = +1j * self.amplitude * norm
        # -A k_f cos(k_f y): F(k) = -A k_f * N/2 at both ny = +/- n_f
        self.f_omega_hat[i_pos, 0] = -self.amplitude * self.k_f * norm
        self.f_omega_hat[i_neg, 0] = -self.amplitude * self.k_f * norm

    # ------------------------------------------------------------- diagnostics
    def energy_input(self, u: np.ndarray, v: np.ndarray) -> float:
        """Instantaneous power input per unit area, P = <u ."""
        return float(np.mean(u * self.f_x + v * self.f_y))

    def enstrophy_input(self, omega: np.ndarray) -> float:
        """Instantaneous enstrophy input per unit area, <omega F_omega>."""
        return float(np.mean(omega * self.f_omega))

    # ---------------------------------------------------------------- laminar
    def laminar_velocity(self, nu: float, drag: float = 0.0):
        """Exact steady laminar solution (u_lam, v_lam, omega_lam), including linear
        drag sigma: u_lam = A sin(k_f y)/(nu k_f^2 + sigma) e_x."""
        denom = nu * self.k_f**2 + drag
        u = (self.amplitude / denom) * np.sin(self.k_f * self.grid.y)[:, None] * np.ones(
            (1, self.grid.Nx)
        )
        v = np.zeros_like(u)
        omega = -(self.amplitude * self.k_f / denom) * np.cos(self.k_f * self.grid.y)[:, None] * np.ones(
            (1, self.grid.Nx)
        )
        return u, v, omega

    def reynolds_laminar(self, nu: float, drag: float = 0.0) -> float:
        """Control-parameter Reynolds number Re_lam = U_lam/(nu k_f) with the laminar
        velocity scale U_lam = A/(nu k_f^2 + sigma)."""
        return self.amplitude / ((nu * self.k_f**2 + drag) * nu * self.k_f)

    def describe(self) -> dict:
        return {
            "type": self.kind,
            "amplitude": self.amplitude,
            "mode_n_f": self.mode,
            "k_f": self.k_f,
            "convention": "f = (A sin(k_f y), 0); F_omega = -A k_f cos(k_f y)",
        }
