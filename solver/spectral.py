"""Spectral primitives for doubly-periodic 2D fields (pseudo-spectral method)."""

from __future__ import annotations

import os

import numpy as np
import scipy.fft as _fft

_WORKERS = int(os.environ.get("FFT_WORKERS", "1"))

__all__ = ["SpectralGrid2D"]


class SpectralGrid2D:
    """Grid, wavenumbers, FFT wrappers, derivatives and truncation utilities."""

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float):
        if Nx % 2 or Ny % 2:
            raise ValueError("Nx and Ny must be even (Nyquist + 2/3-rule conventions).")
        if Nx < 6 or Ny < 6:
            raise ValueError("Nx, Ny >= 6 required for the 2/3 rule to be meaningful.")
        if Lx <= 0 or Ly <= 0:
            raise ValueError("Lx, Ly must be positive.")

        self.Nx, self.Ny = int(Nx), int(Ny)
        self.Lx, self.Ly = float(Lx), float(Ly)

        # Physical collocation points (cell-centered at 0 boundary of each period).
        self.x = np.arange(Nx) * (Lx / Nx)          # shape (Nx,)
        self.y = np.arange(Ny) * (Ly / Ny)          # shape (Ny,)
        self.dx = Lx / Nx
        self.dy = Ly / Ny

        # Integer mode numbers (rfft layout along x: nx = 0..Nx/2).
        self.nx = np.fft.rfftfreq(Nx, d=1.0 / Nx).astype(int)   # (Nx//2+1,)
        self.ny = np.fft.fftfreq(Ny, d=1.0 / Ny).astype(int)    # (Ny,)

        # Physical wavenumbers.
        self.kx = (2.0 * np.pi / Lx) * self.nx                  # (Nx//2+1,)
        self.ky = (2.0 * np.pi / Ly) * self.ny                  # (Ny,)
        self.KX = self.kx[None, :]                              # broadcastable (Ny, Nx//2+1)
        self.KY = self.ky[:, None]
        self.k2 = self.KY**2 + self.KX**2                       # |k|^2
        self.kmag = np.sqrt(self.k2)
        self.k2_safe = np.where(self.k2 == 0.0, 1.0, self.k2)   # for safe division (k=0 result overridden)

        self.nx_max = (Nx - 1) // 3
        self.ny_max = (Ny - 1) // 3
        keep_y = np.abs(self.ny) <= self.ny_max
        keep_x = self.nx <= self.nx_max                          # nx >= 0 by construction
        self.keep_mask = keep_y[:, None] & keep_x[None, :]
        # Guaranteed-resolved wavenumber in every direction (conservative min).
        self.kmax_x = (2.0 * np.pi / Lx) * self.nx_max
        self.kmax_y = (2.0 * np.pi / Ly) * self.ny_max
        self.k_max = min(self.kmax_x, self.kmax_y)
        if self.nx[-1] <= self.nx_max or (self.ny[0] + Ny) <= self.ny_max:  # pragma: no cover
            raise ValueError("Truncation mask does not remove any mode; N too small.")

        # --- rfft Parseval weights ---------------------------------------------
        w = np.full(self.nx.shape, 2.0)
        w[0] = 1.0
        if Nx % 2 == 0:
            w[-1] = 1.0
        self.rfft_w = w[None, :]                                 # (1, Nx//2+1)

        self.shell_index = np.rint(np.sqrt(self.ny[:, None] ** 2 + self.nx[None, :] ** 2)).astype(int)
        self.n_shells = int(self.shell_index.max()) + 1
        if Lx != Ly:  # pragma: no cover
            # Shell wavenumbers below assume Lx == Ly; forbid silent misuse.
            self._shells_valid = False
        else:
            self._shells_valid = True

    # ------------------------------------------------------------------ FFT --
    def fft(self, f: np.ndarray) -> np.ndarray:
        """Forward transform of a real physical field, shape (Ny, Nx)."""
        return _fft.rfft2(f, workers=_WORKERS)

    def ifft(self, f_hat: np.ndarray) -> np.ndarray:
        """Inverse transform back to the real physical grid."""
        return _fft.irfft2(f_hat, s=(self.Ny, self.Nx), workers=_WORKERS)

    # ----------------------------------------------------------- derivatives --
    def ddx(self, f_hat: np.ndarray) -> np.ndarray:
        """Spectral d/dx of a real field given its rfft coefficients."""
        return 1j * self.KX * f_hat

    def ddy(self, f_hat: np.ndarray) -> np.ndarray:
        """Spectral d/dy of a real field given its rfft coefficients."""
        return 1j * self.KY * f_hat

    def laplacian(self, f_hat: np.ndarray) -> np.ndarray:
        """Spectral Laplacian."""
        return -self.k2 * f_hat

    def inv_laplacian(self, f_hat: np.ndarray) -> np.ndarray:
        """Solve laplacian(g) = f."""
        out = -f_hat / self.k2_safe
        out[0, 0] = 0.0
        return out

    def streamfunction_hat(self, omega_hat: np.ndarray) -> np.ndarray:
        out = omega_hat / self.k2_safe
        out[0, 0] = 0.0
        return out

    def velocity_hat(self, omega_hat: np.ndarray):
        u_hat = 1j * self.KY * (omega_hat / self.k2_safe)
        v_hat = -1j * self.KX * (omega_hat / self.k2_safe)
        u_hat[0, 0] = 0.0
        v_hat[0, 0] = 0.0
        return u_hat, v_hat

    def velocity(self, omega_hat: np.ndarray):
        """Return (u, v, psi) physical fields from vorticity coefficients."""
        psi_hat = self.streamfunction_hat(omega_hat)
        u = self.ifft(1j * self.KY * psi_hat)
        v = self.ifft(-1j * self.KX * psi_hat)
        psi = self.ifft(psi_hat)
        return u, v, psi

    def vorticity_hat_from_velocity(self, u_hat, v_hat) -> np.ndarray:
        return 1j * self.KX * v_hat - 1j * self.KY * u_hat

    def divergence_hat(self, u_hat, v_hat) -> np.ndarray:
        return 1j * (self.KX * u_hat + self.KY * v_hat)

    # ------------------------------------------------------------- dealiasing --
    def dealias(self, f_hat: np.ndarray) -> np.ndarray:
        """Project onto the 2/3-rule truncation (returns new array)."""
        return f_hat * self.keep_mask

    def dealias_inplace(self, f_hat: np.ndarray) -> np.ndarray:
        f_hat *= self.keep_mask
        return f_hat

    # ---------------------------------------------------------------- moments --
    def mean(self, f: np.ndarray) -> float:
        """Domain average of a physical field (equals integral / (Lx*Ly))."""
        return float(f.mean())

    def mean_square_hat(self, f_hat: np.ndarray) -> float:
        """<f^2> from rfft coefficients via Parseval (see module docstring)."""
        return float(np.sum(self.rfft_w * np.abs(f_hat) ** 2) / (self.Nx * self.Ny) ** 2)

    def mean_gradient_magnitude_sq(self, f_hat: np.ndarray) -> float:
        """<|grad f|^2> from rfft coefficients."""
        return float(np.sum(self.rfft_w * self.k2 * np.abs(f_hat) ** 2) / (self.Nx * self.Ny) ** 2)

    # ---------------------------------------------------------------- spectra --
    def shell_spectrum(self, u_hat, v_hat) -> tuple[np.ndarray, np.ndarray]:
        """Shell-averaged kinetic-energy spectrum E(k_n)."""
        if not self._shells_valid:
            raise ValueError("Shell spectrum requires Lx == Ly.")
        weights = self.rfft_w * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)
        spec = 0.5 * np.bincount(
            self.shell_index.ravel(), weights=weights.ravel(), minlength=self.n_shells
        ) / (self.Nx * self.Ny) ** 2
        k = np.arange(self.n_shells) * (2.0 * np.pi / self.Lx)
        return k, spec

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SpectralGrid2D(Nx={self.Nx}, Ny={self.Ny}, Lx={self.Lx:.6g}, Ly={self.Ly:.6g}, "
            f"nx_max={self.nx_max}, ny_max={self.ny_max}, k_max={self.k_max:.6g})"
        )
