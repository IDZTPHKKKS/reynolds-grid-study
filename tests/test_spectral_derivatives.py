"""Unit tests: spectral derivatives, Parseval, and related identities."""

import numpy as np
import pytest

from solver.spectral import SpectralGrid2D


GRIDS = [(16, 16), (32, 24)]


@pytest.mark.parametrize("Nx,Ny", GRIDS)
def test_fft_roundtrip(Nx, Ny):
    g = SpectralGrid2D(Nx, Ny, 2 * np.pi, 2 * np.pi)
    f = np.random.default_rng(0).standard_normal((Ny, Nx))
    assert np.allclose(g.ifft(g.fft(f)), f, atol=1e-13)


@pytest.mark.parametrize("Nx,Ny", GRIDS)
def test_first_derivative_exact_on_band_limited_field(Nx, Ny):
    """d/dx of a trigonometric polynomial is exact for band-limited fields."""
    g = SpectralGrid2D(Nx, Ny, 2 * np.pi, 2 * np.pi)
    x, y = g.x[None, :], g.y[:, None]
    # modes well inside the 2/3 truncation
    f = (3.0 * np.sin(2 * x + 1.0) * np.cos(3 * y)
         + 0.5 * np.cos(5 * x - 2 * y)
         + 2.0 * np.sin(y))
    dfdx = (6.0 * np.cos(2 * x + 1.0) * np.cos(3 * y)
            - 2.5 * np.sin(5 * x - 2 * y))
    dfdy = (-9.0 * np.sin(2 * x + 1.0) * np.sin(3 * y)
            + 1.0 * np.sin(5 * x - 2 * y)
            + 2.0 * np.cos(y))
    f_hat = g.fft(f)
    assert np.allclose(g.ifft(g.ddx(f_hat)), dfdx, atol=1e-12)
    assert np.allclose(g.ifft(g.ddy(f_hat)), dfdy, atol=1e-12)


@pytest.mark.parametrize("Nx,Ny", GRIDS)
def test_second_derivative_and_laplacian(Nx, Ny):
    g = SpectralGrid2D(Nx, Ny, 2 * np.pi, 2 * np.pi)
    x, y = g.x[None, :], g.y[:, None]
    f = np.sin(2 * x) * np.cos(y) + 0.3 * np.cos(4 * y)
    exact_lap = -((4 + 1) * np.sin(2 * x) * np.cos(y) + 0.3 * 16 * np.cos(4 * y))
    f_hat = g.fft(f)
    assert np.allclose(g.ifft(g.laplacian(f_hat)), exact_lap, atol=1e-12)
    # ddx(ddx(f)) == lap on y-independent part (note: broadcast to full grid!)
    f2 = (np.sin(3 * x) + np.cos(x)) * np.ones((Ny, 1))
    f2_hat = g.fft(f2)
    assert np.allclose(g.ifft(g.ddx(g.ddx(f2_hat))), (-9 * np.sin(3 * x) - np.cos(x)) * np.ones((Ny, 1)), atol=1e-12)


def test_parseval_domain_average():
    """<f^2> from rfft coefficients equals the physical-space grid average."""
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    rng = np.random.default_rng(1)
    for _ in range(3):
        f = rng.standard_normal((g.Ny, g.Nx))
        f_hat = g.fft(f)
        assert np.isclose(g.mean_square_hat(f_hat), np.mean(f**2), rtol=1e-12)


def test_parseval_for_spectral_gradient():
    """<|grad f|^2> spectrally == physical-space sum of squared derivatives."""
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    x, y = g.x[None, :], g.y[:, None]
    f = np.sin(2 * x + 3 * y) + 0.5 * np.cos(x)
    f_hat = g.fft(f)
    fx = g.ifft(g.ddx(f_hat))
    fy = g.ifft(g.ddy(f_hat))
    assert np.isclose(g.mean_gradient_magnitude_sq(f_hat),
                      np.mean(fx**2 + fy**2), rtol=1e-12)


def test_inv_laplacian_solves_poisson():
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    x, y = g.x[None, :], g.y[:, None]
    f = np.sin(3 * x + 2 * y) - 0.5 * np.cos(4 * y)          # zero mean
    f_hat = g.fft(f)
    g_hat = g.inv_laplacian(f_hat)
    assert np.allclose(g.ifft(g.laplacian(g_hat)), f, atol=1e-12)
    assert g_hat[0, 0] == 0.0                                  # gauge: zero mean


def test_streamfunction_velocity_roundtrip():
    """psi = omega/k^2 and u = dpsi/dy, v = -dpsi/dx invert each other."""
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    rng = np.random.default_rng(2)
    omega_hat = g.dealias(g.fft(rng.standard_normal((g.Ny, g.Nx))))
    omega_hat[0, 0] = 0.0
    omega = g.ifft(omega_hat)
    # recovered vorticity from the reconstructed velocity equals the input
    u_hat, v_hat = g.velocity_hat(omega_hat)
    omega_back_hat = g.vorticity_hat_from_velocity(u_hat, v_hat)
    assert np.allclose(g.ifft(omega_back_hat), omega, atol=1e-12)


def test_anisotropic_wavenumbers():
    """kx, ky carry physical units 2 pi n / L even away from L = 2 pi."""
    Lx, Ly = 3.0, 5.0
    g = SpectralGrid2D(24, 32, Lx, Ly)
    assert np.isclose(g.kx[1], 2 * np.pi / Lx)
    assert np.isclose(g.ky[1], 2 * np.pi / Ly)
    assert np.isclose(g.ky[-1], -2 * np.pi / Ly)  # fftfreq layout: last is -1
    # verify against an analytic derivative on the anisotropic grid
    x, y = g.x[None, :], g.y[:, None]
    f = np.sin(2 * np.pi / Lx * 2 * x) * np.cos(2 * np.pi / Ly * 3 * y)
    f_hat = g.fft(f)
    dfdx = (2 * np.pi / Lx * 2) * np.cos(2 * np.pi / Lx * 2 * x) * np.cos(2 * np.pi / Ly * 3 * y)
    assert np.allclose(g.ifft(g.ddx(f_hat)), dfdx, atol=1e-12)
