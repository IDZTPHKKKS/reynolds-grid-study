"""Unit tests: 2/3-rule dealiasing."""

import numpy as np

from solver.spectral import SpectralGrid2D


def test_mask_geometry_64():
    g = SpectralGrid2D(64, 64, 2 * np.pi, 2 * np.pi)
    assert g.nx_max == 21 and g.ny_max == 21            # floor(64/3)
    kept = int(g.keep_mask.sum())
    assert kept == (2 * 21 + 1) * (21 + 1)              # (2*ny_max+1) x (nx_max+1)
    # Nyquist mode is excluded
    assert not g.keep_mask[0, -1]
    # modes one beyond the cut are excluded
    assert not g.keep_mask[0, 22]
    assert not g.keep_mask[22, 0] and not g.keep_mask[-22, 0]


def test_dealias_removes_forbidden_modes():
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    f_hat = np.ones((32, 17), dtype=complex)
    f_hat_d = g.dealias(f_hat)
    assert np.all(f_hat_d[~g.keep_mask] == 0.0)
    assert np.all(f_hat_d[g.keep_mask] == 1.0)


def test_truncation_is_idempotent():
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    rng = np.random.default_rng(0)
    f_hat = g.fft(rng.standard_normal((32, 32)))
    once = g.dealias(f_hat)
    twice = g.dealias(once)
    assert np.array_equal(once, twice)


def test_quadratic_product_is_alias_free():
    """The projected discrete product equals the projection of the exact product
    spectrum for truncated inputs (the defining property of the 2/3 rule)."""
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    rng = np.random.default_rng(1)
    a = g.dealias(g.fft(rng.standard_normal((32, 32))))
    b = g.dealias(g.fft(rng.standard_normal((32, 32))))
    discrete = g.dealias(g.fft(g.ifft(a) * g.ifft(b)))

    kept_a, kept_b = {}, {}
    for iy, m in enumerate(g.ny):
        for ix, n in enumerate(g.nx):
            if g.keep_mask[iy, ix]:
                kept_a[(int(m), int(n))] = a[iy, ix]
                kept_b[(int(m), int(n))] = b[iy, ix]
    full = dict(kept_a)
    for (m, n), val in kept_a.items():
        if n > 0:
            full[(-m, -n)] = np.conj(val)
    bfull = dict(kept_b)
    for (m, n), val in kept_b.items():
        if n > 0:
            bfull[(-m, -n)] = np.conj(val)

    exact = np.zeros_like(discrete)
    for iy, my in enumerate(g.ny):
        for ix, mx in enumerate(g.nx):
            if not g.keep_mask[iy, ix]:
                continue
            k = (int(my), int(mx))
            s = 0.0 + 0.0j
            for p, av in full.items():
                q = (k[0] - p[0], k[1] - p[1])
                bv = bfull.get(q)
                if bv is not None:
                    s += av * bv
            exact[iy, ix] = s / (g.Nx * g.Ny)
    assert np.allclose(discrete[g.keep_mask], exact[g.keep_mask], atol=1e-10)


def test_worst_case_no_energy_folding():
    """Two modes whose product wavenumber exceeds Nyquist must NOT fold onto a wrong
    mode after dealiasing: the result is exactly zero there."""
    g = SpectralGrid2D(64, 64, 2 * np.pi, 2 * np.pi)
    x = g.x[None, :]
    a = np.cos(21 * x) * np.ones((g.Ny, 1))     # modes (ny=0, nx=+/-21)
    b = np.cos(21 * x) * np.ones((g.Ny, 1))
    prod_hat = g.fft(a * b)
    prod_hat_d = g.dealias(prod_hat)
    # locate mode nx = 22 (rfft layout, nx >= 0)
    idx22 = int(np.argmin(np.abs(g.nx - 22)))
    assert prod_hat[0, idx22] != 0.0          # raw product DOES alias there
    assert prod_hat_d[0, idx22] == 0.0        # dealiased product does not
    assert np.isclose(prod_hat_d[0, 0], 0.5 * g.Nx * g.Ny, rtol=1e-12)
    assert np.all(np.abs(prod_hat_d[0, 1:]) < 1e-8 * g.Nx * g.Ny)


def test_spectral_truncation_preserves_energy_of_bandlimited_field():
    g = SpectralGrid2D(32, 32, 2 * np.pi, 2 * np.pi)
    x, y = g.x[None, :], g.y[:, None]
    f = np.sin(4 * x + 2 * y)          # single mode inside truncation
    f_hat = g.dealias(g.fft(f))
    assert np.isclose(g.mean_square_hat(f_hat), np.mean(f**2), rtol=1e-12)
