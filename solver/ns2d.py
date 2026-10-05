"""Pseudo-spectral 2D incompressible Navier-Stokes solver (Kolmogorov flow)."""

from __future__ import annotations

import os
import time as _walltime
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import SimulationConfig
from .diagnostics import courant_number, state_diagnostics, stationarity_report
from .forcing import KolmogorovForcing
from .integrator import ETDRK4, make_integrator
from .spectral import SpectralGrid2D

__all__ = ["NS2D", "RunResult"]


class _TorchETDRK4:
    """ETDRK4 step of NS2D in torch (complex128), for SOLVER_DEVICE=cuda or torch:<device>."""

    def __init__(self, sim, device):
        import torch
        self.torch = torch
        self.device = torch.device(device)
        g = sim.grid
        c = sim.stepper.coeffs
        ones = np.ones(g.k2.shape)
        self.E, self.E2, self.Q, self.f1, self.f2, self.f3 = (self.tensor(x) for x in
                                                              (c.E, c.E2, c.Q, c.f1, c.f2, c.f3))
        self.ikx = self.tensor(1j * g.KX * ones)
        self.iky = self.tensor(1j * g.KY * ones)
        inv_k2 = 1.0 / g.k2_safe
        inv_k2[0, 0] = 0.0
        self.inv_k2 = self.tensor(inv_k2)
        self.mask = self.tensor(g.keep_mask.astype(float))
        self.shape = (g.Ny, g.Nx)
        self.f = None

    def tensor(self, a):
        return self.torch.as_tensor(np.ascontiguousarray(a), dtype=self.torch.complex128, device=self.device)

    def rhs(self, w):
        fft, ifft = self.torch.fft.rfft2, self.torch.fft.irfft2
        psi = w * self.inv_k2
        u = ifft(self.iky * psi, s=self.shape)
        v = ifft(-self.ikx * psi, s=self.shape)
        om = ifft(w, s=self.shape)
        nl = -(self.ikx * fft(u * om) + self.iky * fft(v * om)) * self.mask + self.f
        nl[0, 0] = 0.0
        return nl

    def step(self, w):
        nw = self.rhs(w)
        a = self.E2 * w + self.Q * nw
        na = self.rhs(a)
        b = self.E2 * w + self.Q * na
        nb = self.rhs(b)
        cc = self.E2 * a + self.Q * (2 * nb - nw)
        nc = self.rhs(cc)
        w = (self.E * w + self.f1 * nw + self.f2 * (na + nb) + self.f3 * nc) * self.mask
        w[0, 0] = 0.0
        return w

    def load(self, omega_hat, forcing_hat):
        self.f = self.tensor(forcing_hat)
        return self.tensor(omega_hat)

    def save(self, w):
        return w.cpu().numpy().astype(np.complex128)


@dataclass
class RunResult:
    times: np.ndarray                      # diagnostic times
    diagnostics: dict                      # lists of scalars, keys as state_diagnostics
    sample_times: np.ndarray
    samples: list                          # physical omega snapshots
    max_courant: float
    wall_time: float
    n_steps: int


class NS2D:
    """Forced 2D Navier-Stokes on a doubly-periodic domain."""

    def __init__(self, config: SimulationConfig, forcing_amplitude: Optional[float] = None):
        self.config = config
        self.grid = SpectralGrid2D(config.grid.Nx, config.grid.Ny,
                                   config.domain.Lx, config.domain.Ly)
        amp = forcing_amplitude if forcing_amplitude is not None else config.forcing.amplitude
        self.forcing = KolmogorovForcing(amp, config.forcing.mode, self.grid)
        self.nu = config.nu
        self.drag = float(config.viscosity.drag)
        if self.drag < 0:
            raise ValueError("drag must be >= 0.")
        self.dt = config.time.dt
        self.L_hat = -(self.nu * self.grid.k2 + self.drag)
        self.stepper = make_integrator(
            config.numerics.integrator, self.dt, self.L_hat,
            rhs_N=self.rhs_nonlinear, rhs_total=self.rhs_total,
        )
        self.omega_hat: Optional[np.ndarray] = None
        self.time = 0.0
        self._accel = None

    def _accelerator(self):
        dev = os.environ.get("SOLVER_DEVICE", "cpu")
        if dev == "cpu" or not isinstance(self.stepper, ETDRK4):
            return None
        if self._accel is None:
            self._accel = _TorchETDRK4(self, dev[len("torch:"):] if dev.startswith("torch:") else dev)
        return self._accel

    # ------------------------------------------------------------------ RHS --
    def rhs_nonlinear(self, omega_hat: np.ndarray) -> np.ndarray:
        """N(omega_hat) = -(div(u omega))_hat (dealiased) + F_omega_hat."""
        g = self.grid
        psi_hat = g.streamfunction_hat(omega_hat)
        u_hat = 1j * g.KY * psi_hat
        v_hat = -1j * g.KX * psi_hat
        u_hat[0, 0] = 0.0
        v_hat[0, 0] = 0.0
        u = g.ifft(u_hat)
        v = g.ifft(v_hat)
        omega = g.ifft(omega_hat)

        flux_x_hat = g.fft(u * omega)
        flux_y_hat = g.fft(v * omega)
        nl = -(g.ddx(flux_x_hat) + g.ddy(flux_y_hat))
        nl = g.dealias(nl)
        nl = nl + self.forcing.f_omega_hat
        nl[0, 0] = 0.0
        return nl

    def rhs_total(self, omega_hat: np.ndarray) -> np.ndarray:
        """Full RHS (for explicit RK4): N + L omega."""
        return self.rhs_nonlinear(omega_hat) + self.L_hat * omega_hat

    def apply_galerkin(self, omega_hat: np.ndarray) -> np.ndarray:
        """Enforce the Galerkin truncation and zero mean after each step."""
        omega_hat *= self.grid.keep_mask
        omega_hat[0, 0] = 0.0
        return omega_hat

    # ------------------------------------------------------------------- IC --
    def set_initial_state(self, seed: int, u_rms: Optional[float] = None,
                          k_cut: Optional[int] = None) -> None:
        """Band-limited random vorticity IC."""
        g = self.grid
        u_rms = self.config.initial_condition.u_rms if u_rms is None else u_rms
        k_cut = self.config.initial_condition.k_cut if k_cut is None else k_cut
        rng = np.random.default_rng(seed)

        radius = np.sqrt(g.ny[:, None] ** 2 + g.nx[None, :] ** 2)
        amp = (radius**1.5) * np.exp(-radius / max(k_cut, 1))   # |omega_hat| ~ sqrt(E_ic)*k
        amp = np.where(radius < 1.0, 0.0, amp)                  # remove k = 0
        noise_hat = g.fft(rng.standard_normal((g.Ny, g.Nx)))    # valid real-field spectrum
        omega_hat = amp * noise_hat
        omega_hat = g.dealias(omega_hat)
        omega_hat[0, 0] = 0.0

        u_hat, v_hat = g.velocity_hat(omega_hat)
        E0 = 0.5 * np.sum(g.rfft_w * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2)) / (g.Nx * g.Ny) ** 2
        if E0 <= 0:
            raise ValueError("IC has zero energy; increase k_cut.")
        omega_hat *= np.sqrt((0.5 * u_rms**2) / E0)

        self.omega_hat = omega_hat.astype(np.complex128)
        self.time = 0.0

    def set_state_from_omega(self, omega: np.ndarray) -> None:
        """Load a physical-space vorticity field (e.g. a dataset snapshot)."""
        self.omega_hat = self.grid.dealias(self.grid.fft(omega)).astype(np.complex128)
        self.omega_hat[0, 0] = 0.0
        self.time = 0.0

    # ---------------------------------------------------------------- state --
    def state(self) -> dict:
        return state_diagnostics(self.grid, self.forcing, self.nu, self.omega_hat,
                                 drag=self.drag)

    def fields(self) -> tuple:
        omega = self.grid.ifft(self.omega_hat)
        u, v, psi = self.grid.velocity(self.omega_hat)
        return omega, u, v, psi

    # ----------------------------------------------------------------- step --
    def step(self) -> None:
        self.omega_hat = self.stepper.step(self.omega_hat)
        self.apply_galerkin(self.omega_hat)
        self.time += self.dt

    # ------------------------------------------------------------------ run --
    def run(self, t_end: float, sample_interval: Optional[float] = None,
            sample_fn: Optional[Callable[[], np.ndarray]] = None,
            diagnostics_interval: Optional[float] = None,
            on_diagnostic: Optional[Callable[[float, dict], None]] = None,
            progress: bool = False) -> RunResult:
        """Integrate from the current state until ``self.time + t_end``."""
        if self.omega_hat is None:
            raise RuntimeError("Initial state not set.")
        cfg = self.config
        n_steps = int(round(t_end / self.dt))
        sample_every = max(1, int(round((sample_interval or t_end) / self.dt)))
        diag_every = max(1, int(round((diagnostics_interval or (sample_interval or t_end))
                                      / self.dt)))

        times, diags = [], []
        sample_times, samples = [], []
        max_courant = 0.0
        t0_wall = _walltime.time()
        acc = self._accelerator()
        w = acc.load(self.omega_hat, self.forcing.f_omega_hat) if acc is not None else None

        for step in range(1, n_steps + 1):
            if acc is None:
                self.step()
            else:
                w = acc.step(w)
                self.time += self.dt
                if step % diag_every == 0 or step % sample_every == 0 or step == n_steps:
                    self.omega_hat = acc.save(w)

            if step % diag_every == 0 or step == n_steps:
                d = self.state()
                omega, u, v, _ = self.fields()
                d["courant"] = courant_number(u, v, self.grid.kmax_x, self.grid.kmax_y, self.dt)
                d["time"] = self.time
                max_courant = max(max_courant, d["courant"])
                times.append(self.time)
                diags.append(d)
                if on_diagnostic is not None:
                    on_diagnostic(self.time, d)
                if not all(np.isfinite(d[k]) for k in ("E", "enstrophy", "courant")):
                    raise RuntimeError(
                        f"Non-finite diagnostics at t={self.time:.4f} "
                        f"(E={d['E']!r}, enstrophy={d['enstrophy']!r}, "
                        f"courant={d['courant']!r}); solver has diverged. "
                        f"Halve time.dt or check forcing/viscosity parameters."
                    )
                if d["courant"] > cfg.time.fail_courant:
                    raise RuntimeError(
                        f"Courant number {d['courant']:.2f} exceeded hard limit "
                        f"{cfg.time.fail_courant}; halve time.dt."
                    )

            if step % sample_every == 0 or step == n_steps:
                sample_times.append(self.time)
                samples.append(sample_fn() if sample_fn is not None else self.grid.ifft(self.omega_hat).copy())

            if progress and step % max(1, n_steps // 20) == 0:
                print(f"  [{cfg.name}] t={self.time:8.2f}  E={diags[-1]['E']:.4f}  "
                      f"CFL={diags[-1]['courant']:.2f}", flush=True)

        diagnostics = {k: [d[k] for d in diags] for k in diags[0]} if diags else {}
        return RunResult(
            times=np.asarray(times), diagnostics=diagnostics,
            sample_times=np.asarray(sample_times), samples=samples,
            max_courant=max_courant, wall_time=_walltime.time() - t0_wall, n_steps=n_steps,
        )

    # ----------------------------------------------------------- burn-in/run --
    def run_until_stationary(self, verbose: bool = False) -> tuple[float, dict]:
        """Evolve in chunks until the documented stationarity criteria hold."""
        sc = self.config.stationarity
        burn = 0.0
        history = {k: [] for k in ("time", "E", "enstrophy", "P_in", "eps")}
        report: dict = {}
        diag_dt = min(sc.chunk / 10.0, self.dt * 100.0)
        while True:
            target = sc.min_burn if burn == 0 else burn + sc.chunk
            result = self.run(target - burn, diagnostics_interval=diag_dt)
            for key in history:
                history[key].extend(result.diagnostics[key])
            burn = target

            t = np.asarray(history["time"])
            w = t >= (burn - sc.window)
            if burn >= sc.min_burn and np.count_nonzero(w) >= 10:
                report = stationarity_report(
                    t[w],
                    np.asarray(history["E"])[w],
                    np.asarray(history["enstrophy"])[w],
                    np.asarray(history["P_in"])[w],
                    np.asarray(history["eps"])[w],
                    sc.drift_e, sc.drift_z, sc.balance,
                )
                if verbose:
                    print(f"  [burn] t={burn:6.1f} driftE={report['drift_E']:.4f} "
                          f"driftZ={report['drift_enstrophy']:.4f} "
                          f"balance={report['production_dissipation_mismatch']:.4f} "
                          f"(window {int(w.sum())} samples)", flush=True)
                if report.get("stationary"):
                    return burn, report
            if burn >= sc.max_burn:
                msg = (f"Stationarity not reached within {sc.max_burn} time units; "
                       f"last report: {report}")
                raise RuntimeError(msg)
