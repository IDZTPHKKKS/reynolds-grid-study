"""Forced 2D Navier-Stokes (Kolmogorov flow) spectral solver package."""

from .config import SimulationConfig, load_config, CONVENTIONS_VERSION
from .spectral import SpectralGrid2D
from .forcing import KolmogorovForcing
from .integrator import RK4, IFRK4, ETDRK4, make_integrator
from .ns2d import NS2D, RunResult
from . import diagnostics

__all__ = [
    "SimulationConfig", "load_config", "CONVENTIONS_VERSION",
    "SpectralGrid2D", "KolmogorovForcing",
    "RK4", "IFRK4", "ETDRK4", "make_integrator",
    "NS2D", "RunResult", "diagnostics",
]
