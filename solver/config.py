"""Configuration loading/validation for the 2D NS benchmark."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml

__all__ = ["SimulationConfig", "load_config", "CONVENTIONS_VERSION"]

CONVENTIONS_VERSION = (
    "ns2d-v1: vorticity formulation, omega = -lap(psi), u = d(psi)/dy, v = -d(psi)/dx; "
    "numpy rfft2 (backward normalization), fields shaped (Ny, Nx); "
    "2/3-rule Galerkin truncation (|nx|<=(Nx-1)//3, |ny|<=(Ny-1)//3); "
    "Kolmogorov forcing f = (A sin(k_f y), 0), k_f = 2 pi n_f / Ly; "
    "conservative nonlinear term -div(u omega); ETDRK4/IFRK4/RK4; "
    "Re_f = u_rms/(nu k_f) with u_rms calibrated to target_u_rms."
)


@dataclass
class DomainConfig:
    Lx: float = 2.0 * 3.141592653589793
    Ly: float = 2.0 * 3.141592653589793


@dataclass
class GridConfig:
    Nx: int = 64
    Ny: int = 64


@dataclass
class ForcingConfig:
    type: str = "kolmogorov"
    amplitude: float = 1.0        # A; may be recalibrated to hit target u_rms
    mode: int = 4


@dataclass
class ViscosityConfig:
    """Dissipation parameters."""
    mode: str = "from_target_reynolds"   # or "explicit"
    nu: Optional[float] = None
    target_re_f: Optional[float] = None
    target_u_rms: Optional[float] = None
    drag: float = 0.0

    def resolve(self, k_f: float) -> float:
        if self.mode == "explicit":
            if self.nu is None or self.nu < 0:
                raise ValueError("viscosity.mode='explicit' requires nu >= 0.")
            return float(self.nu)
        if self.mode == "from_target_reynolds":
            if self.target_re_f is None or self.target_u_rms is None:
                raise ValueError(
                    "viscosity.mode='from_target_reynolds' requires target_re_f "
                    "and target_u_rms."
                )
            # Re_f := u_rms/(nu k_f)  =>  nu = u_rms/(Re_f k_f)
            return float(self.target_u_rms / (self.target_re_f * k_f))
        raise ValueError(f"Unknown viscosity mode '{self.mode}'.")


@dataclass
class TimeConfig:
    dt: float = 0.005
    t_total: float = 200.0
    dt_sample: float = 0.25
    max_courant: float = 1.0     # warn above this Courant number
    fail_courant: float = 2.0    # hard abort above this (RK4 stability ~2.8)


@dataclass
class NumericsConfig:
    dealias: str = "2/3"
    integrator: str = "etdrk4"   # etdrk4 | ifrk4 | rk4


@dataclass
class ICConfig:
    type: str = "band_limited_random"
    u_rms: float = 1.0
    k_cut: int = 8               # integer mode-number cutoff of the IC spectrum
    seed: int = 1000


@dataclass
class CalibrationConfig:
    enabled: bool = True
    target_u_rms: float = 1.0
    tolerance: float = 0.03
    max_iterations: int = 4
    pilot_measure: float = 20.0      # measurement window
    pilot_reequilibrate: float = 60.0  # re-equilibration after amplitude change
                                       # (several drag timescales 1/sigma)


@dataclass
class StationarityConfig:
    min_burn: float = 30.0       # minimum burn-in (time units)
    window: float = 40.0         # trailing window used for the stationarity test
    chunk: float = 5.0           # burn-in proceeds in chunks of this length
    max_burn: float = 150.0      # give up (with error) after this much burn-in
    drift_e: float = 0.02
    drift_z: float = 0.05
    balance: float = 0.03


@dataclass
class OutputConfig:
    n_traj: int = 2
    seed_offset: int = 0
    dtype: str = "float32"
    directory: str = "data"


@dataclass
class SimulationConfig:
    name: str = "unnamed"
    description: str = ""
    domain: DomainConfig = field(default_factory=DomainConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    forcing: ForcingConfig = field(default_factory=ForcingConfig)
    viscosity: ViscosityConfig = field(default_factory=ViscosityConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    initial_condition: ICConfig = field(default_factory=ICConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    stationarity: StationarityConfig = field(default_factory=StationarityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ----------------------------------------------------------------- derived
    @property
    def k_f(self) -> float:
        return (2.0 * 3.141592653589793 / self.domain.Ly) * self.forcing.mode

    @property
    def nu(self) -> float:
        return self.viscosity.resolve(self.k_f)

    def to_dict(self) -> dict:
        d = {k: dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v
             for k, v in self.__dict__.items()}
        d["nu_resolved"] = self.nu
        d["k_f"] = self.k_f
        d["conventions_version"] = CONVENTIONS_VERSION
        return d


_SECTION_TYPES = {
    "domain": DomainConfig,
    "grid": GridConfig,
    "forcing": ForcingConfig,
    "viscosity": ViscosityConfig,
    "time": TimeConfig,
    "numerics": NumericsConfig,
    "initial_condition": ICConfig,
    "calibration": CalibrationConfig,
    "stationarity": StationarityConfig,
    "output": OutputConfig,
}


def load_config(path: str, overrides: Optional[dict] = None) -> SimulationConfig:
    """Load a YAML config; ``overrides`` is a mapping applied after loading (used by
    generate_dataset.py CLI, e.g. {'time': {'dt': 0.002}})."""
    import copy

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    raw = copy.deepcopy(raw)
    if overrides:
        for key, sub in overrides.items():
            if isinstance(sub, dict) and isinstance(raw.get(key), dict):
                raw[key].update(sub)
            else:
                raw[key] = sub

    cfg = SimulationConfig(name=raw.get("name", "unnamed"),
                           description=raw.get("description", ""))
    for section, cls in _SECTION_TYPES.items():
        if section in raw and raw[section] is not None:
            fields = {f.name for f in dataclasses.fields(cls)}
            unknown = set(raw[section]) - fields
            if unknown:
                raise ValueError(f"[{path}] unknown keys in '{section}': {sorted(unknown)}")
            setattr(cfg, section, cls(**raw[section]))

    # Cross-field validation.
    if cfg.forcing.type != "kolmogorov":
        raise ValueError("Only the 'kolmogorov' forcing is implemented.")
    if cfg.numerics.dealias != "2/3":
        raise ValueError("Only the 2/3 rule is implemented.")
    if cfg.numerics.integrator not in ("etdrk4", "ifrk4", "rk4"):
        raise ValueError(f"Unknown integrator {cfg.numerics.integrator!r}.")
    if cfg.forcing.mode > cfg.grid.Ny // 3:
        raise ValueError(
            f"n_f={cfg.forcing.mode} outside 2/3 truncation for Ny={cfg.grid.Ny}."
        )
    if cfg.viscosity.mode == "from_target_reynolds":
        cfg.viscosity.target_u_rms = (
            cfg.viscosity.target_u_rms or cfg.calibration.target_u_rms
        )
        if cfg.viscosity.target_u_rms is None or cfg.viscosity.target_re_f is None:
            raise ValueError("target_re_f and target_u_rms required.")
    _ = cfg.nu  # validate resolvability early
    return cfg
