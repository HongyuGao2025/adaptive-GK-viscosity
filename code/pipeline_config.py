"""Configuration loading and validation for adaptive Green-Kubo analysis.
Simulation parameters and convergence settings are read from a JSON file.
All physical quantities use the units specified in the configuration
classes below.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StageConfig:
    """Base configuration for enabling or disabling a pipeline stage."""
    enabled: bool = True


@dataclass(slots=True)
class CombineConfig(StageConfig):
    """Settings for combining stress tensor files within each replica."""
    stress_file_pattern: str = "stress*.dat"
    output_name: str = "combined_stress.dat"


@dataclass(slots=True)
class ViscosityConfig(StageConfig):
    """Settings for the Green-Kubo viscosity calculation.
    temp: Temperature in K.
    box_length: Side length of a cubic simulation box in angstroms.
    dt_fs: MD integration timestep in fs.
    step_interval: Number of MD steps between stored stress samples.
    points_per_decade: Density of the logarithmic correlation-time grid.
    Stress tensor components are expected in atm (LAMMPS real units).
    """
    input_name: str = "combined_stress.dat"
    folder_pattern: str = "*_nve"
    temp: float = 373.15
    box_length: float = 0.0
    dt_fs: float = 1.0
    step_interval: int = 100
    points_per_decade: int = 5


@dataclass(slots=True)
class BootstrapConfig(StageConfig):
    """Settings for bootstrap fitting and convergence criteria.
    n_boot: Number of bootstrap resamples.
    fit_fractions: Fractions of the correlation-time window used for fitting.
    ci_eta_cutoff: Maximum relative width of the 95% viscosity CI.
    early_t95_fraction: Maximum median t95 relative to block length.
    beta_min, beta_max: Bounds for the stretched-exponential exponent.
    tau_min: Lower bound on relaxation time, in ns.
    """
    n_boot: int = 1000
    fit_fractions: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50)
    random_seed: int = 12345
    ci_eta_cutoff: float = 0.20
    early_t95_fraction: float = 0.20
    beta_min: float = 0.05
    beta_max: float = 2.0
    tau_min: float = 1e-5


@dataclass(slots=True)
class AdaptiveConfig(StageConfig):
    """Settings for adaptive block-length selection.
    All block lengths and search resolutions are specified in ns.
    """
    initial_t_blk_ns: float = 1.0
    refinement_resolution_ns: float = 1.0
    max_growth_factor: float = 2.0
    confirm_next: bool = True
    max_t_blk_ns: float | None = None
    n_blocks: int = 10
    output_dir: str = "adaptive_gk_results"
    report_name: str = "convergence_report.txt"


@dataclass(slots=True)
class PipelineConfig:
    """Complete configuration for the analysis pipeline."""
    project_root: Path
    combine: CombineConfig
    viscosity: ViscosityConfig
    bootstrap: BootstrapConfig
    adaptive: AdaptiveConfig


def load_json(path: Path) -> dict[str, Any]:
    """Load configuration settings from a JSON file."""
    with path.open("r", encoding="utf-8") as rf:
        return json.load(rf)


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Resolve relative paths against the configuration directory."""
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _fit_fractions(value: Any) -> tuple[float, ...]:
    """Convert fit fractions to a sorted tuple and validate their range."""
    if value is None:
        vals = (0.20, 0.30, 0.40, 0.50)
    elif isinstance(value, (int, float)):
        vals = (float(value),)
    else:
        vals = tuple(float(x) for x in value)
        
    if not vals or any(not math.isfinite(x) or x <= 0.0 or x > 1.0 for x in vals):
        raise ValueError("fit_fractions must contain finite values in (0, 1].")

    return tuple(sorted(vals))


def build_config(config_path: Path) -> PipelineConfig:
    """Build and validate a pipeline configuration from a JSON file.
    Relative paths are interpreted with respect to the JSON file's
    directory, rather than the current working directory.
    """
    raw = load_json(config_path)
    base_dir = config_path.parent
    project_root = resolve_path(base_dir, raw.get("project_root", "."))

    combine_raw = raw.get("combine", {})
    viscosity_raw = raw.get("viscosity", {})
    bootstrap_raw = raw.get("bootstrap", {})
    adaptive_raw = raw.get("adaptive", {})

    cfg = PipelineConfig(
        project_root=project_root,
        combine=CombineConfig(
            enabled=bool(combine_raw.get("enabled", True)),
            stress_file_pattern=str(combine_raw.get("stress_file_pattern", "stress*.dat")),
            output_name=str(combine_raw.get("output_name", "combined_stress.dat")),
        ),
        viscosity=ViscosityConfig(
            enabled=bool(viscosity_raw.get("enabled", True)),
            input_name=str(viscosity_raw.get("input_name", "combined_stress.dat")),
            folder_pattern=str(viscosity_raw.get("folder_pattern", "*_nve")),
            temp=float(viscosity_raw.get("temp", 373.15)),
            box_length=float(viscosity_raw.get("box_length", 0.0)),
            dt_fs=float(viscosity_raw.get("dt_fs", 1.0)),
            step_interval=int(viscosity_raw.get("step_interval", 100)),
            points_per_decade=int(viscosity_raw.get("points_per_decade", 5)),
        ),
        bootstrap=BootstrapConfig(
            enabled=bool(bootstrap_raw.get("enabled", True)),
            n_boot=int(bootstrap_raw.get("n_boot", 1000)),
            fit_fractions=_fit_fractions(bootstrap_raw.get("fit_fractions", [0.20, 0.30, 0.40, 0.50])),
            random_seed=int(bootstrap_raw.get("random_seed", 12345)),
            ci_eta_cutoff=float(bootstrap_raw.get("ci_eta_cutoff", 0.20)),
            early_t95_fraction=float(bootstrap_raw.get("early_t95_fraction", 0.20)),
            beta_min=float(bootstrap_raw.get("beta_min", 0.05)),
            beta_max=float(bootstrap_raw.get("beta_max", 2.0)),
            tau_min=float(bootstrap_raw.get("tau_min", 1e-5)),
        ),
        adaptive=AdaptiveConfig(
            enabled=bool(adaptive_raw.get("enabled", True)),
            initial_t_blk_ns=float(adaptive_raw.get("initial_t_blk_ns", 1.0)),
            refinement_resolution_ns=float(adaptive_raw.get("refinement_resolution_ns", 1.0)),
            max_growth_factor=float(adaptive_raw.get("max_growth_factor", 2.0)),
            confirm_next=bool(adaptive_raw.get("confirm_next", True)),
            max_t_blk_ns=(float(adaptive_raw["max_t_blk_ns"]) if adaptive_raw.get("max_t_blk_ns") is not None else None),
            n_blocks=int(adaptive_raw.get("n_blocks", 10)),
            output_dir=str(adaptive_raw.get("output_dir", "adaptive_gk_results")),
            report_name=str(adaptive_raw.get("report_name", "convergence_report.txt")),
        ),
    )

    # Validate physical parameters and numerical settings.
    if not math.isfinite(cfg.viscosity.temp) or cfg.viscosity.temp <= 0:
        raise ValueError("temp must be a positive temperature in K.")
    if not math.isfinite(cfg.viscosity.box_length) or cfg.viscosity.box_length <= 0:
        raise ValueError("box_length must be a positive cubic-box length in angstroms.")
    if not math.isfinite(cfg.viscosity.dt_fs) or cfg.viscosity.dt_fs <= 0:
        raise ValueError("dt_fs must be positive.")
    if cfg.viscosity.step_interval < 1 or cfg.viscosity.points_per_decade < 1:
        raise ValueError("step_interval and points_per_decade must be >= 1.")

    if cfg.bootstrap.n_boot < 1:
        raise ValueError("n_boot must be >= 1.")
    if not math.isfinite(cfg.bootstrap.ci_eta_cutoff) or cfg.bootstrap.ci_eta_cutoff <= 0:
        raise ValueError("ci_eta_cutoff must be positive.")
    if not (0.0 < cfg.bootstrap.early_t95_fraction < 1.0):
        raise ValueError("early_t95_fraction must be in (0, 1).")
    if not (0.0 < cfg.bootstrap.beta_min < cfg.bootstrap.beta_max):
        raise ValueError("Require 0 < beta_min < beta_max.")
    if not math.isfinite(cfg.bootstrap.beta_max):
        raise ValueError("beta_max must be finite.")
    if not math.isfinite(cfg.bootstrap.tau_min) or cfg.bootstrap.tau_min <= 0:
        raise ValueError("tau_min must be positive.")

    if not math.isfinite(cfg.adaptive.initial_t_blk_ns) or cfg.adaptive.initial_t_blk_ns <= 0:
        raise ValueError("initial_t_blk_ns must be positive.")
    if not math.isfinite(cfg.adaptive.refinement_resolution_ns) or cfg.adaptive.refinement_resolution_ns <= 0:
        raise ValueError("refinement_resolution_ns must be positive.")
    if not math.isfinite(cfg.adaptive.max_growth_factor) or cfg.adaptive.max_growth_factor <= 1:
        raise ValueError("max_growth_factor must be > 1.")
    if cfg.adaptive.n_blocks < 1:
        raise ValueError("n_blocks must be >= 1.")
    if cfg.adaptive.max_t_blk_ns is not None:
        if not math.isfinite(cfg.adaptive.max_t_blk_ns) or cfg.adaptive.max_t_blk_ns <= 0:
            raise ValueError("max_t_blk_ns must be positive when specified.")
            
    return cfg
