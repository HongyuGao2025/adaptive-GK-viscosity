from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class StageConfig:
    enabled: bool = True


@dataclass(slots=True)
class CombineConfig(StageConfig):
    stress_file_pattern: str = "stress*.dat"
    output_name: str = "combined_stress.dat"


@dataclass(slots=True)
class ViscosityConfig(StageConfig):
    input_name: str = "combined_stress.dat"
    output_name: str = "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
    folder_pattern: str = "2.*_nve"
    temp: float = 373.15
    box_length: float = 35.169
    dt_fs: float = 1.0
    step_interval: int = 100
    max_time_ns: float = 220.0
    n_blocks: int = 10
    points_per_decade: int = 5


@dataclass(slots=True)
class AverageConfig(StageConfig):
    input_name: str = "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
    output_name: str = "avg_visc_5mode_220ns_10blk_loggrid_5point.dat"
    folder_pattern: str = "2.*_nve"


@dataclass(slots=True)
class BootstrapConfig(StageConfig):
    input_name: str = "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
    output_name: str = "bootstrap_fit_5mode_220ns_10blk_fitfrac_loggrid.dat"
    folder_pattern: str = "2.*_nve"
    n_boot: int = 1000
    fit_fractions: list[float] | None = None
    random_seed: int = 12345
    ci_eta_cutoff: float = 0.20
    beta_min: float = 0.05
    beta_max: float = 2.0
    tau_min: float = 1e-5


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path
    combine: CombineConfig
    viscosity: ViscosityConfig
    average: AverageConfig
    bootstrap: BootstrapConfig


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as rf:
        return json.load(rf)


def resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_config(config_path: Path) -> PipelineConfig:
    raw = load_json(config_path)
    base_dir = config_path.parent

    project_root = resolve_path(
        base_dir, raw.get("project_root", "../Proj_EMD_GK/1_301MPa")
    )

    combine_raw = raw.get("combine", {})
    viscosity_raw = raw.get("viscosity", {})
    average_raw = raw.get("average", {})
    bootstrap_raw = raw.get("bootstrap", {})

    fit_fractions = bootstrap_raw.get("fit_fractions", [0.20, 0.30, 0.40, 0.50])

    return PipelineConfig(
        project_root=project_root,
        combine=CombineConfig(
            enabled=bool(combine_raw.get("enabled", True)),
            stress_file_pattern=str(
                combine_raw.get("stress_file_pattern", "stress*.dat")
            ),
            output_name=str(combine_raw.get("output_name", "combined_stress.dat")),
        ),
        viscosity=ViscosityConfig(
            enabled=bool(viscosity_raw.get("enabled", True)),
            input_name=str(viscosity_raw.get("input_name", "combined_stress.dat")),
            output_name=str(
                viscosity_raw.get(
                    "output_name", "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
                )
            ),
            folder_pattern=str(viscosity_raw.get("folder_pattern", "2.*_nve")),
            temp=float(viscosity_raw.get("temp", 373.15)),
            box_length=float(viscosity_raw.get("box_length", 35.169)),
            dt_fs=float(viscosity_raw.get("dt_fs", 1.0)),
            step_interval=int(viscosity_raw.get("step_interval", 100)),
            max_time_ns=float(viscosity_raw.get("max_time_ns", 220.0)),
            n_blocks=int(viscosity_raw.get("n_blocks", 10)),
            points_per_decade=int(viscosity_raw.get("points_per_decade", 5)),
        ),
        average=AverageConfig(
            enabled=bool(average_raw.get("enabled", True)),
            input_name=str(
                average_raw.get(
                    "input_name", "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
                )
            ),
            output_name=str(
                average_raw.get(
                    "output_name", "avg_visc_5mode_220ns_10blk_loggrid_5point.dat"
                )
            ),
            folder_pattern=str(average_raw.get("folder_pattern", "2.*_nve")),
        ),
        bootstrap=BootstrapConfig(
            enabled=bool(bootstrap_raw.get("enabled", True)),
            input_name=str(
                bootstrap_raw.get(
                    "input_name", "out2_visc_5mode_220ns_10blk_loggrid_5point.dat"
                )
            ),
            output_name=str(
                bootstrap_raw.get(
                    "output_name", "bootstrap_fit_5mode_220ns_10blk_fitfrac_loggrid.dat"
                )
            ),
            folder_pattern=str(bootstrap_raw.get("folder_pattern", "2.*_nve")),
            n_boot=int(bootstrap_raw.get("n_boot", 1000)),
            fit_fractions=[float(x) for x in fit_fractions],
            random_seed=int(bootstrap_raw.get("random_seed", 12345)),
            ci_eta_cutoff=float(bootstrap_raw.get("ci_eta_cutoff", 0.20)),
            beta_min=float(bootstrap_raw.get("beta_min", 0.05)),
            beta_max=float(bootstrap_raw.get("beta_max", 2.0)),
            tau_min=float(bootstrap_raw.get("tau_min", 1e-5)),
        ),
    )
