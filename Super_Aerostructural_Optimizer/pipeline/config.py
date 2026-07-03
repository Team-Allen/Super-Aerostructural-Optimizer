"""Configuration models for the real-physics aerostructural pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class WingGeometryConfig:
    """Half-wing geometry used for structural and aerodynamic discretization."""

    half_span_m: float = 5.0
    root_chord_m: float = 1.6
    taper_ratio: float = 0.45
    sweep_deg: float = 18.0
    dihedral_deg: float = 3.0
    twist_root_deg: float = 2.0
    twist_tip_deg: float = -3.0
    n_span: int = 14
    n_chord: int = 5


@dataclass
class FlightConditionConfig:
    """Cruise condition for coupled analysis."""

    mach: float = 0.18
    velocity_ms: float = 60.0
    altitude_m: float = 1500.0
    alpha_deg: float = 2.0
    target_cl: float = 0.45
    trim_alpha_min_deg: float = -3.0
    trim_alpha_max_deg: float = 12.0
    trim_tol: float = 1.0e-3
    trim_max_iter: int = 30


@dataclass
class StructuralConfig:
    """Shell model and material parameters passed to TACS."""

    density_kg_m3: float = 1600.0
    youngs_modulus_pa: float = 52.0e9
    poisson_ratio: float = 0.31
    yield_stress_pa: float = 800.0e6
    thickness_root_m: float = 0.025
    thickness_tip_m: float = 0.008
    thickness_min_m: float = 0.002
    thickness_max_m: float = 0.080
    ks_weight: float = 50.0
    tip_deflection_limit_m: float = 0.8
    tacs_dv_scale: float = 100.0
    tacs_print_timing: bool = False


@dataclass
class CouplingConfig:
    """Load/displacement transfer and fixed-point coupling controls."""

    elastic_scheme: str = "rbf"
    max_iterations: int = 12
    convergence_tol: float = 3.0e-2
    load_relaxation: float = 0.65
    displacement_relaxation: float = 0.55
    meld_npts: int = 30
    meld_beta: float = 0.5
    aero_z_offset_m: float = 0.03


@dataclass
class OptimizerConfig:
    """Design variables and pyOptSparse settings."""

    enabled: bool = True
    fallback_sequence: List[str] = field(
        default_factory=lambda: ["IPOPT", "SLSQP", "PSQP"]
    )
    max_iterations: int = 80
    tolerance: float = 1.0e-5

    # Design variable bounds (deltas/scales around base config values)
    twist_root_delta_min_deg: float = -6.0
    twist_root_delta_max_deg: float = 6.0
    twist_tip_delta_min_deg: float = -8.0
    twist_tip_delta_max_deg: float = 8.0
    thickness_root_scale_min: float = 0.6
    thickness_root_scale_max: float = 1.5
    thickness_tip_scale_min: float = 0.6
    thickness_tip_scale_max: float = 1.6


@dataclass
class OutputConfig:
    """Artifact output controls."""

    results_dir: str = "results/real_physics"
    case_name: str = "real_physics_mdo"
    write_tacs_solution: bool = True
    save_history: bool = True
    save_optimizer_db: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    wing: WingGeometryConfig = field(default_factory=WingGeometryConfig)
    flight: FlightConditionConfig = field(default_factory=FlightConditionConfig)
    structure: StructuralConfig = field(default_factory=StructuralConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        """Raise ValueError if configuration is physically or numerically invalid."""
        if self.wing.half_span_m <= 0.0:
            raise ValueError("wing.half_span_m must be positive")
        if self.wing.root_chord_m <= 0.0:
            raise ValueError("wing.root_chord_m must be positive")
        if not (0.05 <= self.wing.taper_ratio <= 1.0):
            raise ValueError("wing.taper_ratio must be in [0.05, 1.0]")
        if self.wing.n_span < 4:
            raise ValueError("wing.n_span must be >= 4")
        if self.wing.n_chord < 2:
            raise ValueError("wing.n_chord must be >= 2")

        if self.flight.velocity_ms <= 1.0:
            raise ValueError("flight.velocity_ms must be > 1 m/s")
        if self.flight.trim_alpha_min_deg >= self.flight.trim_alpha_max_deg:
            raise ValueError("trim alpha bounds are invalid")
        if self.flight.target_cl <= 0.0:
            raise ValueError("flight.target_cl must be positive")

        s = self.structure
        if s.thickness_min_m <= 0.0:
            raise ValueError("structure.thickness_min_m must be positive")
        if s.thickness_max_m <= s.thickness_min_m:
            raise ValueError("structure.thickness_max_m must exceed thickness_min_m")
        if s.tip_deflection_limit_m <= 0.0:
            raise ValueError("structure.tip_deflection_limit_m must be positive")

        c = self.coupling
        valid_schemes = {"rbf", "meld", "linearized meld", "beam", "hermes"}
        if c.elastic_scheme not in valid_schemes:
            raise ValueError(f"coupling.elastic_scheme must be one of {sorted(valid_schemes)}")
        if c.max_iterations < 1:
            raise ValueError("coupling.max_iterations must be >= 1")
        if not (0.0 < c.load_relaxation <= 1.0):
            raise ValueError("coupling.load_relaxation must be in (0, 1]")
        if not (0.0 < c.displacement_relaxation <= 1.0):
            raise ValueError("coupling.displacement_relaxation must be in (0, 1]")
        if c.meld_npts < 1:
            raise ValueError("coupling.meld_npts must be >= 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Construct config from a dictionary with optional missing sections."""
        return cls(
            wing=WingGeometryConfig(**data.get("wing", {})),
            flight=FlightConditionConfig(**data.get("flight", {})),
            structure=StructuralConfig(**data.get("structure", {})),
            coupling=CouplingConfig(**data.get("coupling", {})),
            optimizer=OptimizerConfig(**data.get("optimizer", {})),
            output=OutputConfig(**data.get("output", {})),
        )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration JSON file."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cfg = PipelineConfig.from_dict(payload)
    cfg.validate()
    return cfg


def save_pipeline_config(config: PipelineConfig, path: str | Path) -> None:
    """Save a validated configuration JSON file."""
    config.validate()
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)
