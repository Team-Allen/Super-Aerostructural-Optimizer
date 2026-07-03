"""End-to-end coupled aerostructural analysis workflow."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .aero import LiftingLineAeroSolver
from .config import PipelineConfig
from .geometry import (
    aero_nodes_from_structural_mesh,
    build_half_wing_mesh,
    build_spanwise_aero_grid,
    compute_aeroelastic_row_twist_delta_deg,
    distribute_panel_forces_to_nodal_loads,
    spanwise_thickness_distribution,
    validate_tacs_node_order,
    write_structural_bdf,
)
from .structure import StructuralResult, TacsShellModel
from .transfer import FuntofemMeldTransfer


@dataclass
class DesignVariables:
    """Primary design variables for the pipeline."""

    twist_root_delta_deg: float = 0.0
    twist_tip_delta_deg: float = 0.0
    thickness_root_scale: float = 1.0
    thickness_tip_scale: float = 1.0

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.twist_root_delta_deg,
                self.twist_tip_delta_deg,
                self.thickness_root_scale,
                self.thickness_tip_scale,
            ],
            dtype=float,
        )

    @classmethod
    def from_array(cls, values: np.ndarray) -> "DesignVariables":
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != 4:
            raise ValueError("Design vector must contain 4 values")
        return cls(
            twist_root_delta_deg=float(arr[0]),
            twist_tip_delta_deg=float(arr[1]),
            thickness_root_scale=float(arr[2]),
            thickness_tip_scale=float(arr[3]),
        )


@dataclass
class CouplingIterationRecord:
    """Per-iteration convergence and physics snapshot."""

    iteration: int
    alpha_deg: float
    cl: float
    cd: float
    lift_n: float
    drag_n: float
    mass_kg: float
    ks_failure: float
    tip_deflection_m: float
    aero_load_norm: float
    struct_load_norm: float
    struct_disp_norm: float
    aero_disp_norm: float
    load_rel_change: float
    disp_rel_change: float
    handoff_finite: bool
    trim_converged: bool


class AerostructuralMDOPipeline:
    """Coupled aero-structural workflow built from FUNtoFEM + TACS + pyOptSparse."""

    def __init__(
        self,
        config: PipelineConfig,
        event_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> None:
        self.config = config
        self.config.validate()
        self.event_callback = event_callback

        self._emit_event(
            {
                "event": "pipeline_init_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

        self.results_dir = Path(self.config.output.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.mesh = build_half_wing_mesh(self.config.wing)
        self._emit_event(
            {
                "event": "stage_geometry_built",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "n_nodes": int(self.mesh.num_nodes),
                "n_span": int(self.mesh.n_span),
                "n_chord": int(self.mesh.n_chord),
                "half_span_m": float(self.mesh.half_span_m),
                "area_half_m2": float(self.mesh.area_half_m2),
            }
        )
        self.bdf_path = write_structural_bdf(
            self.mesh, self.results_dir / "generated" / "wing_half_struct.bdf"
        )
        self._emit_event(
            {
                "event": "stage_bdf_written",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "bdf_path": str(self.bdf_path),
            }
        )

        self.structure = TacsShellModel(self.bdf_path, self.config.structure)
        nodes_match, node_err = validate_tacs_node_order(self.structure.struct_nodes, self.mesh.nodes)
        if not nodes_match:
            raise RuntimeError(
                f"TACS node order mismatch with generated BDF mesh (max abs error={node_err:.3e})."
            )
        self._emit_event(
            {
                "event": "stage_tacs_initialized",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "num_nodes": int(self.structure.num_nodes),
                "num_design_vars": int(self.structure.num_design_vars),
            }
        )

        self.aero_nodes = aero_nodes_from_structural_mesh(
            self.mesh, self.config.coupling.aero_z_offset_m
        )
        self.transfer = FuntofemMeldTransfer(
            struct_nodes_xyz=self.structure.struct_nodes,
            aero_nodes_xyz=self.aero_nodes,
            elastic_scheme=self.config.coupling.elastic_scheme,
            npts=self.config.coupling.meld_npts,
            beta=self.config.coupling.meld_beta,
        )
        self._emit_event(
            {
                "event": "stage_transfer_initialized",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "scheme": self.config.coupling.elastic_scheme,
                "npts": int(self.config.coupling.meld_npts),
            }
        )

        self.aero_solver = LiftingLineAeroSolver(self.config.flight)
        self._emit_event(
            {
                "event": "stage_aero_initialized",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.eval_count = 0
        self.iteration_callback: Optional[Callable[[Dict[str, object]], None]] = None
        self._emit_event(
            {
                "event": "pipeline_init_complete",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _emit_event(self, payload: Dict[str, object]) -> None:
        cb = self.event_callback
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            pass

    def default_design(self) -> DesignVariables:
        return DesignVariables()

    def _bounded_design(self, design: DesignVariables) -> DesignVariables:
        opt = self.config.optimizer
        return DesignVariables(
            twist_root_delta_deg=float(
                np.clip(
                    design.twist_root_delta_deg,
                    opt.twist_root_delta_min_deg,
                    opt.twist_root_delta_max_deg,
                )
            ),
            twist_tip_delta_deg=float(
                np.clip(
                    design.twist_tip_delta_deg,
                    opt.twist_tip_delta_min_deg,
                    opt.twist_tip_delta_max_deg,
                )
            ),
            thickness_root_scale=float(
                np.clip(
                    design.thickness_root_scale,
                    opt.thickness_root_scale_min,
                    opt.thickness_root_scale_max,
                )
            ),
            thickness_tip_scale=float(
                np.clip(
                    design.thickness_tip_scale,
                    opt.thickness_tip_scale_min,
                    opt.thickness_tip_scale_max,
                )
            ),
        )

    def evaluate(
        self,
        design: DesignVariables,
        iteration_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> Dict[str, object]:
        """Run coupled fixed-point analysis and return objective/constraint metrics."""
        self.eval_count += 1
        cfg = self.config
        design = self._bounded_design(design)
        cb = iteration_callback if iteration_callback is not None else self.iteration_callback
        self._emit_event(
            {
                "event": "evaluation_start",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_id": self.eval_count,
                "design": asdict(design),
            }
        )

        t_root = cfg.structure.thickness_root_m * design.thickness_root_scale
        t_tip = cfg.structure.thickness_tip_m * design.thickness_tip_scale
        thickness_dist = spanwise_thickness_distribution(self.mesh, t_root, t_tip)
        self.structure.set_thickness_distribution(thickness_dist)

        num_nodes = self.mesh.num_nodes
        aero_loads_prev = np.zeros((num_nodes, 3), dtype=float)
        struct_disps = np.zeros((num_nodes, 3), dtype=float)
        struct_disps_prev = np.zeros_like(struct_disps)

        history: List[CouplingIterationRecord] = []
        aero_result = None
        struct_result: StructuralResult | None = None
        converged = False

        for it in range(1, cfg.coupling.max_iterations + 1):
            aero_twist_delta = compute_aeroelastic_row_twist_delta_deg(self.mesh, struct_disps)
            span_grid = build_spanwise_aero_grid(
                self.mesh,
                twist_root_delta_deg=design.twist_root_delta_deg,
                twist_tip_delta_deg=design.twist_tip_delta_deg,
                aeroelastic_twist_row_deg=aero_twist_delta,
            )
            aero_result = self.aero_solver.solve_trimmed(span_grid)

            aero_nodal_target = distribute_panel_forces_to_nodal_loads(
                self.mesh, aero_result.panel_forces_xyz
            )
            aero_nodal = (
                cfg.coupling.load_relaxation * aero_nodal_target
                + (1.0 - cfg.coupling.load_relaxation) * aero_loads_prev
            )

            struct_loads = self.transfer.transfer_loads(aero_nodal)
            if not np.all(np.isfinite(struct_loads)):
                raise RuntimeError(
                    f"Non-finite structural loads after transfer at coupling iteration {it}"
                )
            struct_result = self.structure.solve(struct_loads, self.mesh.tip_node_indices)
            if not np.all(np.isfinite(struct_result.nodal_displacements_xyz)):
                raise RuntimeError(
                    f"Non-finite structural displacements at coupling iteration {it}"
                )

            struct_disps_target = struct_result.nodal_displacements_xyz
            struct_disps = (
                cfg.coupling.displacement_relaxation * struct_disps_target
                + (1.0 - cfg.coupling.displacement_relaxation) * struct_disps
            )
            aero_disps = self.transfer.transfer_displacements(struct_disps)
            if not np.all(np.isfinite(aero_disps)):
                raise RuntimeError(
                    f"Non-finite aerodynamic displacements after transfer at coupling iteration {it}"
                )

            load_rel = float(
                np.linalg.norm(aero_nodal - aero_loads_prev)
                / max(np.linalg.norm(aero_nodal), 1.0)
            )
            disp_rel = float(
                np.linalg.norm(struct_disps - struct_disps_prev)
                / max(np.linalg.norm(struct_disps), 1.0)
            )

            history.append(
                CouplingIterationRecord(
                    iteration=it,
                    alpha_deg=aero_result.alpha_deg,
                    cl=aero_result.cl,
                    cd=aero_result.cd,
                    lift_n=aero_result.lift_n,
                    drag_n=aero_result.drag_n,
                    mass_kg=struct_result.mass_kg,
                    ks_failure=struct_result.ks_failure,
                    tip_deflection_m=struct_result.tip_deflection_m,
                    aero_load_norm=float(np.linalg.norm(aero_nodal)),
                    struct_load_norm=float(np.linalg.norm(struct_loads)),
                    struct_disp_norm=float(np.linalg.norm(struct_disps)),
                    aero_disp_norm=float(np.linalg.norm(aero_disps)),
                    load_rel_change=load_rel,
                    disp_rel_change=disp_rel,
                    handoff_finite=bool(
                        np.all(np.isfinite(aero_nodal))
                        and np.all(np.isfinite(struct_loads))
                        and np.all(np.isfinite(struct_disps))
                        and np.all(np.isfinite(aero_disps))
                    ),
                    trim_converged=bool(aero_result.trim_converged),
                )
            )
            if cb is not None:
                try:
                    cb(
                        {
                            "event": "coupling_iteration",
                            "evaluation_id": self.eval_count,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "design": asdict(design),
                            "iteration": asdict(history[-1]),
                        }
                    )
                except Exception:
                    # Do not let visualization callbacks break solver execution.
                    pass

            aero_loads_prev = aero_nodal
            struct_disps_prev = struct_disps.copy()

            if load_rel < cfg.coupling.convergence_tol and disp_rel < cfg.coupling.convergence_tol:
                converged = True
                break

        if aero_result is None or struct_result is None:
            raise RuntimeError("Coupled solve produced no results")

        ks_constraint = float(struct_result.ks_failure - 1.0)
        tip_constraint = float(abs(struct_result.tip_deflection_m) - cfg.structure.tip_deflection_limit_m)
        cl_constraint = float(aero_result.cl - cfg.flight.target_cl)
        drag_objective = float(aero_result.cd)

        # Smooth penalty for failed coupling convergence or trim.
        penalty = 0.0
        if not converged:
            penalty += 0.05
        if not aero_result.trim_converged:
            penalty += 0.05
        drag_objective_penalized = drag_objective + penalty

        result = {
            "evaluation_id": self.eval_count,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "design": asdict(design),
            "converged": converged,
            "iterations": [asdict(row) for row in history],
            "objective_cd": drag_objective_penalized,
            "objective_cd_raw": drag_objective,
            "constraints": {
                "ks_failure_minus_1": ks_constraint,
                "tip_deflection_minus_limit_m": tip_constraint,
                "cl_minus_target": cl_constraint,
            },
            "aero": {
                "alpha_deg": aero_result.alpha_deg,
                "cl": aero_result.cl,
                "cd": aero_result.cd,
                "cdi": aero_result.cdi,
                "cd0": aero_result.cd0,
                "lift_n": aero_result.lift_n,
                "drag_n": aero_result.drag_n,
                "l_over_d": aero_result.l_over_d,
                "span_efficiency": aero_result.span_efficiency,
                "reynolds_mean": aero_result.reynolds_mean,
                "trim_converged": aero_result.trim_converged,
            },
            "structure": {
                "mass_kg": struct_result.mass_kg,
                "ks_failure": struct_result.ks_failure,
                "tip_deflection_m": struct_result.tip_deflection_m,
                "thickness_root_m": t_root,
                "thickness_tip_m": t_tip,
            },
        }
        self._emit_event(
            {
                "event": "evaluation_complete",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation_id": self.eval_count,
                "converged": bool(converged),
                "objective_cd": float(result["objective_cd"]),
            }
        )
        return result

    def evaluate_from_array(self, values: np.ndarray) -> Dict[str, object]:
        return self.evaluate(DesignVariables.from_array(values))

    def save_result(self, payload: Dict[str, object], filename: str) -> Path:
        out_path = self.results_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return out_path
