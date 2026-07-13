"""
End-to-end pipeline orchestrator for the PIML Aerostructural MDO.

This is the main entry point that wires everything together:
1. Configuration → problem setup
2. Geometry parameterization
3. Aero solver initialization (NeuralFoil or PINN)
4. Structural solver initialization
5. Aerostructural coupler
6. Optimizer
7. Post-processing and output

The orchestrator manages the full workflow, tracks progress,
handles errors, and produces final results.
"""

import numpy as np
import json
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

from ..aero.airfoil_geometry import AirfoilGeometry
from ..aero.neuralfoil_wrapper import NeuralFoilSolver
from ..aero.openaerostruct_solver import WingPlanform
from ..structures.beam_solver import WingStructure, EulerBernoulliBeamSolver
from ..structures.composite_properties import (
    Laminate, CFRP_IM7_8552, quasi_isotropic, optimized_wing_skin, thick_wing_skin
)
from ..structures.structural_surrogate import StructuralSurrogate
from ..coupling.load_transfer import (
    LoadTransfer, FlightCondition, AerostructuralCoupler
)
from ..optimization.mdo_problem import MDOProblem, MDOProblemSetup, MDOResult
from ..optimization.optimizer import MDOOptimizer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Project
    name: str = "PIML_MDO_Wing"
    output_dir: str = "results"

    # Aero solver
    aero_solver: str = "neuralfoil"  # "neuralfoil", "pinn", "openaerostruct", or "surrogate_cfd"
    neuralfoil_model_size: str = "xlarge"
    pinn_pretrained: Optional[str] = None

    # OpenAeroStruct solver
    openaerostruct_with_viscous: bool = True
    openaerostruct_num_x: int = 5
    openaerostruct_num_y: int = 9

    # Surrogate-CFD solver
    surrogate_cfd_model_dir: str = "results/surrogate_cfd"
    surrogate_cfd_n_samples: int = 60
    surrogate_cfd_hidden_layers: tuple[int, ...] = (128, 64, 32)
    surrogate_cfd_retrain: bool = False

    # Wing geometry
    wing_span: float = 15.0        # Semi-span [m]
    chord_root: float = 3.5        # Root chord [m]
    chord_tip: float = 1.4         # Tip chord [m]
    sweep_deg: float = 25.0        # Quarter-chord sweep [deg]
    n_beam_elements: int = 20      # Structural mesh density

    # Initial airfoil
    initial_airfoil: str = "NACA2412"

    # Material / structure
    material: str = "CFRP_IM7_8552"
    layup: str = "quasi_isotropic"  # or "wing_skin" or "spar_cap"
    structural_solver: str = "vam"  # "clt", "vam", or "surrogate"
    mystran_exe: Optional[str] = None
    structural_surrogate_path: Optional[str] = None
    generate_structural_doe: bool = False

    # Flight condition
    velocity: float = 100.0        # [m/s]
    altitude: float = 3000.0       # [m]
    load_factor_ultimate: float = 2.5

    # Optimization
    optimizer_method: str = "L-BFGS-B"
    max_opt_iterations: int = 100
    cl_target: float = 0.5
    n_laminate_stations: int = 5
    n_ply_angles: int = 4
    optimize_layup: bool = True

    # CST parameterization
    n_cst_upper: int = 6
    n_cst_lower: int = 6
    n_twist_stations: int = 5
    n_struct_sizing: int = 5

    # Aerostructural coupling settings
    coupling_max_iterations: int = 8
    coupling_tolerance: float = 1e-4
    coupling_relaxation: float = 0.3

    # Constraints passed to MDO problem
    max_failure_index: float = 0.8
    max_tip_deflection: float = 2.0
    min_thickness: float = 0.08
    max_thickness: float = 0.20

    # Output / visualization
    run_name: Optional[str] = None
    vtk_export: bool = True
    paraview_screenshots: bool = True
    screenshot_width: int = 1920
    screenshot_height: int = 1080
    pvpython_path: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class PipelineStage:
    """Track a pipeline stage."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    message: str = ""

    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time


class PipelineOrchestrator:
    """Main orchestrator for the PIML aerostructural MDO pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stages = []
        self.result: Optional[MDOResult] = None

        # Components (initialized during setup)
        self.aero_solver = None
        self.wing = None
        self.beam_solver = None
        self.coupler = None
        self.mdo_problem = None
        self.optimizer = None
        self._final_coupling_result = None

        # Result directory: <output_dir>/<run_name>
        run_name = self.config.run_name or self.config.name
        safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in run_name)
        self.run_dir = Path(self.config.output_dir) / safe_name

        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S',
        )

    def _add_stage(self, name: str) -> PipelineStage:
        stage = PipelineStage(name=name)
        self.stages.append(stage)
        return stage

    def _run_stage(self, stage: PipelineStage, func, *args, **kwargs):
        """Run a pipeline stage with timing and error handling."""
        stage.status = "running"
        stage.start_time = time.time()
        logger.info(f"{'='*60}")
        logger.info(f"STAGE: {stage.name}")
        logger.info(f"{'='*60}")

        try:
            result = func(*args, **kwargs)
            stage.status = "completed"
            stage.end_time = time.time()
            stage.message = f"Completed in {stage.elapsed:.1f}s"
            logger.info(f"  -> {stage.message}")
            return result
        except Exception as e:
            stage.status = "failed"
            stage.end_time = time.time()
            stage.message = str(e)
            logger.error(f"  -> FAILED: {e}")
            raise

    def run(self) -> MDOResult:
        """Execute the full MDO pipeline end-to-end.

        Pipeline stages:
        1. Initialize components
        2. Create initial geometry
        3. Baseline analysis
        4. Run optimization
        5. Post-process results
        6. Save outputs
        """
        pipeline_start = time.time()
        logger.info(f"\n{'#'*60}")
        logger.info(f"  PIML Aerostructural MDO Pipeline")
        logger.info(f"  Config: {self.config.name}")
        logger.info(f"{'#'*60}\n")

        # Stage 1: Initialize
        stage = self._add_stage("Initialize Components")
        self._run_stage(stage, self._initialize)

        # Stage 2: Initial Geometry
        stage = self._add_stage("Create Initial Geometry")
        initial_geom = self._run_stage(stage, self._create_initial_geometry)

        # Stage 3: Baseline Analysis
        stage = self._add_stage("Baseline Aerostructural Analysis")
        baseline = self._run_stage(stage, self._baseline_analysis, initial_geom)

        # Stage 4: Optimization
        stage = self._add_stage("MDO Optimization")
        self.result = self._run_stage(stage, self._run_optimization)

        # Stage 5: Post-process
        stage = self._add_stage("Post-Processing")
        self._run_stage(stage, self._post_process)

        # Stage 6: Save
        stage = self._add_stage("Save Results")
        self._run_stage(stage, self._save_results)
        # Defensive: ensure the save stage is always reported as completed.
        stage.status = "completed"
        stage.end_time = stage.end_time or time.time()

        # Write the final stage log now that the save stage is completed.
        self._save_stage_log()

        total_time = time.time() - pipeline_start

        # Print summary
        self._print_summary(total_time, baseline)

        return self.result

    def _initialize(self):
        """Initialize all pipeline components."""
        config = self.config

        # Aero solver
        if config.aero_solver == "neuralfoil":
            self.aero_solver = NeuralFoilSolver(model_size=config.neuralfoil_model_size)
            logger.info(f"  Aero solver: NeuralFoil ({config.neuralfoil_model_size})")
        elif config.aero_solver == "pinn":
            from ..aero.pinn_solver import PINNAeroSolver
            self.aero_solver = PINNAeroSolver(
                n_cst_params=config.n_cst_upper + config.n_cst_lower,
                pretrained_path=config.pinn_pretrained,
            )
            logger.info("  Aero solver: PINN (RANS)")
        elif config.aero_solver == "openaerostruct":
            from ..aero.openaerostruct_solver import OpenAeroStructSolver
            self.aero_solver = OpenAeroStructSolver(
                wing_planform=WingPlanform(
                    span=config.wing_span,
                    chord_root=config.chord_root,
                    chord_tip=config.chord_tip,
                    sweep_deg=config.sweep_deg,
                ),
                num_x=config.openaerostruct_num_x,
                num_y=config.openaerostruct_num_y,
                num_twist_cp=max(2, config.n_twist_stations),
                with_viscous=config.openaerostruct_with_viscous,
            )
            visc = "viscous" if config.openaerostruct_with_viscous else "inviscid"
            logger.info(f"  Aero solver: OpenAeroStruct VLM ({visc})")
        elif config.aero_solver == "surrogate_cfd":
            from ..aero.surrogate_cfd import SurrogateCFDSolver
            self.aero_solver = SurrogateCFDSolver(
                model_dir=config.surrogate_cfd_model_dir,
                wing_planform=WingPlanform(
                    span=config.wing_span,
                    chord_root=config.chord_root,
                    chord_tip=config.chord_tip,
                    sweep_deg=config.sweep_deg,
                ),
                n_samples=config.surrogate_cfd_n_samples,
                hidden_layers=config.surrogate_cfd_hidden_layers,
                retrain=config.surrogate_cfd_retrain,
            )
            if not self.aero_solver.is_trained:
                self.aero_solver.train(
                    n_cst_upper=config.n_cst_upper,
                    n_cst_lower=config.n_cst_lower,
                )
            logger.info("  Aero solver: Surrogate CFD (MLP on OpenAeroStruct)")
        else:
            raise ValueError(f"Unknown aero solver: {config.aero_solver}")

        # Material / laminate
        if config.layup == "quasi_isotropic":
            laminate = quasi_isotropic(CFRP_IM7_8552)
        elif config.layup == "wing_skin":
            laminate = optimized_wing_skin(CFRP_IM7_8552)
        elif config.layup == "thick_wing_skin":
            laminate = thick_wing_skin(CFRP_IM7_8552)
        else:
            laminate = quasi_isotropic(CFRP_IM7_8552)
        logger.info(f"  Material: {laminate.material.name}, Layup: {config.layup}")

        # Wing structure
        use_vam = config.structural_solver in ("vam", "surrogate")
        self.wing = WingStructure(
            span=config.wing_span,
            n_elements=config.n_beam_elements,
            chord_root=config.chord_root,
            chord_tip=config.chord_tip,
            sweep_deg=config.sweep_deg,
            skin_laminate=laminate,
            use_vam=use_vam,
        )
        logger.info(f"  Wing: span={config.wing_span}m, "
                    f"chord={config.chord_root}-{config.chord_tip}m")
        logger.info(f"  Structural solver: {config.structural_solver}")

        # Optional MYSTRAN-trained structural surrogate
        surrogate = None
        surrogate_path = None
        if config.structural_solver == "surrogate":
            surrogate_path = Path(config.structural_surrogate_path or "results/structural_doe/surrogate.pt")
        elif config.structural_surrogate_path:
            surrogate_path = Path(config.structural_surrogate_path)

        if surrogate_path is not None:
            if surrogate_path.exists():
                surrogate = StructuralSurrogate.load(surrogate_path)
                logger.info(f"  Loaded structural surrogate: {surrogate_path}")
            else:
                logger.warning(f"  Surrogate path not found: {surrogate_path}")

        # Beam solver
        self.beam_solver = EulerBernoulliBeamSolver(self.wing, surrogate=surrogate)

        # Coupler
        n_stations = config.n_beam_elements + 1
        self.coupler = AerostructuralCoupler(
            aero_solver=self.aero_solver,
            beam_solver=self.beam_solver,
            wing_structure=self.wing,
            n_stations=n_stations,
            max_iterations=config.coupling_max_iterations,
            tolerance=config.coupling_tolerance,
            relaxation=config.coupling_relaxation,
        )

        # Compute baseline ply counts from the symmetric half-stack so the
        # optimizer starts from the same laminate used in the baseline analysis.
        baseline_ply_counts = None
        if config.optimize_layup and laminate is not None:
            from collections import Counter
            half_angles = laminate.angles
            counts = Counter(half_angles)
            baseline_ply_counts = {float(a): float(c) for a, c in counts.items()}
            logger.info(f"  Baseline ply counts (half-stack): {dict(counts)}")

        # MDO problem
        mdo_setup = MDOProblemSetup(
            n_cst_upper=config.n_cst_upper,
            n_cst_lower=config.n_cst_lower,
            n_twist=config.n_twist_stations,
            n_struct_sizing=config.n_struct_sizing,
            cl_target=config.cl_target,
            n_laminate_stations=config.n_laminate_stations,
            n_ply_angles=config.n_ply_angles,
            optimize_layup=config.optimize_layup,
            baseline_ply_counts=baseline_ply_counts,
            max_failure_index=config.max_failure_index,
            max_tip_deflection=config.max_tip_deflection,
            min_thickness=config.min_thickness,
            max_thickness=config.max_thickness,
        )

        self.mdo_problem = MDOProblem(
            setup=mdo_setup,
            aero_solver=self.aero_solver,
            beam_solver=self.beam_solver,
            coupler=self.coupler,
            airfoil_geometry_class=AirfoilGeometry,
            wing_structure=self.wing,
            flight_condition=FlightCondition(
                velocity=config.velocity,
                altitude=config.altitude,
                alpha=3.0,
                load_factor=1.0,
            ),
        )

        # Optimizer
        self.optimizer = MDOOptimizer(
            method=config.optimizer_method,
            max_iterations=config.max_opt_iterations,
        )

        logger.info(f"  Optimizer: {config.optimizer_method}, "
                    f"max_iter={config.max_opt_iterations}")

    def _create_initial_geometry(self) -> AirfoilGeometry:
        """Create initial airfoil geometry."""
        code = self.config.initial_airfoil.replace("NACA", "")
        geom = AirfoilGeometry.naca4(code, n_weights=self.config.n_cst_upper)
        logger.info(f"  Initial airfoil: {self.config.initial_airfoil}")
        logger.info(f"  CST params: {geom.n_params} "
                    f"({self.config.n_cst_upper} upper + {self.config.n_cst_lower} lower)")
        logger.info(f"  Max thickness: {geom.max_thickness():.4f}")
        logger.info(f"  Max camber: {geom.max_camber():.4f}")
        return geom

    def _baseline_analysis(self, geom: AirfoilGeometry) -> dict:
        """Run baseline aerostructural analysis on initial geometry."""
        coords = geom.full_coordinates(150)
        n_stations = self.config.n_beam_elements + 1
        twist = np.linspace(0.0, -2.0, n_stations)  # mild washout

        flight = FlightCondition(
            velocity=self.config.velocity,
            altitude=self.config.altitude,
            alpha=3.0,
            load_factor=1.0,
        )

        result = self.coupler.solve(coords, twist, flight)

        cl_mean = float(np.mean(result['aero']['cl']))
        cd_mean = float(np.mean(result['aero']['cd']))
        ld = cl_mean / cd_mean if cd_mean > 1e-10 else 0.0

        baseline = {
            'cl': cl_mean,
            'cd': cd_mean,
            'ld': ld,
            'mass': result['structure'].total_mass,
            'failure_index': result['structure'].failure_index,
            'tip_deflection': result['structure'].tip_deflection,
            'converged': result['converged'],
        }

        logger.info(f"  Baseline CL: {cl_mean:.4f}")
        logger.info(f"  Baseline CD: {cd_mean:.6f}")
        logger.info(f"  Baseline L/D: {ld:.1f}")
        logger.info(f"  Structural mass: {baseline['mass']:.1f} kg")
        logger.info(f"  Failure index: {baseline['failure_index']:.3f}")
        logger.info(f"  Tip deflection: {baseline['tip_deflection']:.4f} m")
        logger.info(f"  Coupling converged: {baseline['converged']}")

        return baseline

    def _run_optimization(self) -> MDOResult:
        """Run the MDO optimization."""
        return self.optimizer.optimize(self.mdo_problem)

    def _post_process(self):
        """Post-process optimization results."""
        if self.result is None:
            return

        # Generate final airfoil coordinates
        d = self.result.design_dict
        geom = AirfoilGeometry(
            upper_weights=d['cst_upper'],
            lower_weights=d['cst_lower'],
        )
        x_full, y_full = geom.full_coordinates(200)

        # Save airfoil coordinates
        self.run_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            self.run_dir / "optimized_airfoil.dat",
            np.column_stack([x_full, y_full]),
            header="x y",
            fmt="%.6f",
        )

        logger.info(f"  Saved optimized airfoil to {self.run_dir / 'optimized_airfoil.dat'}")

        # Run a final aerostructural analysis for VTK / screenshot generation
        coupling_result = None
        if self.config.vtk_export or self.config.paraview_screenshots:
            try:
                n_stations = self.config.n_beam_elements + 1
                twist_stations = np.linspace(0.0, self.config.wing_span, self.config.n_twist_stations)
                beam_stations = self.wing.spanwise_stations()
                twist_interp = np.interp(beam_stations, twist_stations, d['twist'])

                flight = FlightCondition(
                    velocity=self.config.velocity,
                    altitude=self.config.altitude,
                    alpha=float(d.get('alpha', 3.0)),
                    load_factor=1.0,
                )
                coupling_result = self.coupler.solve(
                    airfoil_coords=geom.full_coordinates(150),
                    initial_twist=twist_interp,
                    flight_condition=flight,
                )
            except Exception as exc:
                logger.warning("Final aerostructural analysis for VTK export failed: %s", exc)
                coupling_result = None

        # Export VTK datasets
        self._final_coupling_result = coupling_result
        if self.config.vtk_export and coupling_result is not None:
            try:
                from ..utils.vtk_export import export_aerostructural_result
                vtk_base = self.run_dir / f"{self.config.name}_wing"
                files = export_aerostructural_result(coupling_result, self.wing, str(vtk_base))
                for f in files:
                    logger.info(f"  Exported VTK: {f}")
            except Exception as exc:
                logger.warning("VTK export failed: %s", exc)

    def _save_results(self):
        """Save all results to files."""
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_json(str(self.run_dir / "config.json"))

        # Save optimization history
        if self.result and self.result.history:
            history_data = []
            for h in self.result.history:
                row = {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in h.items()}
                history_data.append(row)
            with open(self.run_dir / "optimization_history.json", 'w') as f:
                json.dump(history_data, f, indent=2)

        # Save summary
        if self.result:
            summary = {
                'success': self.result.success,
                'objective': self.result.objective_value,
                'cl': self.result.cl,
                'cd': self.result.cd,
                'ld_ratio': self.result.ld_ratio,
                'structural_mass': self.result.structural_mass,
                'failure_index': self.result.failure_index,
                'tip_deflection': self.result.tip_deflection,
                'n_iterations': self.result.n_iterations,
                'n_evals': self.result.n_function_evals,
                'wall_time': self.result.wall_time,
                'design_vars': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.result.design_dict.items()
                },
            }
            with open(self.run_dir / "optimization_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

        # Generate ParaView screenshots after all JSON/VTK outputs are saved
        if self.config.paraview_screenshots:
            try:
                from ..utils.paraview_screenshots import generate_screenshots
                screenshots = generate_screenshots(self.run_dir, config=self.config)
                if screenshots:
                    logger.info(f"  Generated {len(screenshots)} ParaView screenshot(s)")
            except Exception as exc:
                logger.warning("ParaView screenshot generation failed: %s", exc)

        logger.info(f"  Results saved to {self.run_dir}/")

    def _save_stage_log(self):
        """Write the pipeline stage log to JSON.

        This is called after the save stage is finalized so that the save
        stage itself is recorded as completed rather than running.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stage_log = []
        for s in self.stages:
            stage_log.append({
                'name': s.name,
                'status': s.status,
                'elapsed': s.elapsed,
                'message': s.message,
            })
        with open(self.run_dir / "pipeline_stages.json", 'w') as f:
            json.dump(stage_log, f, indent=2)

    def _print_summary(self, total_time: float, baseline: dict):
        """Print final summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  PIPELINE SUMMARY")
        logger.info(f"{'='*60}")

        logger.info(f"\n  Pipeline stages:")
        for s in self.stages:
            status_icon = {"completed": "[OK]", "failed": "[FAIL]", "running": "[...]"}.get(s.status, "[?]")
            logger.info(f"    {status_icon} {s.name}: {s.elapsed:.1f}s")

        if self.result:
            logger.info(f"\n  Baseline → Optimized:")
            logger.info(f"    CL:    {baseline['cl']:.4f} → {self.result.cl:.4f}")
            logger.info(f"    CD:    {baseline['cd']:.6f} → {self.result.cd:.6f}")
            logger.info(f"    L/D:   {baseline['ld']:.1f} → {self.result.ld_ratio:.1f}")
            logger.info(f"    Mass:  {baseline['mass']:.1f} → {self.result.structural_mass:.1f} kg")
            logger.info(f"    FI:    {baseline['failure_index']:.3f} → {self.result.failure_index:.3f}")

            if baseline['cd'] > 1e-10:
                cd_improvement = (1.0 - self.result.cd / baseline['cd']) * 100
                logger.info(f"\n  Drag reduction: {cd_improvement:.1f}%")

            if baseline['ld'] > 1e-10:
                ld_improvement = (self.result.ld_ratio / baseline['ld'] - 1.0) * 100
                logger.info(f"  L/D improvement: {ld_improvement:.1f}%")

        logger.info(f"\n  Total wall time: {total_time:.1f}s")
        logger.info(f"  Function evaluations: {self.result.n_function_evals if self.result else 0}")
        logger.info(f"{'='*60}\n")
