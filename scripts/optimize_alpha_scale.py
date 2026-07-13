#!/usr/bin/env python3
"""
Direct sizing + alpha trim optimization for the aircraft wing.

Uses the full VAM beam / NeuralFoil coupling but only two design variables:
- angle of attack (trim)
- uniform skin-thickness scale (sizing)

This removes the integer-ply-count noise and CST/twist degrees of freedom so a
feasible, low-drag design can be found in minutes rather than hours.  The
result is saved to the same output layout used by the full MDO pipeline.
"""

import sys
import argparse
import logging
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scripts.run_pipeline as rp
from piml_mdo.pipeline.orchestrator import PipelineOrchestrator
from piml_mdo.coupling.load_transfer import FlightCondition
from piml_mdo.aero.airfoil_geometry import AirfoilGeometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("optimize_alpha_scale")


def parse_args():
    parser = argparse.ArgumentParser(description="Direct alpha + sizing optimization")
    parser.add_argument("--config", type=str, default="config/piml_aerostruct_run.yaml",
                        help="Pipeline config YAML")
    parser.add_argument("--run-name", type=str, default="wing_vam_alpha_scale_optimized",
                        help="Output run directory name")
    parser.add_argument("--maxiter", type=int, default=80,
                        help="COBYLA function-evaluation budget")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # Load config and force sizing-only / VAM / no VTK during optimization
    cfg_args = argparse.Namespace(
        config=args.config, quick=False, output="results", solver=None,
        structural_solver=None, method=None, max_iter=None, verbose=False,
    )
    config = rp.load_config(cfg_args)
    config.optimize_layup = False
    config.coupling_max_iterations = 2
    config.coupling_tolerance = 1.0e-3
    config.vtk_export = False
    config.paraview_screenshots = False
    config.run_name = args.run_name

    # Initialize pipeline components (aero, VAM beam, coupler)
    pipe = PipelineOrchestrator(config)
    pipe._initialize()

    geom = pipe._create_initial_geometry()
    coords = geom.full_coordinates(150)
    n_stations = config.n_beam_elements + 1
    twist = np.linspace(0.0, -2.0, n_stations)

    wing = pipe.wing
    coupler = pipe.coupler
    cl_target = config.cl_target
    max_fi = config.max_failure_index
    max_tip = config.max_tip_deflection
    min_t = config.min_thickness
    max_t = config.max_thickness

    stations = wing.spanwise_stations()
    chord = wing.chord_distribution()
    wing_area = float(np.trapz(chord, stations))

    history = []
    cache = {}

    def evaluate(x):
        key = (round(float(x[0]), 8), round(float(x[1]), 8))
        if key in cache:
            return cache[key]

        alpha, scale = x
        wing.set_skin_thickness_scales(np.full(n_stations, scale))
        flight = FlightCondition(
            velocity=config.velocity,
            altitude=config.altitude,
            alpha=float(alpha),
            load_factor=1.0,
        )

        result = coupler.solve(coords, twist, flight)
        aero = result["aero"]
        struct = result["structure"]

        q = flight.dynamic_pressure
        total_lift = float(np.trapz(aero["lift"], stations))
        total_drag = float(np.trapz(aero["drag"], stations))
        cl = total_lift / (q * wing_area) if wing_area > 1e-10 else 0.0
        cd = total_drag / (q * wing_area) if wing_area > 1e-10 else 0.0
        ld = cl / cd if cd > 1e-10 else 0.0

        mass = struct.total_mass
        fi = struct.failure_index
        tip = abs(struct.tip_deflection)
        tc = geom.max_thickness()

        obj = cd + 0.1 * mass / 1000.0
        penalty = 0.0
        cl_err = abs(cl - cl_target)
        if cl_err > 0.01:
            penalty += 10000.0 * cl_err ** 2
        if fi > max_fi:
            penalty += 100.0 * (fi - max_fi) ** 2
        if tip > max_tip:
            penalty += 100.0 * (tip - max_tip) ** 2
        if tc < min_t:
            penalty += 100.0 * (min_t - tc) ** 2
        if tc > max_t:
            penalty += 100.0 * (tc - max_t) ** 2

        total = obj + penalty
        cache[key] = total

        record = {
            "eval": len(history) + 1,
            "objective": total,
            "cl": cl,
            "cd": cd,
            "ld": ld,
            "mass": mass,
            "fi": fi,
            "tip": tip,
            "alpha": alpha,
            "scale": scale,
        }
        history.append(record)
        logger.info(
            f"Eval {record['eval']:3d}: obj={total:.6f} CL={cl:.4f} CD={cd:.6f} "
            f"L/D={ld:.1f} mass={mass:.1f}kg FI={fi:.3f} tip={tip:.3f}m "
            f"alpha={alpha:.3f} scale={scale:.3f}"
        )
        return total

    x0 = np.array([3.0, 1.0])
    bounds = [(-2.0, 12.0), (0.5, 3.0)]
    logger.info("Starting direct alpha + uniform-thickness-scale optimization")
    res = minimize(
        evaluate,
        x0,
        method="COBYLA",
        bounds=bounds,
        options={"maxiter": args.maxiter, "rhobeg": 0.2, "tol": 1e-4},
    )

    alpha_opt, scale_opt = res.x
    logger.info(
        f"Optimization finished in {time.time() - t0:.1f}s "
        f"({len(history)} evals, success={res.success})"
    )
    logger.info(f"Optimal alpha={alpha_opt:.4f} deg, scale={scale_opt:.4f}")

    # Final high-fidelity re-evaluation with full coupling
    logger.info("Running final converged re-evaluation with full coupling...")
    pipe.config.coupling_max_iterations = 8
    pipe.coupler.max_iterations = 8
    wing.set_skin_thickness_scales(np.full(n_stations, scale_opt))
    flight = FlightCondition(
        velocity=config.velocity,
        altitude=config.altitude,
        alpha=float(alpha_opt),
        load_factor=1.0,
    )
    final = coupler.solve(coords, twist, flight)
    aero = final["aero"]
    struct = final["structure"]

    q = flight.dynamic_pressure
    cl = float(np.trapz(aero["lift"], stations) / (q * wing_area))
    cd = float(np.trapz(aero["drag"], stations) / (q * wing_area))
    ld = cl / cd if cd > 1e-10 else 0.0

    logger.info("Final converged result:")
    logger.info(f"  CL={cl:.4f}  CD={cd:.6f}  L/D={ld:.1f}")
    logger.info(f"  Mass={struct.total_mass:.1f} kg  FI={struct.failure_index:.3f}  Tip={struct.tip_deflection:.4f} m")

    # Save a concise summary in the run directory
    run_dir = pipe.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "direct_alpha_scale_COBYLA",
        "maxiter": args.maxiter,
        "n_evals": len(history),
        "optimal_alpha_deg": float(alpha_opt),
        "optimal_thickness_scale": float(scale_opt),
        "cl": cl,
        "cd": cd,
        "ld": ld,
        "mass_kg": struct.total_mass,
        "failure_index": struct.failure_index,
        "tip_deflection_m": struct.tip_deflection,
        "coupling_converged": final["converged"],
        "coupling_iterations": final["iterations"],
        "wall_time_s": time.time() - t0,
        "history": history,
    }
    import json
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary written to {run_dir / 'summary.json'}")

    # Optional VTK / screenshots using the pipeline post-processor
    if config.vtk_export or config.paraview_screenshots:
        pipe.result = pipe.mdo_problem.get_result(res.x)
        pipe._post_process()
        pipe._save_results()

    return summary


if __name__ == "__main__":
    main()
