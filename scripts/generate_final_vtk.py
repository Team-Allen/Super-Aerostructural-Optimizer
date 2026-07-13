#!/usr/bin/env python3
"""
Generate VTK export and optional ParaView screenshots for a given final
(alpha, thickness_scale) design.
"""

import sys
import argparse
import logging
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import scripts.run_pipeline as rp
from piml_mdo.pipeline.orchestrator import PipelineOrchestrator
from piml_mdo.coupling.load_transfer import FlightCondition
from piml_mdo.utils.vtk_export import export_aerostructural_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("generate_final_vtk")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/piml_aerostruct_run.yaml")
    parser.add_argument("--run-name", default="wing_vam_alpha_scale_optimized")
    parser.add_argument("--alpha", type=float, default=1.60)
    parser.add_argument("--scale", type=float, default=1.3343589351105662)
    parser.add_argument("--screenshots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_args = argparse.Namespace(
        config=args.config, quick=False, output="results", solver=None,
        structural_solver=None, method=None, max_iter=None, verbose=False,
    )
    config = rp.load_config(cfg_args)
    config.optimize_layup = False
    config.coupling_max_iterations = 8
    config.coupling_tolerance = 1e-3
    config.vtk_export = True
    config.paraview_screenshots = args.screenshots
    config.run_name = args.run_name

    pipe = PipelineOrchestrator(config)
    pipe._initialize()

    geom = pipe._create_initial_geometry()
    coords = geom.full_coordinates(150)
    n_stations = config.n_beam_elements + 1
    twist = np.linspace(0.0, -2.0, n_stations)

    pipe.wing.set_skin_thickness_scales(np.full(n_stations, args.scale))
    flight = FlightCondition(
        velocity=config.velocity,
        altitude=config.altitude,
        alpha=args.alpha,
        load_factor=1.0,
    )
    logger.info("Running final converged aerostructural analysis...")
    result = pipe.coupler.solve(coords, twist, flight)
    logger.info(
        f"CL={result['aero']['cl'].mean():.4f} "
        f"CD={result['aero']['cd'].mean():.6f} "
        f"FI={result['structure'].failure_index:.3f} "
        f"tip={result['structure'].tip_deflection:.4f}m"
    )

    run_dir = pipe.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    vtk_base = run_dir / f"{config.name}_wing"
    files = export_aerostructural_result(result, pipe.wing, str(vtk_base))
    for f in files:
        logger.info(f"Exported VTK: {f}")

    if args.screenshots:
        try:
            from piml_mdo.utils.paraview_screenshots import generate_screenshots
            screenshots = generate_screenshots(run_dir, config=config)
            logger.info(f"Generated {len(screenshots)} screenshot(s)")
        except Exception as exc:
            logger.warning(f"Screenshot generation failed: {exc}")


if __name__ == "__main__":
    main()
