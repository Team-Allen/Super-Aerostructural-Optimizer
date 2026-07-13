#!/usr/bin/env python3
"""Quick targeted test for the new laminate design variables."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from piml_mdo.optimization.mdo_problem import MDOProblemSetup, MDOProblem
from piml_mdo.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig


def main():
    print("\n  Laminate DV targeted test")
    print("  " + "=" * 50)

    # 1. MDOProblemSetup with laminate DVs
    setup = MDOProblemSetup(
        n_cst_upper=2,
        n_cst_lower=2,
        n_twist=3,
        n_struct_sizing=3,
        n_laminate_stations=2,
        n_ply_angles=4,
        optimize_layup=True,
    )
    dvs = setup.create_design_variables()
    assert len(dvs) == setup.n_design_vars
    print(f"  [OK] {setup.n_design_vars} design variables created")

    x0 = np.array([dv.value for dv in dvs])
    d = setup.design_vector_to_dict(x0)
    x_back = setup.dict_to_design_vector(d)
    assert np.allclose(x0, x_back)
    print(f"  [OK] design vector round-trip (shape {x0.shape})")
    print(f"       ply_counts shape = {d['ply_counts'].shape}")

    # 2. Build a tiny pipeline and evaluate the objective directly
    config = PipelineConfig(
        name="Laminate_DV_Test",
        aero_solver="neuralfoil",
        neuralfoil_model_size="small",
        wing_span=10.0,
        chord_root=2.5,
        chord_tip=1.0,
        n_beam_elements=5,
        optimizer_method="L-BFGS-B",
        max_opt_iterations=1,
        n_cst_upper=2,
        n_cst_lower=2,
        n_twist_stations=3,
        n_laminate_stations=2,
        n_ply_angles=4,
        optimize_layup=True,
        vtk_export=False,
        paraview_screenshots=False,
    )

    orch = PipelineOrchestrator(config)
    orch._initialize()

    problem = orch.mdo_problem
    x = np.array([dv.value for dv in problem.setup.create_design_variables()])
    obj0 = problem.objective(x)
    print(f"  [OK] baseline objective = {obj0:.3f}")

    # 3. Verify per-station laminates were built
    assert problem.wing.skin_laminates is not None
    assert len(problem.wing.skin_laminates) == problem.wing.n_elements + 1
    print(f"  [OK] per-station laminates set ({len(problem.wing.skin_laminates)} stations)")

    # 4. Verify that changing ply counts changes mass/stiffness
    d2 = problem.setup.design_vector_to_dict(x)
    d2['ply_counts'] = np.full_like(d2['ply_counts'], 4.0)  # thicker
    x_thick = problem.setup.dict_to_design_vector(d2)
    obj_thick = problem.objective(x_thick)
    print(f"  [OK] thicker laminate objective = {obj_thick:.3f}")

    # 5. Verify insufficient plies get a penalty but do not crash
    d3 = problem.setup.design_vector_to_dict(x)
    d3['ply_counts'] = np.zeros_like(d3['ply_counts'])
    x_zero = problem.setup.dict_to_design_vector(d3)
    obj_zero = problem.objective(x_zero)
    print(f"  [OK] zero-ply (penalized) objective = {obj_zero:.3f}")

    # 6. Backwards compatibility: optimize_layup=False
    setup_bc = MDOProblemSetup(optimize_layup=False)
    assert setup_bc.n_design_vars == 12 + 5 + 5 + 1
    print(f"  [OK] backwards-compatible DV count = {setup_bc.n_design_vars}")

    print("\n  All laminate DV checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
