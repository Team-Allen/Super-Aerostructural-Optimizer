#!/usr/bin/env python3
"""
Validate that all pipeline components are installed and working.

Runs quick tests on each module to verify the installation.
"""

import sys
import traceback
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name: str, func):
    """Run a check and report pass/fail."""
    try:
        result = func()
        print(f"  {PASS} {name}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        traceback.print_exc(limit=1)
        return False


def main():
    print("\n  PIML MDO Pipeline — Installation Validation")
    print("  " + "=" * 50)

    results = []

    # 1. Core imports
    print("\n  1. Core Dependencies")
    results.append(check("numpy", lambda: f"v{__import__('numpy').__version__}"))
    results.append(check("scipy", lambda: f"v{__import__('scipy').__version__}"))

    # 2. Aerodynamic Solvers
    print("\n  2. Aerodynamic Solver")
    results.append(check("neuralfoil", lambda: f"v{__import__('neuralfoil').__version__}"))

    def check_openaerostruct():
        import openaerostruct
        import openmdao
        return f"OAS {openaerostruct.__version__}, OM {openmdao.__version__}"

    try:
        results.append(check("openaerostruct", check_openaerostruct))
    except Exception:
        print(f"  {WARN} openaerostruct: Not available (optional)")

    # 3. PyTorch
    print("\n  3. Machine Learning")

    def check_torch():
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        return f"v{torch.__version__} ({device})"

    results.append(check("pytorch", check_torch))

    # 4. Airfoil Geometry
    print("\n  4. Pipeline Modules")

    def check_geometry():
        from piml_mdo.aero.airfoil_geometry import AirfoilGeometry
        geom = AirfoilGeometry.naca4("2412")
        x, y_u, y_l = geom.coordinates(50)
        assert len(x) == 50
        assert geom.max_thickness() > 0.1
        return f"NACA2412, t/c={geom.max_thickness():.3f}, {geom.n_params} CST params"

    results.append(check("airfoil_geometry", check_geometry))

    def check_neuralfoil_solver():
        from piml_mdo.aero.neuralfoil_wrapper import NeuralFoilSolver
        from piml_mdo.aero.airfoil_geometry import AirfoilGeometry
        solver = NeuralFoilSolver(model_size="small")
        geom = AirfoilGeometry.naca4("0012")
        coords = geom.full_coordinates(100)
        result = solver.evaluate(coords, alpha=5.0, Re=1e6)
        return f"CL={result.cl:.4f}, CD={result.cd:.6f}, L/D={result.ld_ratio:.1f}"

    results.append(check("neuralfoil_solver", check_neuralfoil_solver))

    def check_composite():
        from piml_mdo.structures.composite_properties import quasi_isotropic, CFRP_IM7_8552
        lam = quasi_isotropic(CFRP_IM7_8552)
        ABD = lam.ABD_matrix()
        props = lam.equivalent_beam_properties(width=0.5)
        return f"{lam.n_plies} plies, EI={props['EI']:.1f} N·m²"

    results.append(check("composite_properties", check_composite))

    def check_beam():
        import numpy as np
        from piml_mdo.structures.beam_solver import WingStructure, EulerBernoulliBeamSolver
        wing = WingStructure(span=15.0, n_elements=10)
        solver = EulerBernoulliBeamSolver(wing)
        lift = np.ones(11) * 5000.0  # 5 kN/m
        result = solver.solve(lift)
        return f"tip_def={result.tip_deflection:.3f}m, mass={result.total_mass:.1f}kg"

    results.append(check("beam_solver", check_beam))

    def check_coupling():
        from piml_mdo.coupling.load_transfer import LoadTransfer, FlightCondition
        lt = LoadTransfer(wing_span=15.0, n_stations=11)
        lift = lt.elliptic_lift_distribution(total_lift=50000.0)
        assert len(lift) == 11
        assert lift[0] > lift[-1]  # Higher at root
        fc = FlightCondition(velocity=100.0, altitude=3000.0, alpha=3.0)
        return f"q={fc.dynamic_pressure:.0f}Pa, Re/m={fc.reynolds:.0f}, M={fc.mach:.3f}"

    results.append(check("coupling", check_coupling))

    def check_mdo_setup():
        from piml_mdo.optimization.mdo_problem import MDOProblemSetup
        setup = MDOProblemSetup()
        dvs = setup.create_design_variables()
        return f"{setup.n_design_vars} design variables"

    results.append(check("mdo_problem", check_mdo_setup))

    def check_layup_design():
        from piml_mdo.optimization.mdo_problem import MDOProblemSetup
        from piml_mdo.structures.beam_solver import WingStructure
        from piml_mdo.structures.composite_properties import Laminate

        setup = MDOProblemSetup(optimize_layup=True)
        dvs = setup.create_design_variables()
        assert len(dvs) == setup.n_design_vars

        # Round-trip design vector
        x = np.array([dv.value for dv in dvs])
        d = setup.design_vector_to_dict(x)
        x2 = setup.dict_to_design_vector(d)
        assert np.allclose(x, x2)

        # Build per-station laminates from ply counts
        wing = WingStructure(span=10.0, n_elements=10)
        base_material = wing.skin_laminate.material
        counts = np.array([[1, 1, 1, 1]] * (wing.n_elements + 1))
        laminates = []
        for i in range(wing.n_elements + 1):
            angles_half = []
            for j, angle in enumerate(setup.ply_angles):
                angles_half.extend([angle] * int(counts[i, j]))
            laminates.append(
                Laminate(material=base_material, angles=angles_half, symmetric=True)
            )
        wing.set_skin_laminates(laminates)
        props = wing.section_properties()
        assert 'skin_thickness' in props
        return f"{setup.n_design_vars} DVs, per-station laminates OK"

    results.append(check("laminate_design", check_layup_design))

    def check_orchestrator():
        from piml_mdo.pipeline.orchestrator import PipelineConfig
        config = PipelineConfig()
        return f"Config OK: {config.name}"

    results.append(check("orchestrator", check_orchestrator))

    # Optional
    print("\n  5. Optional Dependencies")

    def check_matplotlib():
        import matplotlib
        return f"v{matplotlib.__version__}"
    results.append(check("matplotlib", check_matplotlib))

    def check_yaml():
        import yaml
        return "OK"
    results.append(check("pyyaml", check_yaml))

    try:
        import pyoptsparse
        results.append(check("pyoptsparse", lambda: f"v{pyoptsparse.__version__}"))
    except ImportError:
        print(f"  {WARN} pyoptsparse: Not installed (optional)")

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n  {'=' * 50}")
    print(f"  Results: {n_pass}/{n_total} checks passed")

    if n_pass == n_total:
        print(f"  All checks PASSED — pipeline ready to run!")
    else:
        print(f"  {n_total - n_pass} checks FAILED — fix issues above before running pipeline")

    print()
    return 0 if n_pass >= n_total - 2 else 1  # Allow 2 optional failures


if __name__ == "__main__":
    sys.exit(main())
