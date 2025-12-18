"""
Optimize 3D wing using a trained GNN surrogate.

This script provides a light-weight CMA-ES driver that optimizes
`WingParameters` (span, taper, sweep, dihedral, twist_root, twist_tip)
to maximize L/D predicted by the GNN-based analyzer.

It will use CUDA if available for GNN inference.
"""
import argparse
import json
import numpy as np
import os
import time

try:
    import cma
    HAS_CMA = True
except Exception:
    HAS_CMA = False

from pathlib import Path
import torch

from src.aerodynamics_3d.wing_geometry import WingParameters, Wing3D
from src.aerodynamics_3d.gnn_wing_analyzer import GNNWingAnalyzer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=False, help='Path to GNN checkpoint (optional)')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--budget', type=int, default=200, help='CMA-ES budget (function evaluations)')
    p.add_argument('--popsize', type=int, default=16, help='CMA-ES population size')
    p.add_argument('--out', type=str, default='results/stage2', help='Output directory')
    return p.parse_args()


def objective_array(x, analyzer, flight_conditions):
    # x: [span, taper_ratio, sweep_angle, dihedral, twist_root, twist_tip]
    params = WingParameters.from_array(x)
    wing = Wing3D(params)

    # Run analyzer (fast inference if GNN is trained)
    res = analyzer.analyze_wing(wing, flight_conditions)
    L_D = res.get('L/D', 0.0)

    # Negative because CMA minimizes
    return -float(L_D)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # Create analyzer
    analyzer = GNNWingAnalyzer(args.model, device=device)

    # Flight conditions
    flight_conditions = {'velocity': 50.0, 'altitude': 0.0, 'alpha': 5.0}

    # Initial guess (span, taper, sweep, dihedral, twist_root, twist_tip)
    x0 = np.array([10.0, 0.6, 25.0, 3.0, 2.0, -3.0])
    bounds = np.array([
        [4.0, 0.2, 0.0, 0.0, -5.0, -10.0],  # lb
        [30.0, 1.0, 40.0, 10.0, 10.0, 5.0]    # ub
    ])

    # Quick wrapper for CMA objective
    def obj_fn(x):
        # Clip to bounds
        x = np.clip(x, bounds[0], bounds[1])
        return objective_array(x, analyzer, flight_conditions)

    start = time.time()

    if HAS_CMA:
        sigma0 = 0.2
        es = cma.CMAEvolutionStrategy(list(x0), sigma0, {'popsize': args.popsize, 'bounds': [bounds[0].tolist(), bounds[1].tolist()]})
        eval_count = 0
        best = None
        while not es.stop() and eval_count < args.budget:
            solutions = es.ask()
            fitnesses = []
            for s in solutions:
                f = obj_fn(np.array(s))
                fitnesses.append(f)
                eval_count += 1
            es.tell(solutions, fitnesses)
            es.disp()
            cand = es.result.xbest
            if best is None or min(fitnesses) < best[0]:
                best = (min(fitnesses), cand)
        x_best = np.array(best[1])
        f_best = best[0]
    else:
        # Fall back to simple random search if pycma not installed
        print("pycma not available, running simple random search fallback")
        budget = args.budget
        eval_count = 0
        x_best = x0.copy()
        f_best = obj_fn(x_best)
        rng = np.random.default_rng(42)
        while eval_count < budget:
            cand = rng.uniform(bounds[0], bounds[1])
            f = obj_fn(cand)
            eval_count += 1
            if f < f_best:
                f_best = f
                x_best = cand

    duration = time.time() - start

    # Save results
    result = {
        'x_best': x_best.tolist(),
        'f_best': float(f_best),
        'duration_s': duration
    }
    with open(out / 'opt_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Optimization complete. Best L/D = {-result['f_best']:.3f}")
    print(f"Saved results to {out / 'opt_result.json'}")


if __name__ == '__main__':
    main()
