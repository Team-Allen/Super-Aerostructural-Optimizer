#!/usr/bin/env python3
"""
Run the full PIML Aerostructural MDO Pipeline.

Usage:
    python scripts/run_pipeline.py                          # Default config
    python scripts/run_pipeline.py --config config/my.yaml  # Custom config
    python scripts/run_pipeline.py --quick                  # Quick test (10 iters)
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from piml_mdo.pipeline.orchestrator import PipelineOrchestrator, PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="PIML Aerostructural MDO Pipeline")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML or JSON config file")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test run (10 iterations)")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--solver",
                       choices=["neuralfoil", "pinn", "openaerostruct", "surrogate_cfd"],
                       default=None, help="Aero solver (overrides config if given)")
    parser.add_argument("--structural-solver",
                       choices=["clt", "vam", "surrogate"],
                       default=None, help="Structural solver (overrides config if given)")
    parser.add_argument("--method", choices=["L-BFGS-B", "SLSQP", "COBYLA", "differential_evolution"],
                       default=None, help="Optimization method (overrides config)")
    parser.add_argument("--max-iter", type=int, default=None,
                       help="Max optimization iterations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_config(args) -> PipelineConfig:
    """Load config from file or create from CLI args."""
    if args.config:
        path = Path(args.config)
        if path.suffix == '.json':
            config = PipelineConfig.from_json(str(path))
        elif path.suffix in ('.yaml', '.yml'):
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f)
            # Flatten nested YAML into flat config
            flat = {}
            if 'project' in raw:
                flat.update(raw['project'])
            if 'aero' in raw:
                flat['aero_solver'] = raw['aero'].get('solver', 'neuralfoil')
                flat['neuralfoil_model_size'] = raw['aero'].get('neuralfoil_model_size', 'xlarge')
                flat['pinn_pretrained'] = raw['aero'].get('pinn_pretrained')
                flat['openaerostruct_with_viscous'] = raw['aero'].get('openaerostruct_with_viscous', True)
                flat['openaerostruct_num_x'] = raw['aero'].get('openaerostruct_num_x', 5)
                flat['openaerostruct_num_y'] = raw['aero'].get('openaerostruct_num_y', 9)
                flat['surrogate_cfd_model_dir'] = raw['aero'].get('surrogate_cfd_model_dir', 'results/surrogate_cfd')
                flat['surrogate_cfd_n_samples'] = raw['aero'].get('surrogate_cfd_n_samples', 60)
                flat['surrogate_cfd_hidden_layers'] = tuple(raw['aero'].get('surrogate_cfd_hidden_layers', [128, 64, 32]))
                flat['surrogate_cfd_retrain'] = raw['aero'].get('surrogate_cfd_retrain', False)
            if 'wing' in raw:
                flat['wing_span'] = raw['wing'].get('semi_span', 15.0)
                flat['chord_root'] = raw['wing'].get('chord_root', 3.5)
                flat['chord_tip'] = raw['wing'].get('chord_tip', 1.4)
                flat['sweep_deg'] = raw['wing'].get('sweep_deg', 25.0)
                flat['initial_airfoil'] = raw['wing'].get('initial_airfoil', 'NACA2412')
            if 'structure' in raw:
                flat['n_beam_elements'] = raw['structure'].get('n_beam_elements', 20)
                flat['material'] = raw['structure'].get('material', 'CFRP_IM7_8552')
                flat['layup'] = raw['structure'].get('layup', 'quasi_isotropic')
                flat['structural_solver'] = raw['structure'].get('structural_solver', 'vam')
                flat['mystran_exe'] = raw['structure'].get('mystran_exe')
                flat['structural_surrogate_path'] = raw['structure'].get('structural_surrogate_path')
                flat['generate_structural_doe'] = raw['structure'].get('generate_structural_doe', False)
            if 'flight' in raw:
                flat['velocity'] = raw['flight'].get('velocity', 100.0)
                flat['altitude'] = raw['flight'].get('altitude', 3000.0)
                flat['cl_target'] = raw['flight'].get('cl_target', 0.5)
            if 'optimization' in raw:
                flat['optimizer_method'] = raw['optimization'].get('method', 'L-BFGS-B')
                flat['max_opt_iterations'] = raw['optimization'].get('max_iterations', 100)
                flat['n_cst_upper'] = raw['optimization'].get('n_cst_upper', 6)
                flat['n_cst_lower'] = raw['optimization'].get('n_cst_lower', 6)
                flat['n_twist_stations'] = raw['optimization'].get('n_twist_stations', 5)
                flat['n_struct_sizing'] = raw['optimization'].get('n_struct_sizing', 5)
                flat['n_laminate_stations'] = raw['optimization'].get('n_laminate_stations', 5)
                flat['n_ply_angles'] = raw['optimization'].get('n_ply_angles', 4)
                flat['optimize_layup'] = raw['optimization'].get('optimize_layup', True)
                flat['coupling_max_iterations'] = raw['optimization'].get('coupling_max_iterations', 8)
                flat['coupling_tolerance'] = raw['optimization'].get('coupling_tolerance', 1e-4)
                flat['coupling_relaxation'] = raw['optimization'].get('coupling_relaxation', 0.3)
            if 'constraints' in raw:
                for k, v in raw['constraints'].items():
                    if k in PipelineConfig.__dataclass_fields__:
                        flat[k] = v
            if 'output' in raw:
                for k, v in raw['output'].items():
                    if k in PipelineConfig.__dataclass_fields__:
                        flat[k] = v
            config = PipelineConfig(**{k: v for k, v in flat.items()
                                      if k in PipelineConfig.__dataclass_fields__})
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
    else:
        config = PipelineConfig()

    # CLI overrides
    if args.output:
        config.output_dir = args.output
    if args.solver is not None:
        config.aero_solver = args.solver
    if args.structural_solver is not None:
        config.structural_solver = args.structural_solver
    if args.method:
        config.optimizer_method = args.method
    if args.max_iter:
        config.max_opt_iterations = args.max_iter
    if args.quick:
        config.max_opt_iterations = 10

    return config


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    config = load_config(args)

    print(f"\n  PIML Aerostructural MDO Pipeline")
    print(f"  ================================")
    print(f"  Solver: {config.aero_solver}")
    print(f"  Optimizer: {config.optimizer_method}")
    print(f"  Max iterations: {config.max_opt_iterations}")
    print(f"  Wing span: {config.wing_span}m")
    print(f"  Material: {config.material}")
    print(f"  Output: {config.output_dir}")
    print()

    pipeline = PipelineOrchestrator(config)
    result = pipeline.run()

    if result.success:
        print(f"\n  Optimization SUCCEEDED")
        print(f"  Final L/D: {result.ld_ratio:.1f}")
        print(f"  Final CD: {result.cd:.6f}")
        print(f"  Structural mass: {result.structural_mass:.1f} kg")
    else:
        print(f"\n  Optimization completed with warnings: {result.message}")

    # Generate plots
    try:
        from piml_mdo.utils.plotting import plot_optimization_results
        plot_optimization_results(result, {}, str(pipeline.run_dir))
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
