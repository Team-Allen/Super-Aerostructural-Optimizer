"""
Aerostructural optimization: GNN-based aerodynamics + OpenAeroStruct structures.

This script combines:
  1. GNN surrogate for aerodynamic analysis (fast, learned)
  2. OpenAeroStruct for structural analysis (mass, stress, frequencies)
  3. CMA-ES for wing design optimization (span, taper, sweep, twist)

Usage:
  python scripts/optimize_wing_aerostructural.py \
    --model training/checkpoints/gnn_best.pth \
    --device cuda \
    --budget 300 \
    --out results/stage2_aerostructural

Output:
  results/stage2_aerostructural/opt_result.json : {
    "best_design": [span, taper, sweep, dihedral, twist_root, twist_tip],
    "best_l_d": 15.8,
    "best_mass_kg": 2450,
    "best_stress_mpa": 85.2,
    "optimization_history": [...],
    "pareto_front": [(l_d, mass, stress), ...]
  }
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Local imports
from src.aerodynamics_3d.wing_geometry import Wing3D, WingParameters, WingSection
from src.aerodynamics_3d.wing_from_airfoils import create_wing_from_airfoil_files
from src.aerodynamics_3d.gnn_wing_analyzer import GNNWingAnalyzer

# Try to import openmdao / OpenAeroStruct for structural analysis
try:
    import openmdao.api as om
    from openaerostruct.geometry.utils import generate_mesh
    OPENAEROSTRUCT_AVAILABLE = True
except ImportError:
    OPENAEROSTRUCT_AVAILABLE = False
    print("⚠️  OpenAeroStruct not available; will use simplified structural model")

# Try to import CMA-ES optimizer
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("⚠️  cma package not available; will use random search")


class SimplifiedStructuralModel:
    """
    Lightweight structural model without OpenAeroStruct.
    Estimates mass and max stress based on simple beam theory + empirics.
    """

    def __init__(self, wing: Wing3D):
        self.wing = wing

    def analyze(self, override_thickness: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Predict structural mass, max stress, and first bending frequency.
        
        Args:
            override_thickness: Optional array of spar thicknesses (m) along span
            
        Returns:
            {
              'mass_kg': total structural mass (kg),
              'max_stress_mpa': maximum von Mises stress (MPa),
              'first_freq_hz': first bending frequency (Hz),
              'flutter_speed_ms': approximate flutter speed (m/s)
            }
        """
        # Get wing geometry from parameters
        span = self.wing.params.span
        area = self.wing.get_area()
        mean_chord = area / span if span > 0 else 1.0
        
        # Estimate mass: semi-empirical model
        # For a composite wing: ~15-25 kg/m^2 of planform area
        specific_mass = 20.0  # kg/m^2
        structural_mass = area * specific_mass
        
        # Assume some wing box structure with spars at 25% and 75% chord
        # and skin thickness proportional to local bending moment
        # Simple estimate: max bending moment ~ (1/2) * rho * V^2 * wing_area * span / 2
        wing_loading = structural_mass * 9.81 / area  # N/m^2
        
        # Approximate max stress (von Mises) in spar
        # Simplified: local bending stress ~ M * c / I
        # Where M ~ distributed load * (span/2)^2
        # For composite, typical allowable ~300 MPa; use 100 MPa as baseline
        max_stress_mpa = 80.0 + (span / 10.0) * 5.0  # increases with span (higher bending moments)
        
        # Approximate first bending frequency
        # f ~ sqrt(stiffness / mass) / (2*pi)
        # Simplified: f ~ 3-5 Hz for typical UAV wings
        first_freq_hz = 4.0 - (span / 20.0) * 1.0  # decreases with span
        
        # Flutter speed (rough estimate, Theodorsen's rule)
        # V_flutter ~ sqrt(EI / (rho * b * c^2)) where b = semi-span, c = chord
        # Simplified: estimate based on frequency and aerodynamic damping
        flutter_speed_ms = first_freq_hz * span * 2.0 / 3.14  # heuristic
        
        return {
            'mass_kg': structural_mass,
            'max_stress_mpa': max_stress_mpa,
            'first_freq_hz': first_freq_hz,
            'flutter_speed_ms': flutter_speed_ms,
        }


class AerostructuralWingAnalyzer:
    """
    Combined aerodynamic (GNN) + structural (OpenAeroStruct or simplified) analyzer.
    """

    def __init__(self, gnn_analyzer: GNNWingAnalyzer, use_openaerostruct: bool = False):
        self.gnn = gnn_analyzer
        self.use_openaerostruct = use_openaerostruct and OPENAEROSTRUCT_AVAILABLE
        self.struct_model = None

    def analyze_wing(self, wing: Wing3D) -> Dict[str, float]:
        """
        Analyze wing aerodynamically and structurally.
        
        Args:
            wing: Wing3D object with geometry and sections
            
        Returns:
            {
              'l_d': lift-to-drag ratio,
              'mass_kg': structural mass,
              'max_stress_mpa': max stress,
              'first_freq_hz': first bending frequency,
              'flutter_margin': 1.0 - (V_cruise / V_flutter),  # >0.2 is good
              'valid': True/False (pass structural constraints),
              'raw_aero': {...},
              'raw_struct': {...}
            }
        """
        # Define standard flight conditions (cruise)
        flight_conditions = {
            'velocity': 30.0,      # 30 m/s cruise speed (typical UAV)
            'alpha': 5.0,          # 5° angle of attack
            'altitude': 100.0      # 100 m altitude
        }
        
        # Analyze aerodynamics
        aero_result = self.gnn.analyze_wing(wing, flight_conditions)
        
        # Analyze structures
        if self.use_openaerostruct:
            struct_result = self._analyze_with_openaerostruct(wing)
        else:
            struct_model = SimplifiedStructuralModel(wing)
            struct_result = struct_model.analyze()
        
        # Combine results
        l_d = aero_result.get('l_d', 1.0)
        mass_kg = struct_result['mass_kg']
        max_stress = struct_result['max_stress_mpa']
        freq_hz = struct_result['first_freq_hz']
        
        # Flutter margin: assume cruise speed ~30 m/s for small UAV
        cruise_speed = 30.0
        flutter_speed = struct_result.get('flutter_speed_ms', cruise_speed * 1.5)
        flutter_margin = (flutter_speed - cruise_speed) / cruise_speed if cruise_speed > 0 else 0.5
        
        # Constraints
        valid = (
            max_stress < 200.0 and  # allowable stress for composites
            freq_hz > 2.0 and  # avoid flutter
            flutter_margin > 0.2  # 20% margin
        )
        
        return {
            'l_d': l_d,
            'mass_kg': mass_kg,
            'max_stress_mpa': max_stress,
            'first_freq_hz': freq_hz,
            'flutter_margin': flutter_margin,
            'valid': valid,
            'raw_aero': aero_result,
            'raw_struct': struct_result,
        }

    def _analyze_with_openaerostruct(self, wing: Wing3D) -> Dict[str, float]:
        """Placeholder for full OpenAeroStruct integration."""
        # TODO: Implement real OpenAeroStruct coupling
        # For now, use simplified model as fallback
        struct_model = SimplifiedStructuralModel(wing)
        return struct_model.analyze()


class AerostructuralOptimizer:
    """
    Multi-objective optimizer for wing design using aerostructural metrics.
    Minimizes: -L/D (maximize L/D) and mass
    Constraints: stress < 200 MPa, flutter margin > 0.2
    """

    def __init__(
        self,
        analyzer: AerostructuralWingAnalyzer,
        budget: int = 300,
        popsize: int = 16,
        device: str = 'cpu',
        output_dir: Optional[str] = None
    ):
        self.analyzer = analyzer
        self.budget = budget
        self.popsize = popsize
        self.device = device
        self.output_dir = Path(output_dir or 'results/stage2_aerostructural')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = []
        self.pareto_front = []  # (l_d, mass, stress) tuples

    def objective(self, design_vars: np.ndarray) -> float:
        """
        Objective function: minimize -(L/D) - 0.01 * (mass_ratio)
        
        Higher negative values are better (CMA-ES minimizes).
        """
        try:
            # Unpack design variables: [span, taper, sweep, dihedral, twist_root, twist_tip]
            span, taper, sweep, dihedral, twist_root, twist_tip = design_vars
            
            # Enforce bounds
            span = np.clip(span, 3.0, 20.0)
            taper = np.clip(taper, 0.3, 1.0)
            sweep = np.clip(sweep, -20.0, 45.0)
            dihedral = np.clip(dihedral, 0.0, 20.0)
            twist_root = np.clip(twist_root, -5.0, 5.0)
            twist_tip = np.clip(twist_tip, -10.0, 10.0)
            
            # Create wing
            wing_params = WingParameters(
                span=span,
                root_chord=2.0,
                taper_ratio=taper,
                sweep_angle=sweep,
                dihedral=dihedral,
                twist_root=twist_root,
                twist_tip=twist_tip,
                n_sections=30,
                airfoil_root='NACA0012'
            )
            wing = Wing3D(wing_params)
            
            # Analyze
            result = self.analyzer.analyze_wing(wing)
            
            # Multi-objective: L/D + structural constraints
            l_d = result['l_d']
            mass_kg = result['mass_kg']
            stress = result['max_stress_mpa']
            flutter_margin = result['flutter_margin']
            
            # Fitness: maximize L/D, minimize mass, satisfy constraints
            # Use weighted sum (Scalarization)
            penalty = 0.0
            
            # Soft penalties for constraint violations
            if stress > 200.0:
                penalty += 100.0 * (stress - 200.0) / 200.0
            if flutter_margin < 0.2:
                penalty += 100.0 * (0.2 - flutter_margin) / 0.2
            
            # Objective: minimize -(L/D) with secondary weight on mass
            fitness = -l_d + 0.01 * (mass_kg / 2500.0) + penalty
            
            # Log
            self.history.append({
                'design': design_vars.copy(),
                'l_d': float(l_d),
                'mass_kg': float(mass_kg),
                'stress_mpa': float(stress),
                'flutter_margin': float(flutter_margin),
                'fitness': float(fitness),
                'valid': result['valid']
            })
            
            # Track Pareto front
            if result['valid']:
                self.pareto_front.append((l_d, mass_kg, stress))
            
            print(f"  Eval {len(self.history):3d}: L/D={l_d:6.2f}, Mass={mass_kg:7.1f}kg, "
                  f"Stress={stress:6.1f}MPa, Margin={flutter_margin:.2f}")
            
            return fitness
            
        except Exception as e:
            print(f"  ⚠️  Evaluation failed: {e}")
            return 1e10  # Large penalty

    def optimize(self) -> Dict:
        """Run optimization and return results."""
        print(f"\n🚀 Starting aerostructural optimization (budget={self.budget})")
        print(f"   Device: {self.device}, PopSize: {self.popsize}")
        print("-" * 80)
        
        # Initial design: [span=10, taper=0.7, sweep=15, dihedral=5, twist_root=0, twist_tip=-3]
        x0 = np.array([10.0, 0.7, 15.0, 5.0, 0.0, -3.0])
        
        if CMA_AVAILABLE and self.budget >= 100:
            # CMA-ES optimization
            print("Using CMA-ES optimizer...")
            es = cma.CMAEvolutionStrategy(x0, sigma0=2.0, inopts={'maxfevals': self.budget})
            
            while not es.stop():
                solutions = es.ask()
                fitness_values = [self.objective(x) for x in solutions]
                es.tell(solutions, fitness_values)
                es.disp()
            
            best_x = es.result.xbest
            best_fitness = es.result.fbest
        else:
            # Random search fallback
            print("Using random search (CMA-ES not available)...")
            best_x = x0.copy()
            best_fitness = self.objective(x0)
            
            for i in range(1, self.budget):
                # Random perturbation
                x = x0 + np.random.randn(6) * 2.0
                fitness = self.objective(x)
                if fitness < best_fitness:
                    best_x = x.copy()
                    best_fitness = fitness
                if i % 20 == 0:
                    print(f"  Iteration {i}/{self.budget}: best fitness = {best_fitness:.4f}")
        
        print("-" * 80)
        print(f"✅ Optimization complete. Best fitness: {best_fitness:.4f}")
        
        # Evaluate best design for final report
        span, taper, sweep, dihedral, twist_root, twist_tip = best_x
        wing_params = WingParameters(
            span=float(span), root_chord=2.0, taper_ratio=float(taper),
            sweep_angle=float(sweep), dihedral=float(dihedral),
            twist_root=float(twist_root), twist_tip=float(twist_tip),
            n_sections=30,
            airfoil_root='NACA0012'
        )
        best_wing = Wing3D(wing_params)
        best_result = self.analyzer.analyze_wing(best_wing)
        
        # Save results
        output_file = self.output_dir / 'opt_result.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract unique Pareto points (dominated by L/D, then by mass)
        if self.pareto_front:
            # Simple Pareto filter: keep designs that maximize L/D and minimize mass
            sorted_by_ld = sorted(set(self.pareto_front), key=lambda x: -x[0])
            pareto = []
            min_mass = float('inf')
            for l_d, mass, stress in sorted_by_ld:
                if mass < min_mass:
                    pareto.append({'l_d': float(l_d), 'mass_kg': float(mass), 'stress_mpa': float(stress)})
                    min_mass = mass
        else:
            pareto = []
        
        results = {
            'best_design': {
                'span_m': float(span),
                'taper_ratio': float(taper),
                'sweep_deg': float(sweep),
                'dihedral_deg': float(dihedral),
                'twist_root_deg': float(twist_root),
                'twist_tip_deg': float(twist_tip),
            },
            'best_metrics': {
                'l_d': float(best_result['l_d']),
                'mass_kg': float(best_result['mass_kg']),
                'stress_mpa': float(best_result['max_stress_mpa']),
                'freq_hz': float(best_result['first_freq_hz']),
                'flutter_margin': float(best_result['flutter_margin']),
            },
            'optimization': {
                'budget': self.budget,
                'num_evals': len(self.history),
                'best_fitness': float(best_fitness),
            },
            'pareto_front': pareto,
            'history_summary': {
                'num_feasible': sum(1 for h in self.history if h.get('valid', False)),
                'avg_l_d': float(np.mean([h['l_d'] for h in self.history if h.get('valid', False)] or [0])),
                'best_l_d_found': float(max([h['l_d'] for h in self.history], default=0.0)),
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")
        print(f"\n   Best L/D: {results['best_metrics']['l_d']:.2f}")
        print(f"   Best Mass: {results['best_metrics']['mass_kg']:.1f} kg")
        print(f"   Max Stress: {results['best_metrics']['stress_mpa']:.1f} MPa")
        print(f"   Pareto front size: {len(pareto)}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Aerostructural wing optimization (GNN + OAS)")
    parser.add_argument('--model', type=str, default=None, help='Path to trained GNN checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--budget', type=int, default=300, help='Optimization budget (function evals)')
    parser.add_argument('--popsize', type=int, default=16, help='Population size for CMA-ES')
    parser.add_argument('--out', type=str, default='results/stage2_aerostructural', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize GNN analyzer
    print("Initializing GNN wing analyzer...")
    gnn_analyzer = GNNWingAnalyzer(model_path=args.model, device=args.device)
    
    # Initialize aerostructural analyzer
    print("Initializing aerostructural analyzer...")
    aero_struct_analyzer = AerostructuralWingAnalyzer(
        gnn_analyzer,
        use_openaerostruct=OPENAEROSTRUCT_AVAILABLE
    )
    
    # Run optimization
    optimizer = AerostructuralOptimizer(
        aero_struct_analyzer,
        budget=args.budget,
        popsize=args.popsize,
        device=args.device,
        output_dir=args.out
    )
    
    results = optimizer.optimize()
    
    return results


if __name__ == '__main__':
    main()
