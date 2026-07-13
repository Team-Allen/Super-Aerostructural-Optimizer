"""
Visualization utilities for the MDO pipeline.
"""

import numpy as np
from typing import Optional
from pathlib import Path


def plot_optimization_results(result, baseline: dict, output_dir: str = "results"):
    """Generate all optimization result plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PIML Aerostructural MDO Results", fontsize=14, fontweight='bold')

    # 1. Convergence history
    if result.history:
        evals = [h['eval'] for h in result.history]
        objs = [h['objective'] for h in result.history]
        axes[0, 0].semilogy(evals, objs, 'b-', linewidth=1)
        axes[0, 0].set_xlabel('Function Evaluations')
        axes[0, 0].set_ylabel('Objective')
        axes[0, 0].set_title('Convergence History')
        axes[0, 0].grid(True, alpha=0.3)

    # 2. CL/CD history
    if result.history:
        cls = [h['cl'] for h in result.history]
        cds = [h['cd'] for h in result.history]
        ax2 = axes[0, 1]
        ax2.plot(evals, cls, 'b-', label='CL')
        ax2.set_xlabel('Evaluations')
        ax2.set_ylabel('CL', color='b')
        ax2_r = ax2.twinx()
        ax2_r.plot(evals, cds, 'r-', label='CD')
        ax2_r.set_ylabel('CD', color='r')
        ax2.set_title('Aero Coefficients')
        ax2.grid(True, alpha=0.3)

    # 3. L/D history
    if result.history:
        lds = [h['ld'] for h in result.history]
        axes[0, 2].plot(evals, lds, 'g-', linewidth=1)
        axes[0, 2].axhline(y=baseline.get('ld', 0), color='k', linestyle='--',
                           alpha=0.5, label='Baseline')
        axes[0, 2].set_xlabel('Evaluations')
        axes[0, 2].set_ylabel('L/D')
        axes[0, 2].set_title('Lift-to-Drag Ratio')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. Optimized airfoil shape
    from ..aero.airfoil_geometry import AirfoilGeometry
    d = result.design_dict
    geom = AirfoilGeometry(
        upper_weights=d['cst_upper'],
        lower_weights=d['cst_lower'],
    )
    x, y_u, y_l = geom.coordinates(150)
    axes[1, 0].fill_between(x, y_u, y_l, alpha=0.3, color='blue')
    axes[1, 0].plot(x, y_u, 'b-', linewidth=1.5)
    axes[1, 0].plot(x, y_l, 'b-', linewidth=1.5)
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.15, 0.15)
    axes[1, 0].set_aspect('equal')
    axes[1, 0].set_xlabel('x/c')
    axes[1, 0].set_ylabel('y/c')
    axes[1, 0].set_title('Optimized Airfoil')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Structural mass history
    if result.history:
        masses = [h['mass'] for h in result.history]
        axes[1, 1].plot(evals, masses, 'm-', linewidth=1)
        axes[1, 1].axhline(y=baseline.get('mass', 0), color='k', linestyle='--',
                           alpha=0.5, label='Baseline')
        axes[1, 1].set_xlabel('Evaluations')
        axes[1, 1].set_ylabel('Mass [kg]')
        axes[1, 1].set_title('Structural Mass')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Failure index history
    if result.history:
        fis = [h['failure_index'] for h in result.history]
        axes[1, 2].plot(evals, fis, 'r-', linewidth=1)
        axes[1, 2].axhline(y=0.8, color='k', linestyle='--', alpha=0.5, label='Limit')
        axes[1, 2].set_xlabel('Evaluations')
        axes[1, 2].set_ylabel('Failure Index')
        axes[1, 2].set_title('Structural Failure Index')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "mdo_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plots saved to {output_path / 'mdo_results.png'}")
