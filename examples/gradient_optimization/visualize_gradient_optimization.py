"""
VISUALIZE GRADIENT COMPUTATION - Show How Optimizer Finds Direction
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

## SUBPLOT 1: Airfoil with Gradient Vectors
## =========================================
ax1 = fig.add_subplot(gs[0, :])

# Baseline airfoil (simplified)
x = np.linspace(0, 1, 50)
y_baseline = 0.03 * np.sin(np.pi * x) * (1 - x)**0.5  # Simple camber

# Compute "fake" gradients (for visualization)
# In reality, these come from finite differences with NeuralFoil
gradients = np.zeros_like(y_baseline)
gradients[5:20] = 0.02 * (1 - np.abs(np.linspace(-1, 1, 15)))  # Increase camber forward
gradients[30:45] = -0.01 * (1 - np.abs(np.linspace(-1, 1, 15)))  # Decrease aft

# Plot baseline
ax1.plot(x, y_baseline, 'b-', linewidth=2, label='Baseline Airfoil', zorder=3)
ax1.plot(x, -y_baseline * 0.3, 'b-', linewidth=2, zorder=3)

# Plot gradient vectors
for i in range(0, len(x), 3):
    if abs(gradients[i]) > 0.001:
        arrow = FancyArrowPatch(
            (x[i], y_baseline[i]),
            (x[i], y_baseline[i] + gradients[i]),
            arrowstyle='->', mutation_scale=20, 
            linewidth=2, color='red', zorder=4
        )
        ax1.add_patch(arrow)

# Labels
ax1.set_xlabel('x/c (Chord Position)', fontsize=12, fontweight='bold')
ax1.set_ylabel('y/c (Thickness)', fontsize=12, fontweight='bold')
ax1.set_title('Gradient Vectors: ∂(L/D)/∂yᵢ at Each Point', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.02, 0.08)
ax1.axhline(0, color='k', linewidth=0.5)

# Add annotations
ax1.annotate('Positive gradient\n→ Move UP\n→ Increase CL', 
             xy=(0.15, 0.025), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax1.annotate('Negative gradient\n→ Move DOWN\n→ Reduce drag', 
             xy=(0.75, 0.02), fontsize=10, color='red',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))


## SUBPLOT 2: Finite Difference Computation
## =========================================
ax2 = fig.add_subplot(gs[1, 0])

# Show perturbed airfoil
x_zoom = np.linspace(0.2, 0.4, 20)
y_zoom = 0.03 * np.sin(np.pi * x_zoom) * (1 - x_zoom)**0.5

ax2.plot(x_zoom, y_zoom, 'b-', linewidth=3, label='Baseline', zorder=2)

# Perturbed point
i_perturb = 10
epsilon = 0.002
y_perturbed = y_zoom.copy()
y_perturbed[i_perturb] += epsilon

ax2.plot(x_zoom, y_perturbed, 'r--', linewidth=2, label='Perturbed', zorder=3)
ax2.plot(x_zoom[i_perturb], y_zoom[i_perturb], 'bo', markersize=12, zorder=4)
ax2.plot(x_zoom[i_perturb], y_perturbed[i_perturb], 'ro', markersize=12, zorder=4)

# Show epsilon
ax2.annotate('', xy=(x_zoom[i_perturb]+0.01, y_perturbed[i_perturb]), 
             xytext=(x_zoom[i_perturb]+0.01, y_zoom[i_perturb]),
             arrowprops=dict(arrowstyle='<->', color='green', linewidth=2))
ax2.text(x_zoom[i_perturb]+0.015, (y_zoom[i_perturb] + y_perturbed[i_perturb])/2, 
         'ε = 10⁻⁸', fontsize=11, color='green', fontweight='bold')

ax2.set_xlabel('x/c', fontsize=11, fontweight='bold')
ax2.set_ylabel('y/c', fontsize=11, fontweight='bold')
ax2.set_title('Finite Difference: Perturb One Point', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add formula
formula_text = r'$\frac{\partial (L/D)}{\partial y_i} \approx \frac{L/D(y_i + \epsilon) - L/D(y_i)}{\epsilon}$'
ax2.text(0.5, 0.05, formula_text, transform=ax2.transAxes, 
         fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         ha='center')


## SUBPLOT 3: L/D Response to Perturbation
## ========================================
ax3 = fig.add_subplot(gs[1, 1])

# Simulated L/D vs perturbation
delta_y = np.linspace(-0.005, 0.005, 100)
LD_baseline = 100
# Quadratic approximation around optimum
LD_response = LD_baseline + 150 * delta_y - 5000 * delta_y**2

ax3.plot(delta_y * 1000, LD_response, 'b-', linewidth=3)
ax3.axvline(0, color='r', linestyle='--', linewidth=2, label='Current position')
ax3.axhline(LD_baseline, color='g', linestyle='--', linewidth=1, alpha=0.5)

# Show gradient (tangent line)
gradient = 150  # At delta_y = 0
tangent = LD_baseline + gradient * delta_y
ax3.plot(delta_y * 1000, tangent, 'r--', linewidth=2, alpha=0.7, label='Gradient (tangent)')

# Annotations
ax3.plot(0, LD_baseline, 'ro', markersize=10, zorder=5)
ax3.annotate(f'Gradient = {gradient:.0f}\n(positive → move UP)', 
             xy=(0, LD_baseline), xytext=(1.5, 105),
             arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax3.set_xlabel('Δy (mm)', fontsize=11, fontweight='bold')
ax3.set_ylabel('L/D Ratio', fontsize=11, fontweight='bold')
ax3.set_title('L/D Response to Point Movement', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)


## SUBPLOT 4: Optimization Trajectory
## ===================================
ax4 = fig.add_subplot(gs[2, 0])

# Simulated optimization path
iterations = np.arange(0, 51)
LD_history = 92.4 + 95.3 * (1 - np.exp(-iterations / 15))  # Exponential approach

ax4.plot(iterations, LD_history, 'b-', linewidth=3, marker='o', 
         markersize=5, markevery=5)
ax4.axhline(187.7, color='r', linestyle='--', linewidth=2, 
            label='Optimum (L/D = 187.7)', alpha=0.7)
ax4.axhline(92.4, color='g', linestyle='--', linewidth=2, 
            label='Baseline (L/D = 92.4)', alpha=0.7)

# Fill improvement
ax4.fill_between(iterations, 92.4, LD_history, alpha=0.3, color='green', 
                 label='Improvement: +103%')

ax4.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax4.set_ylabel('L/D Ratio', fontsize=11, fontweight='bold')
ax4.set_title('L-BFGS-B Convergence History', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, 50)


## SUBPLOT 5: Gradient Magnitude Over Time
## ========================================
ax5 = fig.add_subplot(gs[2, 1])

# Simulated gradient norm
gradient_norm = 500 * np.exp(-iterations / 12)  # Decreases as we approach optimum

ax5.semilogy(iterations, gradient_norm, 'r-', linewidth=3, marker='s', 
             markersize=5, markevery=5)
ax5.axhline(1e-5, color='g', linestyle='--', linewidth=2, 
            label='Convergence Threshold', alpha=0.7)

# Add phases
ax5.axvspan(0, 15, alpha=0.2, color='red', label='Initial (large gradients)')
ax5.axvspan(15, 35, alpha=0.2, color='yellow', label='Middle (decreasing)')
ax5.axvspan(35, 50, alpha=0.2, color='green', label='Final (near optimum)')

ax5.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax5.set_ylabel('||∇(L/D)|| (Gradient Magnitude)', fontsize=11, fontweight='bold')
ax5.set_title('Gradient Decay Toward Optimum', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, which='both')
ax5.set_xlim(0, 50)

# Overall title
fig.suptitle('GRADIENT-BASED OPTIMIZATION: How L-BFGS-B Finds Optimal Airfoil', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('f:/MDO LAB/Gradient_Optimization_Visualization.png', 
            dpi=150, bbox_inches='tight')
print("✅ Visualization saved: Gradient_Optimization_Visualization.png")
plt.show()
