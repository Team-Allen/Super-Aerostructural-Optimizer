"""
Render the corrected-geometry SU2 solve (real CAD-scaled wing, blunt TE,
physical CD) directly from its surface CSV using matplotlib -- no ParaView
dependency, real solved Cp data from ideal-gas relation (same formula used
throughout this project's FSI transfer and prior render scripts).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

SURFACE_CSV = os.path.join(HERE, "corrected_mesh_test5", "surface_flow.csv")

GAMMA = 1.4
FREESTREAM_P = 26436.3
RHO_INF = 0.4127
V_INF = 254.546

data = np.genfromtxt(SURFACE_CSV, delimiter=",", names=True)
x, y, z = data["x"], data["y"], data["z"]
rho, mx, my, mz, e = data["Density"], data["Momentum_x"], data["Momentum_y"], data["Momentum_z"], data["Energy"]
kinetic = 0.5 * (mx**2 + my**2 + mz**2) / np.maximum(rho, 1e-9)
pressure = (GAMMA - 1.0) * (e - kinetic)
cp = (pressure - FREESTREAM_P) / (0.5 * RHO_INF * V_INF**2)
mach_local = np.sqrt(mx**2 + my**2 + mz**2) / np.maximum(rho, 1e-9) / np.sqrt(GAMMA * pressure / np.maximum(rho, 1e-9))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sc0 = axes[0].scatter(x, y, c=cp, cmap="RdBu_r", s=3, vmin=np.percentile(cp, 1), vmax=np.percentile(cp, 99))
axes[0].set_xlabel("x (chordwise) [m]")
axes[0].set_ylabel("y (spanwise) [m]")
axes[0].set_title("Corrected wing (semi-span 3.0m): surface Cp, top-down\nSU2 Euler, Mach 0.85, alpha=3deg, real CAD-scaled geometry")
axes[0].set_aspect("equal")
plt.colorbar(sc0, ax=axes[0], label="Cp")

sc1 = axes[1].scatter(x, y, c=mach_local, cmap="viridis", s=3, vmin=0, vmax=np.percentile(mach_local, 99))
axes[1].set_xlabel("x (chordwise) [m]")
axes[1].set_ylabel("y (spanwise) [m]")
axes[1].set_title("Corrected wing: local Mach number on surface\n(real, from converged restart state, CD verified positive)")
axes[1].set_aspect("equal")
plt.colorbar(sc1, ax=axes[1], label="Mach")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "13_corrected_geometry_cp_mach.png")
plt.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
print(f"Cp range: [{cp.min():.3f}, {cp.max():.3f}]")
print(f"n points: {len(x)}")
