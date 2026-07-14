"""
Render the FINAL optimized wing (-2deg washout, real CAD-scaled, blunt-TE
geometry): Cp/Mach from the real converged SU2 solution, plus the real
150-iteration structural convergence (mass, FI, per-zone thickness/plies).
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Cp/Mach render, from the real -2deg SU2 surface solution ---
SURFACE_CSV = os.path.join(HERE, "repro_check", "run1", "surface_flow.csv")
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
axes[0].set_xlabel("x (chordwise) [m]"); axes[0].set_ylabel("y (spanwise) [m]")
axes[0].set_title("Final optimized wing (-2deg washout): surface Cp\nSU2 Euler, Mach 0.85, alpha=3deg")
axes[0].set_aspect("equal")
plt.colorbar(sc0, ax=axes[0], label="Cp")

sc1 = axes[1].scatter(x, y, c=mach_local, cmap="viridis", s=3, vmin=0, vmax=np.percentile(mach_local, 99))
axes[1].set_xlabel("x (chordwise) [m]"); axes[1].set_ylabel("y (spanwise) [m]")
axes[1].set_title(f"Final optimized wing: local Mach number (peak {mach_local.max():.3f})\n"
                   f"vs 0deg baseline peak Mach 1.186 -- washout reduces shock strength")
axes[1].set_aspect("equal")
plt.colorbar(sc1, ax=axes[1], label="Mach")
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "16_optimized_wing_cp_mach.png")
plt.savefig(out1, dpi=150)
print(f"Wrote {out1}, Cp range [{cp.min():.3f}, {cp.max():.3f}], Mach max {mach_local.max():.3f}")
plt.close()

# --- Structural convergence, from the real 150-iteration optimized run ---
with open(os.path.join(HERE, "production_run_optimized", "iteration_log.csv")) as f:
    rows = list(csv.DictReader(f))

it = [int(r["iter"]) for r in rows]
mass = [float(r["mass_kg"]) for r in rows]
fi = [float(r["global_max_fi"]) for r in rows]
n_zones = 8
thickness = {z: [float(r[f"zone{z}_thickness_scale"]) for r in rows] for z in range(n_zones)}
plies = {z: [int(float(r[f"zone{z}_total_plies"])) for r in rows] for z in range(n_zones)}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(it, mass, color="tab:blue")
axes[0, 0].set_xlabel("Iteration"); axes[0, 0].set_ylabel("Mass [kg]")
axes[0, 0].set_title("Final optimized wing: mass convergence")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(it, fi, color="tab:red")
axes[0, 1].axhline(0.8, color="k", linestyle="--", linewidth=1, label="target FI=0.8")
axes[0, 1].set_xlabel("Iteration"); axes[0, 1].set_ylabel("Global max FI")
axes[0, 1].set_title("Failure index convergence")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

for z in range(n_zones):
    axes[1, 0].plot(it, thickness[z], label=f"zone {z}")
axes[1, 0].set_xlabel("Iteration"); axes[1, 0].set_ylabel("Thickness scale")
axes[1, 0].set_title("Per-zone thickness scale convergence")
axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].grid(alpha=0.3)

for z in range(n_zones):
    axes[1, 1].plot(it, plies[z], label=f"zone {z}")
axes[1, 1].set_xlabel("Iteration"); axes[1, 1].set_ylabel("Total plies")
axes[1, 1].set_title("Per-zone ply count convergence")
axes[1, 1].legend(fontsize=7, ncol=2); axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "17_optimized_structural_convergence.png")
plt.savefig(out2, dpi=150)
print(f"Wrote {out2}")

# --- Final per-zone thickness/ply bar chart (the actual sized composite wing) ---
last = rows[-1]
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
zones = list(range(n_zones))
axes[0].bar(zones, [float(last[f"zone{z}_thickness_scale"]) for z in zones], color="tab:purple")
axes[0].set_xlabel("Zone (root -> tip)"); axes[0].set_ylabel("Final thickness scale")
axes[0].set_title("Final optimized wing: per-zone thickness")
axes[0].grid(alpha=0.3, axis="y")

axes[1].bar(zones, [int(float(last[f"zone{z}_total_plies"])) for z in zones], color="tab:orange")
axes[1].set_xlabel("Zone (root -> tip)"); axes[1].set_ylabel("Total plies")
axes[1].set_title("Final optimized wing: per-zone ply count")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "18_optimized_final_sizing.png")
plt.savefig(out3, dpi=150)
print(f"Wrote {out3}")
