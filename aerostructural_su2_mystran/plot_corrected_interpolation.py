"""
Real result interpolation on the corrected-geometry washout sweep. Only CL
is interpolated -- it is positive and well-behaved at all 4 real points.
CD/L-D are deliberately NOT interpolated here: 2 of the 4 real points have
unreliable (small negative residual) CD near zero lift, and fitting a smooth
curve through that would manufacture a misleadingly clean CD/L-D trend from
data that isn't there. See README Sec6 for the honest characterization.
"""

import csv
import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
LOG_CSV = os.path.join(HERE, "aero_shape_opt", "shape_sweep_log.csv")
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

with open(LOG_CSV) as f:
    rows = [r for r in csv.DictReader(f) if r["exit_ok"] == "True"]

twist = np.array([float(r["twist_deg"]) for r in rows])
cl = np.array([float(r["cl"]) for r in rows])
order = np.argsort(twist)
twist, cl = twist[order], cl[order]

spline = CubicSpline(twist, cl)
t_fine = np.linspace(twist.min(), twist.max(), 200)
cl_fine = spline(t_fine)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(t_fine, cl_fine, "-", color="tab:blue", label="Cubic spline interpolation")
ax.plot(twist, cl, "o", color="tab:blue", markersize=10, label="Real SU2 solves (n=4)")
ax.set_xlabel("Tip washout [deg]")
ax.set_ylabel("CL")
ax.set_title("CL vs washout -- corrected geometry, real CAD-scaled wing\n"
             "(CD/L-D not interpolated: 2 of 4 points have unreliable near-zero-lift CD -- see README Sec6)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "15_cl_interpolation_corrected.png")
plt.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
