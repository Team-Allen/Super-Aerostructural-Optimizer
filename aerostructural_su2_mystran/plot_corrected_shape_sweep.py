"""
Plot the corrected-geometry (real CAD-scaled, blunt-TE) washout sweep --
real SU2 solves only, no interpolation invented here (see the separate
interpolation script for that, built only after real points exist).
"""

import csv
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
LOG_CSV = os.path.join(HERE, "aero_shape_opt", "shape_sweep_log.csv")
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

with open(LOG_CSV) as f:
    rows = list(csv.DictReader(f))

rows = [r for r in rows if r["exit_ok"] == "True"]
twist = [float(r["twist_deg"]) for r in rows]
cl = [float(r["cl"]) for r in rows]
cd = [float(r["cd"]) for r in rows]
# L/D is only meaningful (and only reported by the sweep script) where CD > 0;
# at the two lowest-CL candidates here, CD is a tiny residual negative value
# (near-zero-lift regime, not the original sharp-TE bug -- see README), so
# L/D is left out of this plot for those points rather than fabricated.
ld_pts = [(t, float(r["ld"])) for t, r in zip(twist, rows) if r["ld"] not in ("", None)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(twist, cl, "o-", color="tab:blue")
axes[0].set_xlabel("Tip washout [deg]")
axes[0].set_ylabel("CL")
axes[0].set_title("Lift coefficient vs washout\n(corrected geometry, real CAD-scaled)")
axes[0].grid(alpha=0.3)

axes[1].plot(twist, cd, "o-", color="tab:red")
axes[1].set_xlabel("Tip washout [deg]")
axes[1].set_ylabel("CD")
axes[1].set_title("Drag coefficient vs washout\n(Euler-only, no viscous floor)")
axes[1].grid(alpha=0.3)

if ld_pts:
    lt, lv = zip(*ld_pts)
    axes[2].plot(lt, lv, "o-", color="tab:green")
axes[2].set_xlabel("Tip washout [deg]")
axes[2].set_ylabel("L/D")
axes[2].set_title("L/D vs washout (only where CD>0)\n(real SU2 solves, corrected scale)")
axes[2].set_xlim(min(twist) - 0.5, max(twist) + 0.5)
axes[2].grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "14_aero_shape_sweep_corrected.png")
plt.savefig(out_path, dpi=150)
print(f"Wrote {out_path}")
for r in rows:
    print(f"  twist={r['twist_deg']:>6} CL={r['cl']:>8} CD={r['cd']:>10} L/D={r['ld']:>8}")
