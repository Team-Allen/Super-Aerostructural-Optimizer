"""
Visualize the real 200-point LHS VLM DOE screening: design-space coverage
and the resulting L/D landscape.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(HERE, "doe_vlm", "doe_vlm_log.csv")) as f:
    rows = list(csv.DictReader(f))

t_mid = np.array([float(r["twist_mid_deg"]) for r in rows])
t_tip = np.array([float(r["twist_tip_deg"]) for r in rows])
alpha = np.array([float(r["alpha_deg"]) for r in rows])
feasible = np.array([r["feasible"] == "True" for r in rows])
ld = np.array([float(r["LD"]) if r["LD"] not in ("", "None") else np.nan for r in rows])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

sc0 = axes[0].scatter(t_mid[feasible], t_tip[feasible], c=ld[feasible], cmap="viridis", s=25)
axes[0].scatter(t_mid[~feasible], t_tip[~feasible], c="lightgray", s=15, marker="x", label="infeasible")
axes[0].set_xlabel("Twist at mid-span [deg]")
axes[0].set_ylabel("Twist at tip [deg]")
axes[0].set_title(f"200-point LHS DOE: design space coverage\n({feasible.sum()}/{len(rows)} feasible, CL>=0.08)")
plt.colorbar(sc0, ax=axes[0], label="VLM L/D")
axes[0].legend(fontsize=8)

sc1 = axes[1].scatter(alpha[feasible], ld[feasible], c=t_mid[feasible], cmap="coolwarm", s=25)
axes[1].set_xlabel("Angle of attack [deg]")
axes[1].set_ylabel("VLM L/D")
axes[1].set_title("L/D vs trim alpha\n(colored by mid-span twist)")
plt.colorbar(sc1, ax=axes[1], label="twist_mid [deg]")

cl = np.array([float(r["CL"]) if r["CL"] not in ("", "None") else np.nan for r in rows])
axes[2].scatter(cl[feasible], ld[feasible], c="tab:blue", s=20, alpha=0.7)
axes[2].set_xlabel("CL")
axes[2].set_ylabel("VLM L/D")
axes[2].set_title("L/D vs CL, all 200 real VLM solves\n(top cluster: high CL, high twist+alpha)")

plt.tight_layout()
out = os.path.join(OUT_DIR, "21_doe_vlm_screening.png")
plt.savefig(out, dpi=150)
print(f"Wrote {out}")
print(f"n={len(rows)}, feasible={feasible.sum()}, max L/D={np.nanmax(ld):.2f}")
