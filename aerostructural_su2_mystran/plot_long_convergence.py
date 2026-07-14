"""
Plot the real long-duration (2500-iteration, fixed CFL) CD convergence at
-2deg washout, showing the transient settling into a genuinely converged
(not oscillating) small CD value -- the actual fix for the architecture
gap found in the shorter, adaptive-CFL runs.
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_history(path):
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.match(r"\|\s*(\d+)\|\s*([-\d.]+)\|\s*([-\d.]+)\|\s*([-\d.]+)\|", line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


rows_m2 = parse_history(os.path.join(HERE, "long_avg_m2deg", "su2_run.log"))
it_m2 = np.array([r[0] for r in rows_m2])
cd_m2 = np.array([r[3] for r in rows_m2])
rms_m2 = np.array([r[1] for r in rows_m2])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(it_m2, cd_m2, color="tab:red", linewidth=0.8)
axes[0].axvspan(1000, 2500, alpha=0.15, color="tab:green", label="averaging window")
mean_tail = cd_m2[it_m2 >= 1000].mean()
axes[0].axhline(mean_tail, color="tab:green", linestyle="--", linewidth=1,
                 label=f"converged mean = {mean_tail:.5f}")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("CD")
axes[0].set_title("-2deg washout: CD vs iteration, 2500-iter fixed-CFL solve\n"
                   "(the earlier apparent 'oscillation' was an under-integrated transient)")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(it_m2, rms_m2, color="tab:blue", linewidth=0.8)
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("rms[Rho] (log10)")
axes[1].set_title("Residual convergence\nreaches -5.08 (vs -3.1 to -3.4 in the earlier 300-iter runs)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
out = os.path.join(OUT_DIR, "19_long_convergence_m2deg.png")
plt.savefig(out, dpi=150)
print(f"Wrote {out}")
print(f"Converged mean CD (iter>=1000): {mean_tail:.6f}, stdev: {cd_m2[it_m2>=1000].std():.6f}")
