"""
Plot the multi-fidelity DOE funnel result: real VLM-predicted L/D vs real
SU2-verified L/D for the refined candidates, showing the systematic gap
from missing wave drag at Mach 0.85.
"""
import csv
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(HERE, "doe_su2_refine", "su2_refine_log.csv")) as f:
    rows = list(csv.DictReader(f))

ranks = [int(r["rank"]) for r in rows]
vlm_ld = [float(r["vlm_LD"]) for r in rows]
su2_ld = [float(r["su2_LD"]) for r in rows]
labels = [f"rank{r}" for r in ranks]

fig, ax = plt.subplots(figsize=(8, 6))
x = range(len(ranks))
w = 0.35
ax.bar([i - w/2 for i in x], vlm_ld, width=w, label="VLM prediction (screening)", color="tab:orange")
ax.bar([i + w/2 for i in x], su2_ld, width=w, label="SU2 verified (real, converged)", color="tab:red")
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("L/D")
ax.set_title("Multi-fidelity DOE: VLM screening prediction vs real SU2-verified L/D\n"
             "(VLM systematically overestimates -- missing Mach 0.85 wave drag)")
ax.legend()
ax.grid(alpha=0.3, axis="y")
for i, (v, s) in enumerate(zip(vlm_ld, su2_ld)):
    ax.annotate(f"{v:.1f}", (i - w/2, v), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{s:.1f}", (i + w/2, s), ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = os.path.join(OUT_DIR, "22_doe_vlm_vs_su2.png")
plt.savefig(out, dpi=150)
print(f"Wrote {out}")
for r in rows:
    print(f"  {r['tag']}: VLM L/D={r['vlm_LD']}, SU2 L/D={r['su2_LD']}, "
          f"ratio={float(r['vlm_LD'])/float(r['su2_LD']):.2f}x")
