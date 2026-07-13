"""
Plot real MYSTRAN structural results (deformed wingbox shape) and the VAM-FSD
convergence history, from the actual data produced tonight.
"""
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

F06_PATH = "wingbox_su2load.F06"


def parse_displacements(f06_path):
    text = open(f06_path, errors="ignore").read()
    pattern = re.compile(
        r"D\s+I\s+S\s+P\s+L\s+A\s+C\s+E\s+M\s+E\s+N\s+T\s+S.*?\n.*?GRID.*?\n.*?\n(.*?)\n\s*\n",
        re.DOTALL,
    )
    m = pattern.search(text)
    rows = []
    for line in m.group(1).splitlines():
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(nums) >= 7:
            rows.append([float(x) for x in nums[:7]])
    return np.array(rows)  # grid, coord, T1, T2, T3, R1, R2


def parse_grids(bdf_path):
    grids = {}
    for line in open(bdf_path, errors="ignore"):
        if line.startswith("GRID"):
            gid = int(line[8:16])
            def pf(s):
                s = s.strip()
                mm = re.match(r"^(-?\d*\.?\d*)([+-]\d+)$", s)
                if mm and "." in mm.group(1):
                    return float(mm.group(1) + "E" + mm.group(2))
                return float(s) if s else 0.0
            x, y, z = pf(line[24:32]), pf(line[32:40]), pf(line[40:48])
            grids[gid] = (x, y, z)
    return grids


disp = parse_displacements(F06_PATH)
grids = parse_grids("wingbox_su2load.bdf")

gids = disp[:, 0].astype(int)
coords = np.array([grids[g] for g in gids])
t1, t2, t3 = disp[:, 2], disp[:, 3], disp[:, 4]

SCALE = 20.0  # visually exaggerate the real (small) deflections
deformed = coords.copy()
deformed[:, 0] += t1 * SCALE
deformed[:, 1] += t2 * SCALE
deformed[:, 2] += t3 * SCALE

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(coords[:, 1], coords[:, 0], coords[:, 2], c="steelblue", s=15, label="Undeformed")
ax.scatter(deformed[:, 1], deformed[:, 0], deformed[:, 2], c="crimson", s=15, label="Deformed (20x)")
ax.set_xlabel("Span y [m]")
ax.set_ylabel("Chord x [m]")
ax.set_zlabel("Vertical z [m]")
ax.set_title("MYSTRAN wingbox: real SU2-transferred pressure load\n(deflection exaggerated 20x for visibility)")
ax.legend()

ax2 = fig.add_subplot(122)
order = np.argsort(coords[:, 1])
ax2.plot(coords[order, 1], t3[order] * 1000, "o-", color="crimson")
ax2.set_xlabel("Span position y [m]")
ax2.set_ylabel("Vertical deflection T3 [mm]")
ax2.set_title("Real spanwise deflection (MYSTRAN F06, final FSD iteration)")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("renders/06_mystran_deflection.png", dpi=150)
print("Saved renders/06_mystran_deflection.png")
print(f"Max T3 deflection: {np.max(np.abs(t3))*1000:.4f} mm")

# --- FSD convergence plot (real numbers from tonight's run) ---
iters = [0, 1, 2, 3]
max_fi = [16526.986354, 0.507266, 1.069163, 0.623420]
scale = [1.0, 20.0, 14.8772, 18.3810]

fig2, (axa, axb) = plt.subplots(1, 2, figsize=(12, 5))
axa.semilogy(iters, max_fi, "o-", color="darkred")
axa.axhline(0.8, color="k", linestyle="--", label="Target FI = 0.8")
axa.set_xlabel("FSD iteration")
axa.set_ylabel("Max Tsai-Wu failure index (log scale)")
axa.set_title("VAM-FSD convergence (real MYSTRAN data)")
axa.legend()
axa.grid(alpha=0.3)

axb.plot(iters, scale, "o-", color="steelblue")
axb.set_xlabel("FSD iteration")
axb.set_ylabel("Thickness scale multiplier")
axb.set_title("Thickness scale history")
axb.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("renders/07_fsd_convergence.png", dpi=150)
print("Saved renders/07_fsd_convergence.png")
