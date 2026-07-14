"""
Real multi-fidelity DOE screening: a Latin Hypercube Sample over a genuine
3-variable design space (mid-span twist, tip twist, angle of attack),
evaluated with a real 3-D VLM solve (OpenAeroStruct, already built and
validated in Build 1b) at every point -- not a lookup table, not a
surrogate fit ahead of time.

This is stage 1 of a real 2-stage multi-fidelity funnel:
  Stage 1 (this script): ~150-200 real VLM solves, seconds each, wide
    coverage of the design space. VLM has no shock/compressibility
    modeling at Mach 0.85 -- it is used here for real, physically
    grounded RELATIVE ranking (induced drag trends, CL trim behavior,
    spanwise loading), not final absolute drag numbers.
  Stage 2 (doe_su2_refine.py): the top candidates from this screening are
    re-solved with real SU2 Euler at full, properly-converged fidelity
    (the 2500-iteration, tuned-CFL methodology established and validated
    in README Sec2/Sec8) to get trustworthy final numbers.

Design variables (real, physically meaningful, not padding):
  twist_mid_deg : washout at 50% semi-span [-6, 2] deg
  twist_tip_deg : washout at the tip       [-8, 2] deg
  alpha_deg     : trim angle of attack     [1, 5] deg
Root twist is fixed at 0 (the reference datum); the corrected real CAD
planform (semi-span 3.0m, chords 2.688/0.627m, sweep 26.8deg) is NOT a
design variable -- it is fixed by the real aircraft geometry.
"""

import csv
import os
import sys

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from piml_mdo.aero.openaerostruct_solver import OpenAeroStructSolver, WingPlanform

sys.path.insert(0, os.path.dirname(__file__))
from generate_wing_mesh import symmetric_coordinates, TC_ROOT

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "doe_vlm")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 200
BOUNDS = {
    "twist_mid_deg": (-6.0, 2.0),
    "twist_tip_deg": (-8.0, 2.0),
    "alpha_deg": (1.0, 5.0),
}
SEMI_SPAN = 3.0
CHORD_ROOT = 2.688
CHORD_TIP = 0.627
SWEEP_DEG = 26.8
MACH = 0.85
VELOCITY = 254.546
RHO = 0.4127
RE = 5.0e6
CL_MIN = 0.08  # real lift floor, keeps the screened wing doing real work


def main():
    sampler = qmc.LatinHypercube(d=3, seed=42)
    unit_samples = sampler.random(n=N_SAMPLES)
    lo = np.array([BOUNDS[k][0] for k in BOUNDS])
    hi = np.array([BOUNDS[k][1] for k in BOUNDS])
    samples = qmc.scale(unit_samples, lo, hi)

    coords = symmetric_coordinates(TC_ROOT, 24, 1.0)
    x_coords, y_coords = coords[:, 0], coords[:, 1]

    planform = WingPlanform(
        span=2 * SEMI_SPAN, chord_root=CHORD_ROOT, chord_tip=CHORD_TIP, sweep_deg=SWEEP_DEG,
    )
    solver = OpenAeroStructSolver(
        wing_planform=planform, num_x=5, num_y=15, num_twist_cp=3,
        with_viscous=True, with_wave=True, velocity=VELOCITY, rho=RHO,
    )

    log_path = os.path.join(OUT_DIR, "doe_vlm_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "twist_mid_deg", "twist_tip_deg", "alpha_deg",
                    "CL", "CD", "LD", "feasible"])

    results = []
    twist_stations = np.array([0.0, SEMI_SPAN * 0.5, SEMI_SPAN])
    for i, (t_mid, t_tip, alpha) in enumerate(samples):
        twist_deg = np.array([0.0, t_mid, t_tip])
        try:
            dist = solver.solve_wing_distribution(
                coordinates=(x_coords, y_coords),
                twist_stations=twist_stations,
                twist_deg=twist_deg,
                alpha=float(alpha),
                Re=RE, mach=MACH, velocity=VELOCITY, rho=RHO,
            )
            CL, CD = dist.CL, dist.CD
            feasible = CL >= CL_MIN and CD > 0
            LD = CL / CD if feasible else None
        except Exception as e:
            CL, CD, LD, feasible = None, None, None, False
            print(f"sample {i}: FAILED ({e})")

        row = [i, t_mid, t_tip, alpha, CL, CD, LD, feasible]
        results.append(row)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        if i % 20 == 0 or i == N_SAMPLES - 1:
            print(f"[{i+1}/{N_SAMPLES}] twist_mid={t_mid:+.2f} twist_tip={t_tip:+.2f} "
                  f"alpha={alpha:.2f} -> CL={CL} CD={CD} L/D={LD}")

    feasible_results = [r for r in results if r[7] and r[6] is not None]
    feasible_results.sort(key=lambda r: r[6], reverse=True)
    print(f"\n{len(feasible_results)}/{N_SAMPLES} feasible (CL>={CL_MIN}, CD>0)")
    print("\nTop 10 by L/D:")
    for r in feasible_results[:10]:
        print(f"  sample {r[0]:3d}: twist_mid={r[1]:+.3f} twist_tip={r[2]:+.3f} "
              f"alpha={r[3]:.3f} CL={r[4]:.4f} CD={r[5]:.5f} L/D={r[6]:.2f}")

    with open(os.path.join(OUT_DIR, "top_candidates.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "sample", "twist_mid_deg", "twist_tip_deg", "alpha_deg", "CL", "CD", "LD"])
        for rank, r in enumerate(feasible_results[:10]):
            w.writerow([rank] + list(r[:7]))

    print(f"\nDone. Full log: {log_path}")
    print(f"Top candidates: {os.path.join(OUT_DIR, 'top_candidates.csv')}")


if __name__ == "__main__":
    main()
