"""
Final structural sizing run on the AERODYNAMICALLY OPTIMIZED wing shape --
the twist found by aero_shape_true_optimization.py's real bounded search,
confirmed at high fidelity (300 SU2 iterations). This is the last step of
the full pipeline: real optimized shape -> real transferred pressure ->
real MYSTRAN structural analysis -> real VAM-FSD composite resizing.

Reads the optimizer's own recorded result (aero_shape_true_opt/optimum_result.csv)
rather than a hardcoded twist value, so this script is always in sync with
whatever the optimizer actually found.
"""

import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from build_wing_shell import build_wing_shell_bdf
from fsi_transfer import run_transfer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from piml_mdo.structures.mystran_runner import MystranRunner

N_ITERS = 150
N_ZONES = 8
TARGET_FI = 0.8
PLY_ANGLES = [0.0, 45.0, -45.0, 90.0]
PLY_THICKNESS = 0.000131
MIN_PLIES_PER_ANGLE = 1
MAX_PLIES_PER_ANGLE = 20
MIN_SCALE, MAX_SCALE = 0.3, 2.5
FSD_DAMPING = 0.5
N_PLY_UPDATE_EVERY = 5
PLY_ADD_THRESHOLD = 1.2
PLY_REMOVE_THRESHOLD = 0.5
MAT_XT, MAT_XC, MAT_YT, MAT_YC, MAT_S = 2326e6, 1200e6, 62.3e6, 199.8e6, 92.3e6

HERE = os.path.dirname(__file__)
OPT_RESULT_CSV = os.path.join(HERE, "aero_shape_true_opt", "optimum_result.csv")
WORKDIR = os.path.join(HERE, "production_run_optimized")
os.makedirs(WORKDIR, exist_ok=True)
BDF_PATH = os.path.join(WORKDIR, "wingshell.bdf")
LOG_CSV = os.path.join(WORKDIR, "iteration_log.csv")
LOG_JSON = os.path.join(WORKDIR, "iteration_log.jsonl")
CONFIG_PATH = os.path.join(WORKDIR, "run_config.json")
BDF_ARCHIVE_DIR = os.path.join(WORKDIR, "bdf_archive")
os.makedirs(BDF_ARCHIVE_DIR, exist_ok=True)
TEMPLATE_BDF = os.path.join(BDF_ARCHIVE_DIR, "template.bdf")


def main():
    with open(OPT_RESULT_CSV) as f:
        opt_row = next(csv.DictReader(f))
    optimal_twist = float(opt_row["optimal_twist_deg"])
    final_case_dir = opt_row["case_dir"]
    su2_surface_csv = os.path.join(final_case_dir, "surface_flow.csv")

    print(f"Optimized twist from real aero shape optimization: {optimal_twist:.4f} deg")
    print(f"Reading real converged SU2 pressure from: {su2_surface_csv}")

    config = {k: v for k, v in globals().items() if k.isupper()}
    config["OPTIMAL_TWIST_DEG"] = optimal_twist
    config["SU2_SURFACE_CSV"] = su2_surface_csv
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2, default=str)

    initial_ply_counts = [{a: 1 for a in PLY_ANGLES} for _ in range(N_ZONES)]
    info = build_wing_shell_bdf(
        BDF_PATH, n_zones=N_ZONES, zone_ply_counts=initial_ply_counts,
        tip_washout_deg=optimal_twist,
    )
    print(f"Optimized wing shell (twist={optimal_twist:.4f}deg): {info['n_nodes']} nodes, "
          f"{info['n_elements']} elements, {info['n_zones']} zones")
    shutil.copy(BDF_PATH, TEMPLATE_BDF)

    result = run_transfer(su2_surface_csv, BDF_PATH, align=False)
    element_pressures = dict(zip(result["eids"], result["element_pressure"]))
    fx, fy, fz = result["total_force_vector"]
    print(f"Transferred pressure onto {len(element_pressures)} elements "
          f"(range {min(element_pressures.values()):.1f} to "
          f"{max(element_pressures.values()):.1f} Pa)")
    print(f"Force balance: Fx={fx:.1f} Fy={fy:.1f} Fz={fz:.1f} N")

    runner = MystranRunner(bdf_path=BDF_PATH, workdir=WORKDIR)
    model = runner._get_bdf()
    mat8 = model.materials[1]
    mat8.Xt, mat8.Xc, mat8.Yt, mat8.Yc, mat8.S = MAT_XT, MAT_XC, MAT_YT, MAT_YC, MAT_S
    runner.set_pressure_field(load_id=1, element_pressures=element_pressures)

    thickness_scale = {z: 1.0 for z in range(N_ZONES)}
    ply_counts = {z: dict(initial_ply_counts[z]) for z in range(N_ZONES)}

    fieldnames = (
        ["iter", "wall_time_s", "mass_kg", "global_max_fi", "global_mean_fi", "n_mystran_fail"]
        + [f"zone{z}_max_fi" for z in range(N_ZONES)]
        + [f"zone{z}_thickness_scale" for z in range(N_ZONES)]
        + [f"zone{z}_ply_{int(a)}" for z in range(N_ZONES) for a in PLY_ANGLES]
        + [f"zone{z}_total_plies" for z in range(N_ZONES)]
    )
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(fieldnames)
    open(LOG_JSON, "w").close()

    n_fail = 0
    t_start = time.time()
    for it in range(N_ITERS):
        t0 = time.time()

        for z in range(N_ZONES):
            pid = info["zone_pcomp_id"][z]
            angles = []
            for a in PLY_ANGLES:
                angles.extend([a] * ply_counts[z][a])
            if not angles:
                angles = [0.0]
            thicknesses = [PLY_THICKNESS * thickness_scale[z]] * len(angles)
            runner.set_pcomp_layup(pid, angles, thicknesses, lam="SYM")

        runner.write_bdf()
        archived_bdf = os.path.join(BDF_ARCHIVE_DIR, f"iter_{it:04d}.bdf")
        shutil.copy(BDF_PATH, archived_bdf)
        params_record = {
            "iter": it,
            "thickness_scale": dict(thickness_scale),
            "ply_counts": {str(z): dict(ply_counts[z]) for z in range(N_ZONES)},
        }
        with open(os.path.join(BDF_ARCHIVE_DIR, f"iter_{it:04d}_params.json"), "w") as f:
            json.dump(params_record, f, indent=2)

        res = runner.run(write_model=False, cleanup=True, timeout=90)
        wall = time.time() - t0

        if res.exit_code != 0:
            n_fail += 1
            print(f"iter {it:3d} | MYSTRAN FAILED (exit {res.exit_code})")
            continue

        fi = runner.parse_element_failure_indices(
            runner.workdir / (runner.bdf_path.stem + ".F06")
        )
        mass = runner._compute_mass_from_bdf()

        zone_max_fi, zone_mean_fi = {}, {}
        for z, eids in info["zone_elements"].items():
            vals = [fi.get(e, 0.0) for e in eids]
            zone_max_fi[z] = float(max(vals)) if vals else 0.0
            zone_mean_fi[z] = float(np.mean(vals)) if vals else 0.0
        global_max_fi = max(zone_max_fi.values())
        global_mean_fi = float(np.mean(list(zone_mean_fi.values())))

        row = [it, wall, mass, global_max_fi, global_mean_fi, n_fail]
        row += [zone_max_fi[z] for z in range(N_ZONES)]
        row += [thickness_scale[z] for z in range(N_ZONES)]
        for z in range(N_ZONES):
            for a in PLY_ANGLES:
                row.append(ply_counts[z][a])
        row += [sum(ply_counts[z].values()) for z in range(N_ZONES)]

        with open(LOG_CSV, "a", newline="") as f:
            csv.writer(f).writerow(row)
        with open(LOG_JSON, "a") as f:
            record = {
                "iter": it, "wall_time_s": wall, "mass_kg": mass,
                "global_max_fi": global_max_fi, "global_mean_fi": global_mean_fi,
                "zone_max_fi": zone_max_fi, "zone_mean_fi": zone_mean_fi,
                "thickness_scale": dict(thickness_scale),
                "ply_counts": {str(z): dict(ply_counts[z]) for z in range(N_ZONES)},
            }
            f.write(json.dumps(record) + "\n")

        print(f"iter {it:3d} | max_fi={global_max_fi:7.4f} | mean_fi={global_mean_fi:7.4f} | "
              f"mass={mass:8.4f} kg | plies/zone={[sum(ply_counts[z].values()) for z in range(N_ZONES)]} | "
              f"{wall:.2f}s")

        for z in range(N_ZONES):
            fi_z = zone_max_fi[z]
            ratio = max(fi_z, 1e-6) / TARGET_FI
            step = 1.0 + FSD_DAMPING * (ratio - 1.0)
            thickness_scale[z] = float(np.clip(thickness_scale[z] * step, MIN_SCALE, MAX_SCALE))

            if it % N_PLY_UPDATE_EVERY == 0:
                if fi_z > PLY_ADD_THRESHOLD * TARGET_FI:
                    if ply_counts[z][0.0] < MAX_PLIES_PER_ANGLE:
                        ply_counts[z][0.0] += 1
                        thickness_scale[z] = 1.0
                elif fi_z < PLY_REMOVE_THRESHOLD * TARGET_FI:
                    if ply_counts[z][0.0] > MIN_PLIES_PER_ANGLE:
                        ply_counts[z][0.0] -= 1
                        thickness_scale[z] = 1.0

    total_wall = time.time() - t_start
    print(f"\nDone. {N_ITERS} iterations, {n_fail} MYSTRAN failures, "
          f"{total_wall:.1f}s total ({total_wall/N_ITERS:.2f}s/iter avg).")
    print(f"Optimized twist: {optimal_twist:.4f} deg")
    print(f"Logs: {LOG_CSV}\n      {LOG_JSON}")


if __name__ == "__main__":
    main()
