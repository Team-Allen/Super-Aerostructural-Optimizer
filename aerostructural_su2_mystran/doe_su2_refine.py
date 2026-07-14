"""
Stage 2 of the multi-fidelity DOE funnel: real SU2 Euler refinement of the
top candidates found by the real 200-point VLM DOE screening
(doe_vlm_screening.py). Uses the proven, properly-converged methodology
established in README Sec2/Sec8: 2500 iterations, FIXED (non-adaptive) CFL
-- not the short, adaptive-CFL runs shown to give unreliable CD reads.

Reads doe_vlm/top_candidates.csv (real VLM screening results) rather than
hardcoded design points, so this script always refines whatever the
screening actually found, not an assumed "obvious" answer.
"""

import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
WORKDIR = os.path.join(HERE, "doe_su2_refine")
os.makedirs(WORKDIR, exist_ok=True)

N_REFINE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ITER_LONG = 2500
MACH = 0.85
FREESTREAM_P = 26436.3
FREESTREAM_T = 223.15

CFG_TEMPLATE = """SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= {mach}
AOA= {alpha}
FREESTREAM_PRESSURE= {p}
FREESTREAM_TEMPERATURE= {t}
REF_ORIGIN_MOMENT_X = 0.896
REF_ORIGIN_MOMENT_Y = 0.00
REF_ORIGIN_MOMENT_Z = 0.00
REF_LENGTH= 1.792
REF_AREA= 4.9725
MARKER_EULER= ( wing )
MARKER_FAR= ( farfield )
MARKER_PLOTTING= ( wing )
MARKER_MONITORING= ( wing )
LINEAR_SOLVER= FGMRES
LINEAR_SOLVER_PREC= ILU
LINEAR_SOLVER_ERROR= 1E-6
LINEAR_SOLVER_ITER= 5
CFL_NUMBER= {cfl}
CFL_ADAPT= NO
CONV_NUM_METHOD_FLOW= JST
MUSCL_FLOW= NO
JST_SENSOR_COEFF= ( {jst1}, {jst2} )
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_RESIDUAL_MINVAL= -12
CONV_STARTITER= 10
ITER= {n_iter}
MESH_FILENAME= mesh.su2
MESH_FORMAT= SU2
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
OUTPUT_FILES= (RESTART, SURFACE_CSV)
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
"""


def run_case(tag, twist_mid, twist_tip, alpha, cfl=1.0, jst=(0.7, 0.04)):
    case_dir = os.path.join(WORKDIR, tag)
    os.makedirs(case_dir, exist_ok=True)
    mesh_path = os.path.join(case_dir, "mesh.su2")

    gen_script = os.path.join(HERE, "generate_wing_mesh.py")
    t0 = time.time()
    subprocess.run(
        [sys.executable, gen_script, mesh_path, "9", str(twist_tip), str(twist_mid)],
        check=True, capture_output=True, text=True,
    )
    t_mesh = time.time() - t0

    cfg_path = os.path.join(case_dir, "case.cfg")
    with open(cfg_path, "w") as f:
        f.write(CFG_TEMPLATE.format(
            mach=MACH, alpha=alpha, p=FREESTREAM_P, t=FREESTREAM_T,
            cfl=cfl, jst1=jst[0], jst2=jst[1], n_iter=ITER_LONG,
        ))

    abs_path = os.path.abspath(case_dir).replace("\\", "/")
    drive, rest = abs_path.split(":", 1)
    wsl_case_dir = f"/mnt/{drive.lower()}{rest}"
    t0 = time.time()
    cmd = (
        f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate su2_cfd && "
        f"cd '{wsl_case_dir}' && timeout 3600 mpirun -np 8 --oversubscribe SU2_CFD "
        f"case.cfg > su2_run.log 2>&1; echo EXIT=$?"
    )
    subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-lc", cmd],
                   capture_output=True, text=True, timeout=3700)
    t_su2 = time.time() - t0

    log_path = os.path.join(case_dir, "su2_run.log")
    log_text = open(log_path, errors="ignore").read() if os.path.exists(log_path) else ""
    exit_ok = "Exit Success" in log_text

    import re
    rows = []
    for line in log_text.splitlines():
        m = re.match(r"\|\s*(\d+)\|\s*([-\d.]+)\|\s*([-\d.]+)\|\s*([-\d.]+)\|", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))

    if not rows:
        return {"exit_ok": False, "cl": None, "cd": None, "residual": None,
                "t_mesh_s": t_mesh, "t_su2_s": t_su2}

    tail = [r for r in rows if r[0] >= max(rows[-1][0] - 1500, 0)]
    cl_mean = sum(r[2] for r in tail) / len(tail)
    cd_mean = sum(r[3] for r in tail) / len(tail)
    final_residual = rows[-1][1]

    return {
        "exit_ok": exit_ok, "cl": cl_mean, "cd": cd_mean, "residual": final_residual,
        "t_mesh_s": t_mesh, "t_su2_s": t_su2, "n_iter_completed": rows[-1][0],
    }


def main():
    with open(os.path.join(HERE, "doe_vlm", "top_candidates.csv")) as f:
        candidates = list(csv.DictReader(f))[:N_REFINE]

    log_path = os.path.join(WORKDIR, "su2_refine_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "tag", "twist_mid_deg", "twist_tip_deg", "alpha_deg",
                    "vlm_CL", "vlm_CD", "vlm_LD",
                    "su2_exit_ok", "su2_CL", "su2_CD", "su2_LD", "su2_residual",
                    "n_iter_completed", "t_mesh_s", "t_su2_s"])

    for c in candidates:
        rank = c["rank"]
        tag = f"rank{rank}"
        t_mid, t_tip, alpha = float(c["twist_mid_deg"]), float(c["twist_tip_deg"]), float(c["alpha_deg"])
        print(f"=== Refining rank {rank}: twist_mid={t_mid:+.3f} twist_tip={t_tip:+.3f} "
              f"alpha={alpha:.3f} (VLM L/D={c['LD']}) ===", flush=True)
        r = run_case(tag, t_mid, t_tip, alpha)
        su2_ld = (r["cl"] / r["cd"]) if (r["cl"] and r["cd"] and r["cd"] > 0) else None
        print(f"  -> SU2: exit_ok={r['exit_ok']} CL={r['cl']} CD={r['cd']} L/D={su2_ld} "
              f"residual={r['residual']} (mesh {r['t_mesh_s']:.0f}s, su2 {r['t_su2_s']:.0f}s)", flush=True)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                rank, tag, t_mid, t_tip, alpha,
                c["CL"], c["CD"], c["LD"],
                r["exit_ok"], r["cl"], r["cd"], su2_ld, r["residual"],
                r.get("n_iter_completed"), r["t_mesh_s"], r["t_su2_s"],
            ])

    print(f"\nDone. Refinement log: {log_path}")


if __name__ == "__main__":
    main()
