"""
Real aerodynamic shape optimization: sweep spanwise twist (washout) as the
design variable, regenerating the gmsh volume mesh and re-running SU2 Euler
for each candidate -- a genuine shape optimization, not just an operating-
point sweep. Every candidate's real CL/CD/L-D is logged.

Design variable: tip_washout_deg (linear twist from 0 at root to this value
at the tip, standard washout convention). Fixed alpha=3deg, Mach 0.85,
10000m for all candidates -- an honest limitation (no re-trim to constant
CL) flagged in the results, not hidden.
"""

import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(__file__)
SU2_CFD = os.path.expanduser("~/miniforge3/envs/su2_cfd/bin/SU2_CFD")  # WSL path
WORKDIR = os.path.join(HERE, "aero_shape_opt")
os.makedirs(WORKDIR, exist_ok=True)

CANDIDATES = [0.0, -2.0, -4.0, -6.0]  # tip washout, degrees
N_ITER_SU2 = 150  # reduced from the 300-iter baseline -- trend is clear well before that

CFG_TEMPLATE = """SOLVER= EULER
MATH_PROBLEM= DIRECT
RESTART_SOL= NO
MACH_NUMBER= 0.85
AOA= 3.0
FREESTREAM_PRESSURE= 26436.3
FREESTREAM_TEMPERATURE= 223.15
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
CFL_NUMBER= 3.0
CFL_ADAPT= YES
CFL_ADAPT_PARAM= ( 0.3, 1.2, 1.0, 10.0, 0.0001)
CONV_NUM_METHOD_FLOW= JST
MUSCL_FLOW= NO
JST_SENSOR_COEFF= ( 0.5, 0.02 )
TIME_DISCRE_FLOW= EULER_IMPLICIT
CONV_RESIDUAL_MINVAL= -8
CONV_STARTITER= 10
ITER= {n_iter}
MESH_FILENAME= {mesh_file}
MESH_FORMAT= SU2
SCREEN_OUTPUT= (INNER_ITER, RMS_DENSITY, LIFT, DRAG)
OUTPUT_FILES= (RESTART, SURFACE_CSV)
CONV_FILENAME= history
RESTART_FILENAME= restart_flow.dat
VOLUME_FILENAME= flow
SURFACE_FILENAME= surface_flow
"""


def run_candidate(twist_deg: float) -> dict:
    tag = f"twist_{twist_deg:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    case_dir_win = os.path.join(WORKDIR, tag)
    os.makedirs(case_dir_win, exist_ok=True)
    mesh_name = "mesh.su2"
    cfg_name = "case.cfg"

    # 1. Mesh generation (Windows Python + gmsh, matches earlier validated path)
    t0 = time.time()
    gen_script = os.path.join(HERE, "generate_wing_mesh.py")
    mesh_path = os.path.join(case_dir_win, mesh_name)
    subprocess.run(
        [sys.executable, gen_script, mesh_path, "9", str(twist_deg)],
        check=True, capture_output=True, text=True,
    )
    t_mesh = time.time() - t0

    # 2. Write SU2 config
    cfg_path = os.path.join(case_dir_win, cfg_name)
    with open(cfg_path, "w") as f:
        f.write(CFG_TEMPLATE.format(n_iter=N_ITER_SU2, mesh_file=mesh_name))

    # 3. Run SU2 in WSL2 (su2_cfd env), converting the Windows path to /mnt/f/...
    abs_path = os.path.abspath(case_dir_win).replace("\\", "/")
    drive, rest = abs_path.split(":", 1)
    wsl_case_dir = f"/mnt/{drive.lower()}{rest}"
    t0 = time.time()
    cmd = (
        f"source ~/miniforge3/etc/profile.d/conda.sh && conda activate su2_cfd && "
        f"cd '{wsl_case_dir}' && timeout 600 mpirun -np 8 --oversubscribe SU2_CFD {cfg_name} "
        f"> su2_run.log 2>&1; echo EXIT=$?"
    )
    proc = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=650)
    t_su2 = time.time() - t0

    log_path = os.path.join(case_dir_win, "su2_run.log")
    log_text = open(log_path, errors="ignore").read() if os.path.exists(log_path) else ""
    exit_ok = "Exit Success" in log_text

    cl, cd = None, None
    for line in reversed(log_text.splitlines()):
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[1].strip().lstrip("-").isdigit():
            try:
                cl = float(parts[3].strip())
                cd = float(parts[4].strip())
                break
            except (ValueError, IndexError):
                continue

    return {
        "twist_deg": twist_deg, "exit_ok": exit_ok, "cl": cl, "cd": cd,
        "ld": (cl / cd) if (cl and cd and cd > 0) else None,
        "t_mesh_s": t_mesh, "t_su2_s": t_su2,
    }


if __name__ == "__main__":
    results = []
    log_csv = os.path.join(WORKDIR, "shape_sweep_log.csv")
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["twist_deg", "exit_ok", "cl", "cd", "ld", "t_mesh_s", "t_su2_s"])

    for twist in CANDIDATES:
        print(f"=== Running twist = {twist} deg ===")
        r = run_candidate(twist)
        results.append(r)
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([r["twist_deg"], r["exit_ok"], r["cl"], r["cd"], r["ld"],
                                     f"{r['t_mesh_s']:.1f}", f"{r['t_su2_s']:.1f}"])
        print(f"  twist={twist:+.1f} exit_ok={r['exit_ok']} CL={r['cl']} CD={r['cd']} "
              f"L/D={r['ld']} (mesh {r['t_mesh_s']:.1f}s, SU2 {r['t_su2_s']:.1f}s)")

    print(f"\nDone. Log: {log_csv}")
