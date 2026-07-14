"""
4th SU2 refinement point: the diverse rank-5 VLM candidate, run separately
from doe_su2_refine.py's top-3 (which are a tight cluster) to get real
coverage of a genuinely different region of the design space.
"""
import sys
sys.path.insert(0, ".")
from doe_su2_refine import run_case
import csv

t_mid, t_tip, alpha = 0.23406244040319724, -1.3254176369223476, 4.8319193121851844
vlm_cl, vlm_cd, vlm_ld = 0.35858089441263086, 0.014732286989379492, 24.339798340280222

print(f"=== Refining rank 5 (diverse): twist_mid={t_mid:+.3f} twist_tip={t_tip:+.3f} "
      f"alpha={alpha:.3f} (VLM L/D={vlm_ld}) ===", flush=True)
r = run_case("rank5", t_mid, t_tip, alpha)
su2_ld = (r["cl"] / r["cd"]) if (r["cl"] and r["cd"] and r["cd"] > 0) else None
print(f"  -> SU2: exit_ok={r['exit_ok']} CL={r['cl']} CD={r['cd']} L/D={su2_ld} "
      f"residual={r['residual']} (mesh {r['t_mesh_s']:.0f}s, su2 {r['t_su2_s']:.0f}s)", flush=True)

with open("doe_su2_refine/su2_refine_log.csv", "a", newline="") as f:
    csv.writer(f).writerow([
        5, "rank5", t_mid, t_tip, alpha, vlm_cl, vlm_cd, vlm_ld,
        r["exit_ok"], r["cl"], r["cd"], su2_ld, r["residual"],
        r.get("n_iter_completed"), r["t_mesh_s"], r["t_su2_s"],
    ])
print("Done.")
