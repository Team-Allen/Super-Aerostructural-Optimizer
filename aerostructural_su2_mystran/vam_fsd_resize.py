"""
VAM Fully-Stressed-Design (FSD) resizing engine.

Build 2 Module 4 (PIML_PIPELINE.md Sec 16): given MYSTRAN's per-element
Tsai-Wu failure index, propose an updated ply thickness scale per element --
a direct closed-form stress-ratio inversion, not a trained network or search.

thickness_new = thickness_old * (FI_actual / FI_target), clipped to bounds.
"""

from __future__ import annotations

import numpy as np


def fsd_resize(
    element_failure_index: dict[int, float],
    current_thickness_scale: dict[int, float],
    target_fi: float = 0.8,
    min_scale: float = 0.2,
    max_scale: float = 5.0,
    damping: float = 1.0,
) -> dict[int, float]:
    """Stress-ratio (fully-stressed-design) thickness update, per element.

    Args:
        element_failure_index: {element_id: Tsai-Wu FI} from MYSTRAN.
        current_thickness_scale: {element_id: current thickness multiplier}.
        target_fi: Design failure-index target (e.g. 0.8, matching the
            existing VAM-beam pipeline's max_failure_index constraint).
        min_scale, max_scale: Manufacturing / design bounds on the multiplier.
        damping: 1.0 = full stress-ratio step; <1.0 under-relaxes the update
            (useful once FI is already near-target, to avoid overshoot).

    Returns:
        {element_id: new thickness multiplier}.
    """
    new_scale = {}
    for eid, scale in current_thickness_scale.items():
        fi = max(element_failure_index.get(eid, 0.0), 1e-12)
        ratio = fi / target_fi
        # Full stress-ratio step, damped, then clipped to design bounds.
        step = 1.0 + damping * (ratio - 1.0)
        new_scale[eid] = float(np.clip(scale * step, min_scale, max_scale))
    return new_scale


def apply_thickness_scale(base_thickness: float, scale: dict[int, float]) -> dict[int, float]:
    """Convert a per-element scale multiplier into an absolute ply thickness."""
    return {eid: base_thickness * s for eid, s in scale.items()}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from piml_mdo.structures.mystran_runner import MystranRunner
    from fsi_transfer import run_transfer

    BASE_PLY_THICKNESS = 0.000131  # IM7/8552 t_ply [m]
    TARGET_FI = 0.8
    N_ITERS = 4

    workdir = "mdo-tools/su2_validation/mystran_run"
    bdf_path = f"{workdir}/wingbox_su2load.bdf"

    runner = MystranRunner(bdf_path=bdf_path, workdir=workdir)
    model = runner._get_bdf()
    mat8 = model.materials[1]
    mat8.Xt, mat8.Xc, mat8.Yt, mat8.Yc, mat8.S = 2326e6, 1200e6, 62.3e6, 199.8e6, 92.3e6

    # Re-transfer the SU2 pressure field (unchanged across iterations here --
    # only the composite design changes, matching Build 2's "SU2 reruns only
    # on shape change" rule; this loop varies only structural sizing).
    result = run_transfer(
        f"mdo-tools/su2_validation/surface_flow_wing.csv",
        f"mdo-tools/su2_validation/wingbox.bdf",
    )
    element_pressures = dict(zip(result["eids"], result["element_pressure"]))
    runner.set_pressure_field(load_id=1, element_pressures=element_pressures)

    # NOTE (honest limitation found while running this smoke test): this
    # wingbox has a single shared PCOMP property for the whole structure (no
    # per-zone breakdown), so the per-element FSD formula's output must be
    # collapsed to one scale to apply here. The per-element math in
    # fsd_resize() is still real and correct -- extending build_wingbox_bdf /
    # set_pcomp_layup to multiple PCOMP zones (matching Build 1's existing
    # n_struct_sizing=3-zone convention) is the next step to get true
    # per-zone resizing. For this run, drive a single global scale from the
    # worst-case (max) element FI -- a standard, valid FSD formulation for a
    # uniformly-sized design.
    scale = 1.0
    damping = 0.7

    print(f"{'iter':>4} | {'max FI':>10} | {'mean FI':>10} | {'scale':>10}")
    for it in range(N_ITERS):
        for pid, prop in model.properties.items():
            if prop.type == "PCOMP":
                prop.thicknesses = [BASE_PLY_THICKNESS * scale] * len(prop.thicknesses)

        runner.write_bdf()
        res = runner.run(write_model=False, cleanup=True, timeout=60)
        if res.exit_code != 0:
            print(f"MYSTRAN failed at iter {it}: exit {res.exit_code}")
            break

        fi = runner.parse_element_failure_indices(runner.workdir / (runner.bdf_path.stem + ".F06"))
        if not fi:
            print("No failure-index data parsed -- stopping.")
            break

        vals = np.array(list(fi.values()))
        print(f"{it:>4} | {vals.max():>10.6f} | {vals.mean():>10.6f} | {scale:>10.4f}")

        max_fi = float(vals.max())
        ratio = max(max_fi, 1e-12) / TARGET_FI
        step = 1.0 + damping * (ratio - 1.0)
        scale = float(np.clip(scale * step, 0.05, 20.0))

    print("Final scale:", scale)
