"""Meaningful integration proof for the full MDO chain.

This script runs targeted causal checks and generates a matplotlib report:
1) Stage-order and initialization evidence
2) Direct handoff health (aero -> transfer -> TACS -> transfer)
3) One-way vs coupled response difference
4) Design perturbation sensitivity
5) Optimization progress trend
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Super_Aerostructural_Optimizer.pipeline.config import PipelineConfig, load_pipeline_config
from Super_Aerostructural_Optimizer.pipeline.geometry import (
    build_spanwise_aero_grid,
    distribute_panel_forces_to_nodal_loads,
)
from Super_Aerostructural_Optimizer.pipeline.openmdao_opt import run_openmdao_optimization
from Super_Aerostructural_Optimizer.pipeline.workflow import AerostructuralMDOPipeline, DesignVariables


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _make_cfg(base: PipelineConfig, results_dir: str, coupling_iters: int) -> PipelineConfig:
    cfg = copy.deepcopy(base)
    cfg.output.results_dir = results_dir
    cfg.output.write_tacs_solution = False
    cfg.output.save_optimizer_db = False
    cfg.coupling.max_iterations = coupling_iters
    return cfg


def _stage_order_ok(init_events: List[Dict[str, object]]) -> bool:
    expected = [
        "pipeline_init_start",
        "stage_geometry_built",
        "stage_bdf_written",
        "stage_tacs_initialized",
        "stage_transfer_initialized",
        "stage_aero_initialized",
        "pipeline_init_complete",
    ]
    seq = [e.get("event") for e in init_events if e.get("event") in expected]
    pos = 0
    for name in expected:
        while pos < len(seq) and seq[pos] != name:
            pos += 1
        if pos >= len(seq):
            return False
        pos += 1
    return True


def _handoff_probe(pipeline: AerostructuralMDOPipeline, design: DesignVariables) -> Dict[str, float]:
    span_grid = build_spanwise_aero_grid(
        pipeline.mesh,
        twist_root_delta_deg=design.twist_root_delta_deg,
        twist_tip_delta_deg=design.twist_tip_delta_deg,
    )
    aero = pipeline.aero_solver.solve_trimmed(span_grid)
    aero_nodal = distribute_panel_forces_to_nodal_loads(pipeline.mesh, aero.panel_forces_xyz)
    struct_loads = pipeline.transfer.transfer_loads(aero_nodal)
    struct_res = pipeline.structure.solve(struct_loads, pipeline.mesh.tip_node_indices)
    aero_disps = pipeline.transfer.transfer_displacements(struct_res.nodal_displacements_xyz)

    return {
        "aero_load_norm": float(np.linalg.norm(aero_nodal)),
        "struct_load_norm": float(np.linalg.norm(struct_loads)),
        "struct_disp_norm": float(np.linalg.norm(struct_res.nodal_displacements_xyz)),
        "aero_disp_norm": float(np.linalg.norm(aero_disps)),
        "all_finite": bool(
            np.all(np.isfinite(aero_nodal))
            and np.all(np.isfinite(struct_loads))
            and np.all(np.isfinite(struct_res.nodal_displacements_xyz))
            and np.all(np.isfinite(aero_disps))
        ),
        "tip_deflection_probe": float(struct_res.tip_deflection_m),
        "cl_probe": float(aero.cl),
        "cd_probe": float(aero.cd),
    }


def _plot_report(
    out_png: Path,
    init_events: List[Dict[str, object]],
    handoff: Dict[str, float],
    one_way: Dict[str, object],
    coupled: Dict[str, object],
    perturbed: Dict[str, object],
    opt_hist: List[float],
    checks: Dict[str, bool],
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Meaningful MDO Integration Proof", fontsize=16, fontweight="bold")

    # Panel 1: Stage timeline
    expected = [
        "pipeline_init_start",
        "stage_geometry_built",
        "stage_bdf_written",
        "stage_tacs_initialized",
        "stage_transfer_initialized",
        "stage_aero_initialized",
        "pipeline_init_complete",
    ]
    t0 = None
    points = []
    for e in init_events:
        if not isinstance(e.get("timestamp_utc"), str):
            continue
        if t0 is None:
            t0 = _ts(e["timestamp_utc"])
        if e.get("event") in expected:
            dt = (_ts(e["timestamp_utc"]) - t0).total_seconds()
            points.append((e["event"], dt))
    ax = axs[0, 0]
    for i, name in enumerate(expected):
        vals = [t for n, t in points if n == name]
        if vals:
            ax.scatter(vals[0], i, s=60)
            ax.text(vals[0] + 0.01, i + 0.05, f"{vals[0]:.3f}s", fontsize=8)
    ax.set_yticks(range(len(expected)))
    ax.set_yticklabels(expected)
    ax.set_xlabel("Seconds from init start")
    ax.set_title("Stage Order + Timing")
    ax.grid(True, alpha=0.3)

    # Panel 2: Handoff norms + finite flag
    ax = axs[0, 1]
    labels = ["||F_a||", "||F_s||", "||u_s||", "||u_a||"]
    vals = [
        handoff["aero_load_norm"],
        handoff["struct_load_norm"],
        handoff["struct_disp_norm"],
        handoff["aero_disp_norm"],
    ]
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_title("Direct Handoff Signal Propagation")
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.text(
        0.02,
        0.95,
        f"all_finite={handoff['all_finite']}",
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # Panel 3: One-way vs coupled + perturbation
    ax = axs[1, 0]
    metrics = ["CD", "|Tip defl|", "CL"]
    one_vals = [
        float(one_way["aero"]["cd"]),
        abs(float(one_way["structure"]["tip_deflection_m"])),
        float(one_way["aero"]["cl"]),
    ]
    cou_vals = [
        float(coupled["aero"]["cd"]),
        abs(float(coupled["structure"]["tip_deflection_m"])),
        float(coupled["aero"]["cl"]),
    ]
    per_vals = [
        float(perturbed["aero"]["cd"]),
        abs(float(perturbed["structure"]["tip_deflection_m"])),
        float(perturbed["aero"]["cl"]),
    ]
    x = np.arange(len(metrics))
    w = 0.25
    ax.bar(x - w, one_vals, width=w, label="one-way (1 iter)")
    ax.bar(x, cou_vals, width=w, label="coupled")
    ax.bar(x + w, per_vals, width=w, label="perturbed (twist_tip +1 deg)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("Causality: Coupling + Design Perturbation Change Outputs")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    # Panel 4: Optimization trend + pass/fail checks
    ax = axs[1, 1]
    if opt_hist:
        ax.plot(np.arange(1, len(opt_hist) + 1), opt_hist, marker="o")
    ax.set_title("Optimization Objective by Evaluation")
    ax.set_xlabel("Evaluation index")
    ax.set_ylabel("objective_cd (penalized)")
    ax.grid(True, alpha=0.3)

    text_lines = [f"{k}: {'PASS' if v else 'FAIL'}" for k, v in checks.items()]
    ax.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=ax.transAxes,
        va="top",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Meaningful full-chain proof")
    parser.add_argument("--config", default="configs/real_physics_pipeline.json")
    parser.add_argument(
        "--out-dir",
        default="results/real_physics/meaningful_proof",
        help="Output directory for proof artifacts",
    )
    parser.add_argument("--opt-iters", type=int, default=5)
    parser.add_argument("--coupled-iters", type=int, default=8)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (repo / cfg_path).resolve()
    base = load_pipeline_config(cfg_path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (repo / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    init_events: List[Dict[str, object]] = []
    cfg_probe = _make_cfg(base, str((out_dir / "probe_run").resolve()), coupling_iters=2)
    probe = AerostructuralMDOPipeline(cfg_probe, event_callback=lambda e: init_events.append(e))
    design0 = probe.default_design()
    handoff = _handoff_probe(probe, design0)

    # One-way baseline (single coupling iteration)
    cfg_one = _make_cfg(base, str((out_dir / "one_way_run").resolve()), coupling_iters=1)
    p_one = AerostructuralMDOPipeline(cfg_one)
    res_one = p_one.evaluate(design0)

    # Coupled baseline
    cfg_cpl = _make_cfg(base, str((out_dir / "coupled_run").resolve()), coupling_iters=args.coupled_iters)
    p_cpl = AerostructuralMDOPipeline(cfg_cpl)
    res_cpl = p_cpl.evaluate(design0)

    # Perturbation case
    d_pert = DesignVariables(
        twist_root_delta_deg=design0.twist_root_delta_deg,
        twist_tip_delta_deg=design0.twist_tip_delta_deg + 1.0,
        thickness_root_scale=design0.thickness_root_scale,
        thickness_tip_scale=design0.thickness_tip_scale,
    )
    res_pert = p_cpl.evaluate(d_pert)

    # Optimization trend (few iterations, but enough for proof)
    opt_events: List[Dict[str, object]] = []
    cfg_opt = _make_cfg(base, str((out_dir / "opt_run").resolve()), coupling_iters=6)
    cfg_opt.optimizer.max_iterations = int(args.opt_iters)
    p_opt = AerostructuralMDOPipeline(cfg_opt, event_callback=lambda e: opt_events.append(e))
    sol = run_openmdao_optimization(p_opt, record_path=None)
    opt_hist = [float(e["objective_cd"]) for e in opt_events if e.get("event") == "evaluation_complete"]

    # Causal deltas
    delta_cd_coupling = float(res_cpl["aero"]["cd"]) - float(res_one["aero"]["cd"])
    delta_tip_coupling = float(res_cpl["structure"]["tip_deflection_m"]) - float(
        res_one["structure"]["tip_deflection_m"]
    )
    delta_cd_pert = float(res_pert["aero"]["cd"]) - float(res_cpl["aero"]["cd"])
    delta_tip_pert = float(res_pert["structure"]["tip_deflection_m"]) - float(
        res_cpl["structure"]["tip_deflection_m"]
    )

    checks = {
        "stage_order_ok": _stage_order_ok(init_events),
        "handoff_all_finite": bool(handoff["all_finite"]),
        "handoff_signal_present": bool(
            handoff["aero_load_norm"] > 1e-8
            and handoff["struct_load_norm"] > 1e-8
            and handoff["struct_disp_norm"] > 1e-8
            and handoff["aero_disp_norm"] > 1e-8
        ),
        "coupling_changes_response": bool(
            abs(delta_cd_coupling) > 1e-6 or abs(delta_tip_coupling) > 1e-4
        ),
        "design_perturbation_changes_response": bool(
            abs(delta_cd_pert) > 1e-6 or abs(delta_tip_pert) > 1e-4
        ),
        "optimization_progress": bool(
            len(opt_hist) >= 2 and min(opt_hist) < (opt_hist[0] - 1e-6)
        ),
    }
    overall_ok = all(checks.values())

    out_plot = out_dir / "meaningful_workflow_proof.png"
    _plot_report(out_plot, init_events, handoff, res_one, res_cpl, res_pert, opt_hist, checks)

    summary = {
        "overall_ok": overall_ok,
        "checks": checks,
        "handoff_probe": handoff,
        "coupling_vs_oneway_delta": {
            "delta_cd": delta_cd_coupling,
            "delta_tip_deflection_m": delta_tip_coupling,
        },
        "perturbation_delta": {
            "delta_cd": delta_cd_pert,
            "delta_tip_deflection_m": delta_tip_pert,
            "base_design": asdict(design0),
            "perturbed_design": asdict(d_pert),
        },
        "optimization": {
            "optimizer": sol.get("optimizer"),
            "objective_history": opt_hist,
            "final_outputs": sol.get("outputs", {}),
        },
        "outputs": {
            "proof_plot": str(out_plot),
            "summary_json": str(out_dir / "meaningful_workflow_proof_summary.json"),
        },
    }
    out_json = out_dir / "meaningful_workflow_proof_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved proof plot:", out_plot)
    print("Saved summary:", out_json)
    print(json.dumps(summary, indent=2))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
