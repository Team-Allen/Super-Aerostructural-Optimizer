"""Run full workflow and generate a start-to-finish matplotlib proof chart."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _run_workflow(repo_root: Path, config_rel: str, progress_rel: str, cuda_devices: str) -> None:
    cmd = [
        "powershell",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(repo_root / "run_scripts" / "run_real_physics_mdo_wsl.ps1"),
        "-Mode",
        "optimize",
        "-Config",
        config_rel,
        "-ProgressFile",
        progress_rel,
    ]
    if cuda_devices.strip():
        cmd.extend(["-CudaVisibleDevices", cuda_devices.strip()])
    proc = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr)
        raise RuntimeError(f"Workflow command failed with code {proc.returncode}")


def _build_proof_figure(
    progress_file: Path,
    optimum_file: Path,
    out_png: Path,
    out_json: Path,
) -> Dict[str, object]:
    events = []
    for line in progress_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    if not events:
        raise RuntimeError(f"No events found in {progress_file}")

    timed = [e for e in events if isinstance(e.get("timestamp_utc"), str)]
    if not timed:
        raise RuntimeError("No timestamped events found")
    t0 = _parse_iso(timed[0]["timestamp_utc"])

    stage_names = [
        "run_start",
        "pipeline_init_start",
        "stage_geometry_built",
        "stage_bdf_written",
        "stage_tacs_initialized",
        "stage_transfer_initialized",
        "stage_aero_initialized",
        "pipeline_init_complete",
        "optimization_start",
        "optimization_complete",
    ]
    stage_y = {name: i for i, name in enumerate(stage_names)}
    stage_t = {name: [] for name in stage_names}

    coupl = []
    eval_complete = []
    for e in events:
        ts = e.get("timestamp_utc")
        if not isinstance(ts, str):
            continue
        dt = (_parse_iso(ts) - t0).total_seconds()
        name = e.get("event", "")
        if name in stage_t:
            stage_t[name].append(dt)
        if name == "coupling_iteration":
            it = e["iteration"]
            coupl.append(
                {
                    "t": dt,
                    "step": len(coupl) + 1,
                    "eval": int(e["evaluation_id"]),
                    "iter": int(it["iteration"]),
                    "cl": float(it["cl"]),
                    "cd": float(it["cd"]),
                    "load_rel": float(it["load_rel_change"]),
                    "disp_rel": float(it["disp_rel_change"]),
                    "handoff": bool(it["handoff_finite"]),
                }
            )
        if name == "evaluation_complete":
            eval_complete.append(
                {
                    "t": dt,
                    "eval": int(e["evaluation_id"]),
                    "obj": float(e["objective_cd"]),
                    "conv": bool(e["converged"]),
                }
            )

    if not coupl:
        raise RuntimeError("No coupling_iteration events found")

    x_step = np.array([c["step"] for c in coupl])
    cl = np.array([c["cl"] for c in coupl])
    cd = np.array([c["cd"] for c in coupl])
    load_rel = np.array([c["load_rel"] for c in coupl])
    disp_rel = np.array([c["disp_rel"] for c in coupl])
    handoff = np.array([1.0 if c["handoff"] else 0.0 for c in coupl])

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Full MDO Workflow Proof (Start -> Finish)", fontsize=16, fontweight="bold")

    # Panel 1: stage timeline
    ax = axs[0, 0]
    for name, ys in stage_y.items():
        xs = stage_t.get(name, [])
        if xs:
            ax.scatter(xs, [ys] * len(xs), s=40, label=name)
    ax.set_yticks(list(stage_y.values()))
    ax.set_yticklabels(list(stage_y.keys()))
    ax.set_xlabel("Time Since Start (s)")
    ax.set_title("Stage Timeline")
    ax.grid(True, alpha=0.3)

    # Panel 2: CL/CD across coupling steps
    ax = axs[0, 1]
    ax.plot(x_step, cl, marker="o", label="CL")
    ax.plot(x_step, cd, marker="s", label="CD")
    ax.set_title("Coupling Steps: Aerodynamic Response")
    ax.set_xlabel("Global Coupling Step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 3: convergence + handoff
    ax = axs[1, 0]
    ax.plot(x_step, load_rel, marker="o", label="load_rel_change")
    ax.plot(x_step, disp_rel, marker="s", label="disp_rel_change")
    ax.step(x_step, handoff, where="mid", label="handoff_finite (1=OK)")
    ax.set_yscale("log")
    ax.set_title("Convergence + Handoff Health")
    ax.set_xlabel("Global Coupling Step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    # Panel 4: objective by evaluation
    ax = axs[1, 1]
    if eval_complete:
        ev = np.array([e["eval"] for e in eval_complete])
        obj = np.array([e["obj"] for e in eval_complete])
        conv = np.array([1.0 if e["conv"] else 0.0 for e in eval_complete])
        ax.plot(ev, obj, marker="o", label="objective_cd (penalized)")
        ax.step(ev, conv, where="mid", label="evaluation_converged (1=yes)")
    else:
        ax.text(0.1, 0.5, "No evaluation_complete events", transform=ax.transAxes)
    ax.set_title("Optimization Progress")
    ax.set_xlabel("Evaluation ID")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    opt = json.loads(optimum_file.read_text(encoding="utf-8"))
    summary = {
        "progress_file": str(progress_file),
        "optimum_file": str(optimum_file),
        "proof_plot": str(out_png),
        "total_events": len(events),
        "coupling_steps": len(coupl),
        "evaluations": len(eval_complete),
        "all_handoff_finite": bool(np.all(handoff > 0.5)),
        "final_optimizer": opt.get("optimizer"),
        "final_outputs": opt.get("outputs", {}),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate full-workflow matplotlib proof")
    parser.add_argument("--config", default="configs/real_physics_pipeline.proof.json")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--progress-file",
        default="results/real_physics/live/full_workflow_proof_progress.ndjson",
    )
    parser.add_argument(
        "--out-plot",
        default="results/real_physics/proof_graphs/full_workflow_proof.png",
    )
    parser.add_argument(
        "--out-summary",
        default="results/real_physics/proof_graphs/full_workflow_proof_summary.json",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open generated proof image using default OS image viewer",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_rel = args.config
    progress_rel = args.progress_file
    progress_file = (repo_root / progress_rel).resolve()
    out_plot = (repo_root / args.out_plot).resolve()
    out_summary = (repo_root / args.out_summary).resolve()

    progress_file.parent.mkdir(parents=True, exist_ok=True)
    if progress_file.exists():
        progress_file.unlink()

    _run_workflow(repo_root, config_rel, progress_rel, args.cuda_visible_devices)

    optimum_file = (repo_root / "results" / "real_physics" / "real_physics_proof_optimum.json").resolve()
    summary = _build_proof_figure(progress_file, optimum_file, out_plot, out_summary)

    print("Saved proof plot:", out_plot)
    print("Saved summary:", out_summary)
    print(json.dumps(summary, indent=2))

    if args.open:
        try:
            subprocess.Popen(["powershell", "-Command", f"Start-Process '{out_plot}'"])
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

