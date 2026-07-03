"""Create an animated proof (GIF/MP4) for the full MDO workflow.

This script can run the workflow end-to-end, parse the progress NDJSON, and
generate a time-evolving animation showing stage execution and metric handoff.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import rcParams
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


def _load_events(progress_file: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for line in progress_file.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw))
        except Exception:
            continue
    if not events:
        raise RuntimeError(f"No events found in {progress_file}")
    return events


def _extract_series(events: List[Dict[str, object]]) -> Dict[str, object]:
    timed = [e for e in events if isinstance(e.get("timestamp_utc"), str)]
    if not timed:
        raise RuntimeError("No timestamped events found in progress log")

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

    event_time: List[float] = []
    event_name: List[str] = []
    stage_points: List[Tuple[int, float, int]] = []
    coupling: List[Dict[str, float]] = []
    evals: List[Dict[str, float]] = []

    for idx, e in enumerate(events):
        ts = e.get("timestamp_utc")
        if not isinstance(ts, str):
            continue
        t = (_parse_iso(ts) - t0).total_seconds()
        name = str(e.get("event", ""))
        event_time.append(t)
        event_name.append(name)

        if name in stage_y:
            stage_points.append((idx, t, stage_y[name]))

        if name == "coupling_iteration":
            it = e.get("iteration", {})
            coupling.append(
                {
                    "event_index": float(idx),
                    "t": float(t),
                    "step": float(len(coupling) + 1),
                    "cl": float(it.get("cl", 0.0)),
                    "cd": float(it.get("cd", 0.0)),
                    "load_rel": float(it.get("load_rel_change", 0.0)),
                    "disp_rel": float(it.get("disp_rel_change", 0.0)),
                    "handoff": 1.0 if bool(it.get("handoff_finite", False)) else 0.0,
                }
            )
        elif name == "evaluation_complete":
            evals.append(
                {
                    "event_index": float(idx),
                    "eval": float(e.get("evaluation_id", len(evals) + 1)),
                    "objective_cd": float(e.get("objective_cd", np.nan)),
                    "converged": 1.0 if bool(e.get("converged", False)) else 0.0,
                }
            )

    if not coupling:
        raise RuntimeError("No coupling_iteration events found")

    return {
        "stage_names": stage_names,
        "stage_y": stage_y,
        "stage_points": stage_points,
        "event_time": event_time,
        "event_name": event_name,
        "coupling": coupling,
        "evals": evals,
        "total_events": len(events),
    }


def _animate(
    series: Dict[str, object],
    out_gif: Path,
    out_mp4: Path | None,
    fps: int,
) -> Dict[str, object]:
    stage_names = series["stage_names"]
    stage_points = series["stage_points"]
    event_name = series["event_name"]
    coupling = series["coupling"]
    evals = series["evals"]
    total_events = int(series["total_events"])

    x_c = np.array([float(c["step"]) for c in coupling], dtype=float)
    cl = np.array([float(c["cl"]) for c in coupling], dtype=float)
    cd = np.array([float(c["cd"]) for c in coupling], dtype=float)
    load_rel = np.array([max(float(c["load_rel"]), 1.0e-12) for c in coupling], dtype=float)
    disp_rel = np.array([max(float(c["disp_rel"]), 1.0e-12) for c in coupling], dtype=float)
    handoff = np.array([float(c["handoff"]) for c in coupling], dtype=float)
    c_event_idx = np.array([int(c["event_index"]) for c in coupling], dtype=int)

    if evals:
        x_e = np.array([float(e["eval"]) for e in evals], dtype=float)
        obj = np.array([float(e["objective_cd"]) for e in evals], dtype=float)
        conv = np.array([float(e["converged"]) for e in evals], dtype=float)
        e_event_idx = np.array([int(e["event_index"]) for e in evals], dtype=int)
    else:
        x_e = np.array([], dtype=float)
        obj = np.array([], dtype=float)
        conv = np.array([], dtype=float)
        e_event_idx = np.array([], dtype=int)

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle("Animated Full MDO Workflow Proof (Start -> Finish)", fontsize=16, fontweight="bold")

    # Panel 1: stage timeline
    ax0 = axs[0, 0]
    ax0.set_yticks(range(len(stage_names)))
    ax0.set_yticklabels(stage_names)
    ax0.set_xlabel("Time Since Start (s)")
    ax0.set_title("Stage Timeline")
    ax0.grid(True, alpha=0.3)
    stage_scatter = ax0.scatter([], [], s=45, c="tab:blue")
    current_marker = ax0.scatter([], [], s=90, c="tab:red", marker="x")
    time_label = ax0.text(0.02, 0.02, "", transform=ax0.transAxes)

    # Panel 2: CL/CD
    ax1 = axs[0, 1]
    ax1.set_title("Coupling Steps: Aerodynamic Response")
    ax1.set_xlabel("Global Coupling Step")
    ax1.grid(True, alpha=0.3)
    line_cl, = ax1.plot([], [], marker="o", label="CL")
    line_cd, = ax1.plot([], [], marker="s", label="CD")
    ax1.legend()

    # Panel 3: convergence + handoff
    ax2 = axs[1, 0]
    ax2.set_title("Convergence + Handoff Health")
    ax2.set_xlabel("Global Coupling Step")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", alpha=0.3)
    line_load, = ax2.plot([], [], marker="o", label="load_rel_change")
    line_disp, = ax2.plot([], [], marker="s", label="disp_rel_change")
    line_handoff, = ax2.plot([], [], marker="^", label="handoff_finite (1=OK)")
    ax2.legend()

    # Panel 4: objective by evaluation
    ax3 = axs[1, 1]
    ax3.set_title("Optimization Progress")
    ax3.set_xlabel("Evaluation ID")
    ax3.grid(True, alpha=0.3)
    line_obj, = ax3.plot([], [], marker="o", label="objective_cd (penalized)")
    line_conv, = ax3.plot([], [], marker="s", label="evaluation_converged (1=yes)")
    ax3.legend()

    # Set static limits
    if len(coupling) > 0:
        ax1.set_xlim(1.0, max(2.0, float(np.max(x_c))))
        ymin = min(float(np.min(cd)), float(np.min(cl)))
        ymax = max(float(np.max(cd)), float(np.max(cl)))
        pad = 0.05 * max(1e-6, ymax - ymin)
        ax1.set_ylim(ymin - pad, ymax + pad)

        ax2.set_xlim(1.0, max(2.0, float(np.max(x_c))))
        lower = min(float(np.min(load_rel)), float(np.min(disp_rel)), 0.8)
        upper = max(float(np.max(load_rel)), float(np.max(disp_rel)), 1.2)
        ax2.set_ylim(max(1e-8, lower * 0.8), upper * 1.2)

    if len(evals) > 0:
        ax3.set_xlim(1.0, max(2.0, float(np.max(x_e))))
        y3min = float(np.min(obj))
        y3max = float(np.max(obj))
        pad3 = 0.08 * max(1e-6, y3max - y3min)
        ax3.set_ylim(y3min - pad3, y3max + pad3)

    if stage_points:
        max_t = max(t for _, t, _ in stage_points)
        ax0.set_xlim(-0.02 * max_t, max_t * 1.05 if max_t > 0 else 1.0)

    def _update(frame: int):
        # Stage updates
        pts = [(t, y) for i, t, y in stage_points if i <= frame]
        if pts:
            stage_scatter.set_offsets(np.array(pts, dtype=float))
        else:
            stage_scatter.set_offsets(np.empty((0, 2)))

        # Current event marker
        curr_stage = [(t, y) for i, t, y in stage_points if i == frame]
        if curr_stage:
            current_marker.set_offsets(np.array(curr_stage, dtype=float))
        else:
            current_marker.set_offsets(np.empty((0, 2)))
        name = event_name[frame] if frame < len(event_name) else "?"
        time_label.set_text(f"event {frame + 1}/{total_events}: {name}")

        # Coupling traces up to current frame
        mask_c = c_event_idx <= frame
        if np.any(mask_c):
            line_cl.set_data(x_c[mask_c], cl[mask_c])
            line_cd.set_data(x_c[mask_c], cd[mask_c])
            line_load.set_data(x_c[mask_c], load_rel[mask_c])
            line_disp.set_data(x_c[mask_c], disp_rel[mask_c])
            line_handoff.set_data(x_c[mask_c], np.maximum(handoff[mask_c], 1e-8))
        else:
            line_cl.set_data([], [])
            line_cd.set_data([], [])
            line_load.set_data([], [])
            line_disp.set_data([], [])
            line_handoff.set_data([], [])

        # Objective traces up to current frame
        if len(evals) > 0:
            mask_e = e_event_idx <= frame
            if np.any(mask_e):
                line_obj.set_data(x_e[mask_e], obj[mask_e])
                line_conv.set_data(x_e[mask_e], np.maximum(conv[mask_e], 1e-8))
            else:
                line_obj.set_data([], [])
                line_conv.set_data([], [])

        return (
            stage_scatter,
            current_marker,
            line_cl,
            line_cd,
            line_load,
            line_disp,
            line_handoff,
            line_obj,
            line_conv,
            time_label,
        )

    anim = FuncAnimation(
        fig,
        _update,
        frames=total_events,
        interval=max(50, int(1000 / max(1, fps))),
        blit=False,
        repeat=False,
    )

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=140)

    mp4_saved = None
    mp4_error = None
    if out_mp4 is not None:
        try:
            from matplotlib.animation import FFMpegWriter

            # Fallback to packaged ffmpeg binary if system ffmpeg is not installed.
            try:
                import imageio_ffmpeg

                rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                pass

            writer = FFMpegWriter(fps=fps, bitrate=1800)
            anim.save(out_mp4, writer=writer, dpi=140)
            mp4_saved = str(out_mp4)
        except Exception as exc:
            mp4_error = str(exc)

    plt.close(fig)
    return {"gif": str(out_gif), "mp4": mp4_saved, "mp4_error": mp4_error}


def main() -> int:
    parser = argparse.ArgumentParser(description="Animated full-workflow proof generator")
    parser.add_argument("--config", default="configs/real_physics_pipeline.proof.json")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument(
        "--progress-file",
        default="results/real_physics/live/full_workflow_animation_progress.ndjson",
    )
    parser.add_argument(
        "--out-gif",
        default="results/real_physics/proof_graphs/full_workflow_animation.gif",
    )
    parser.add_argument(
        "--out-mp4",
        default="results/real_physics/proof_graphs/full_workflow_animation.mp4",
    )
    parser.add_argument(
        "--out-summary",
        default="results/real_physics/proof_graphs/full_workflow_animation_summary.json",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not run workflow, only animate existing progress file",
    )
    parser.add_argument("--no-mp4", action="store_true", help="Generate GIF only")
    parser.add_argument("--open", action="store_true", help="Open generated GIF when done")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    progress_rel = args.progress_file
    progress_file = (repo_root / progress_rel).resolve()
    out_gif = (repo_root / args.out_gif).resolve()
    out_summary = (repo_root / args.out_summary).resolve()
    out_mp4 = None if args.no_mp4 else (repo_root / args.out_mp4).resolve()

    if not args.skip_run:
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        if progress_file.exists():
            progress_file.unlink()
        _run_workflow(repo_root, args.config, progress_rel, args.cuda_visible_devices)

    events = _load_events(progress_file)
    series = _extract_series(events)
    saved = _animate(series, out_gif, out_mp4, args.fps)

    summary = {
        "progress_file": str(progress_file),
        "total_events": int(series["total_events"]),
        "coupling_steps": len(series["coupling"]),
        "evaluations": len(series["evals"]),
        "outputs": saved,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved GIF:", out_gif)
    if saved["mp4"] is not None:
        print("Saved MP4:", saved["mp4"])
    elif saved["mp4_error"]:
        print("MP4 skipped:", saved["mp4_error"])
    print("Saved summary:", out_summary)
    print(json.dumps(summary, indent=2))

    if args.open:
        try:
            subprocess.Popen(["powershell", "-Command", f"Start-Process '{out_gif}'"])
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
