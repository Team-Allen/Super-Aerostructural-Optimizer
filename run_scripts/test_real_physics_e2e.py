"""End-to-end validation for the real-physics aerostructural pipeline.

This validates:
1) Coupled analysis loop executes end-to-end
2) Aero<->structure transfer handoff remains finite
3) Optimization loop executes and returns finite outputs
4) CUDA availability is reported (where applicable)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Super_Aerostructural_Optimizer.pipeline.config import (  # noqa: E402
    PipelineConfig,
    load_pipeline_config,
)
from Super_Aerostructural_Optimizer.pipeline.openmdao_opt import (  # noqa: E402
    run_openmdao_optimization,
)
from Super_Aerostructural_Optimizer.pipeline.workflow import (  # noqa: E402
    AerostructuralMDOPipeline,
)


def _scalar(value) -> float:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0])


def detect_cuda() -> Dict[str, object]:
    """Detect CUDA availability across common Python backends."""
    report: Dict[str, object] = {
        "env_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "nvidia_smi": {"available": False, "gpus": []},
        "torch": {"installed": False, "cuda_available": False, "device_count": 0},
        "jax": {"installed": False, "gpu_devices": 0, "devices": []},
        "cupy": {"installed": False, "cuda_available": False, "device_count": 0},
    }

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            gpus = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            report["nvidia_smi"] = {"available": True, "gpus": gpus}
    except Exception:
        pass

    if importlib.util.find_spec("torch") is not None:
        import torch  # type: ignore

        report["torch"] = {
            "installed": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }

    if importlib.util.find_spec("jax") is not None:
        import jax  # type: ignore

        gpu_devs = [str(d) for d in jax.devices() if d.platform == "gpu"]
        report["jax"] = {
            "installed": True,
            "gpu_devices": int(len(gpu_devs)),
            "devices": [str(d) for d in jax.devices()],
        }

    if importlib.util.find_spec("cupy") is not None:
        import cupy  # type: ignore

        try:
            count = int(cupy.cuda.runtime.getDeviceCount())
            ok = count > 0
        except Exception:
            count = 0
            ok = False
        report["cupy"] = {
            "installed": True,
            "cuda_available": ok,
            "device_count": count,
        }

    return report


def validate_analysis_result(result: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Validate coupled analysis report."""
    errors: List[str] = []

    iterations = result.get("iterations", [])
    if not isinstance(iterations, list) or len(iterations) == 0:
        errors.append("No coupling iterations were recorded")
        return False, errors

    for i, row in enumerate(iterations, start=1):
        for key in [
            "cl",
            "cd",
            "mass_kg",
            "ks_failure",
            "tip_deflection_m",
            "aero_load_norm",
            "struct_load_norm",
            "struct_disp_norm",
            "aero_disp_norm",
            "load_rel_change",
            "disp_rel_change",
        ]:
            if not np.isfinite(_scalar(row[key])):
                errors.append(f"Iteration {i}: non-finite value for '{key}'")
        if not bool(row.get("handoff_finite", False)):
            errors.append(f"Iteration {i}: handoff_finite is False")

    for key in ["objective_cd", "objective_cd_raw"]:
        if not np.isfinite(_scalar(result[key])):
            errors.append(f"Non-finite top-level metric '{key}'")

    constraints = result.get("constraints", {})
    for key in [
        "ks_failure_minus_1",
        "tip_deflection_minus_limit_m",
        "cl_minus_target",
    ]:
        if not np.isfinite(_scalar(constraints[key])):
            errors.append(f"Non-finite constraint '{key}'")

    return len(errors) == 0, errors


def validate_optimization_result(solution: Dict[str, object]) -> Tuple[bool, List[str]]:
    """Validate optimizer output report."""
    errors: List[str] = []
    outputs = solution.get("outputs", {})
    for key in [
        "objective_cd",
        "objective_cd_raw",
        "cl",
        "cd",
        "l_over_d",
        "mass_kg",
        "alpha_deg",
        "ks_failure_minus_1",
        "tip_deflection_minus_limit_m",
        "cl_minus_target",
    ]:
        if not np.isfinite(_scalar(outputs[key])):
            errors.append(f"Non-finite optimization output '{key}'")

    if int(solution.get("evaluation_count", 0)) < 1:
        errors.append("Optimization evaluation_count < 1")

    return len(errors) == 0, errors


def main() -> int:
    parser = argparse.ArgumentParser(description="E2E test for real-physics aerostructural pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/real_physics_pipeline.json",
        help="Pipeline config path",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/real_physics/e2e_test",
        help="Directory where test reports are written",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default="real_physics_e2e",
        help="Test case name prefix",
    )
    parser.add_argument(
        "--coupling-iters",
        type=int,
        default=8,
        help="Coupling iterations used for the test",
    )
    parser.add_argument(
        "--optimizer-iters",
        type=int,
        default=3,
        help="Optimizer iterations used for the test",
    )
    parser.add_argument(
        "--skip-optimize",
        action="store_true",
        help="Run only coupled analysis test",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail test if no CUDA backend is available",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg: PipelineConfig = load_pipeline_config(cfg_path)
    cfg.output.results_dir = str((PROJECT_ROOT / args.results_dir).resolve())
    cfg.output.case_name = args.case_name
    cfg.output.save_optimizer_db = False
    cfg.output.write_tacs_solution = False
    cfg.coupling.max_iterations = int(args.coupling_iters)
    cfg.optimizer.max_iterations = int(args.optimizer_iters)
    cfg.validate()

    cuda_report = detect_cuda()
    cuda_python_backend_available = (
        bool(cuda_report["torch"]["cuda_available"])
        or int(cuda_report["jax"]["gpu_devices"]) > 0
        or bool(cuda_report["cupy"]["cuda_available"])
    )
    cuda_hw_available = bool(cuda_report["nvidia_smi"]["available"]) and (
        len(cuda_report["nvidia_smi"]["gpus"]) > 0
    )

    started = time.time()
    pipeline = AerostructuralMDOPipeline(cfg)

    analysis = pipeline.evaluate(pipeline.default_design())
    analysis_ok, analysis_errors = validate_analysis_result(analysis)

    optimize_ok = True
    optimize_errors: List[str] = []
    optimize = None
    if not args.skip_optimize:
        optimize = run_openmdao_optimization(pipeline, record_path=None)
        optimize_ok, optimize_errors = validate_optimization_result(optimize)

    cuda_ok = (not args.require_cuda) or cuda_hw_available
    cuda_errors = [] if cuda_ok else ["CUDA is required but no NVIDIA GPU was detected by nvidia-smi"]
    cuda_warnings: List[str] = []
    if cuda_hw_available and not cuda_python_backend_available:
        cuda_warnings.append(
            "GPU detected, but no CUDA-enabled Python backend (torch/jax/cupy) is installed in this env."
        )

    overall_ok = analysis_ok and optimize_ok and cuda_ok
    ended = time.time()

    report = {
        "ok": overall_ok,
        "analysis_ok": analysis_ok,
        "optimization_ok": optimize_ok,
        "cuda_ok": cuda_ok,
        "cuda_required": bool(args.require_cuda),
        "cuda_hw_available": cuda_hw_available,
        "cuda_python_backend_available": cuda_python_backend_available,
        "cuda_report": cuda_report,
        "analysis_errors": analysis_errors,
        "optimization_errors": optimize_errors,
        "cuda_errors": cuda_errors,
        "cuda_warnings": cuda_warnings,
        "analysis_summary": {
            "converged": bool(analysis["converged"]),
            "iterations": len(analysis["iterations"]),
            "objective_cd": _scalar(analysis["objective_cd"]),
            "cl": _scalar(analysis["aero"]["cl"]),
            "cd": _scalar(analysis["aero"]["cd"]),
            "l_over_d": _scalar(analysis["aero"]["l_over_d"]),
            "mass_kg": _scalar(analysis["structure"]["mass_kg"]),
            "ks_failure": _scalar(analysis["structure"]["ks_failure"]),
            "tip_deflection_m": _scalar(analysis["structure"]["tip_deflection_m"]),
        },
        "optimization_summary": optimize["outputs"] if optimize is not None else None,
        "elapsed_seconds": ended - started,
    }

    out_dir = Path(cfg.output.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.case_name}_e2e_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"E2E report: {out_path}")
    print(json.dumps(report, indent=2))

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
