"""Run the real-physics aerostructural MDO pipeline.

Usage (WSL mdo-best):
  python run_scripts/run_real_physics_mdo.py --mode analyze
  python run_scripts/run_real_physics_mdo.py --mode optimize
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Super_Aerostructural_Optimizer.pipeline.config import (  # noqa: E402
    PipelineConfig,
    load_pipeline_config,
    save_pipeline_config,
)
from Super_Aerostructural_Optimizer.pipeline.openmdao_opt import (  # noqa: E402
    run_openmdao_optimization,
)
from Super_Aerostructural_Optimizer.pipeline.workflow import (  # noqa: E402
    AerostructuralMDOPipeline,
)


def _ensure_config(path: Path) -> PipelineConfig:
    if path.exists():
        return load_pipeline_config(path)
    cfg = PipelineConfig()
    save_pipeline_config(cfg, path)
    return cfg


def _make_progress_writer(progress_file: Optional[Path]) -> Optional[Callable[[Dict[str, object]], None]]:
    if progress_file is None:
        return None
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    if progress_file.exists():
        progress_file.unlink()

    def _write(event: Dict[str, object]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        with progress_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    return _write


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-physics aerostructural MDO pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/real_physics_pipeline.json",
        help="Path to pipeline JSON configuration",
    )
    parser.add_argument(
        "--mode",
        choices=["analyze", "optimize"],
        default="analyze",
        help="analyze: single coupled solve, optimize: pyOptSparse optimization",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Optional override for output.results_dir",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help="Optional override for output.case_name",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="Optional NDJSON path for live progress events",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = _ensure_config(config_path)

    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
        if not results_dir.is_absolute():
            results_dir = (PROJECT_ROOT / results_dir).resolve()
        config.output.results_dir = str(results_dir)
    if args.case_name is not None:
        config.output.case_name = args.case_name
    config.validate()

    progress_file = None
    if args.progress_file is not None:
        progress_file = Path(args.progress_file)
        if not progress_file.is_absolute():
            progress_file = (PROJECT_ROOT / progress_file).resolve()
    progress_writer = _make_progress_writer(progress_file)

    try:
        pipeline = AerostructuralMDOPipeline(config, event_callback=progress_writer)
        pipeline.iteration_callback = progress_writer
        case_name = config.output.case_name
        if progress_writer is not None:
            progress_writer(
                {
                    "event": "run_start",
                    "mode": args.mode,
                    "case_name": case_name,
                    "config_path": str(config_path),
                    "results_dir": str(pipeline.results_dir),
                }
            )

        if args.mode == "analyze":
            result = pipeline.evaluate(pipeline.default_design())
            out_file = pipeline.save_result(result, f"{case_name}_analysis.json")
            if config.output.write_tacs_solution:
                pipeline.structure.write_solution(pipeline.results_dir / "tacs_solution", case_name)
            if progress_writer is not None:
                progress_writer(
                    {
                        "event": "analysis_complete",
                        "output_file": str(out_file),
                        "result": {
                            "converged": result["converged"],
                            "objective_cd": result["objective_cd"],
                            "cl": result["aero"]["cl"],
                            "cd": result["aero"]["cd"],
                            "l_over_d": result["aero"]["l_over_d"],
                            "mass_kg": result["structure"]["mass_kg"],
                            "ks_failure": result["structure"]["ks_failure"],
                            "tip_deflection_m": result["structure"]["tip_deflection_m"],
                        },
                    }
                )
            print(f"analysis saved: {out_file}")
            print(
                "summary:",
                json.dumps(
                    {
                        "converged": result["converged"],
                        "cl": result["aero"]["cl"],
                        "cd": result["aero"]["cd"],
                        "l_over_d": result["aero"]["l_over_d"],
                        "mass_kg": result["structure"]["mass_kg"],
                        "ks_failure": result["structure"]["ks_failure"],
                        "tip_deflection_m": result["structure"]["tip_deflection_m"],
                    },
                    indent=2,
                ),
            )
            return 0

        record_path = None
        if config.output.save_optimizer_db:
            record_path = pipeline.results_dir / f"{case_name}_opt_cases.sql"
        if progress_writer is not None:
            progress_writer({"event": "optimization_start"})

        solution = run_openmdao_optimization(pipeline, record_path=record_path)
        out_file = pipeline.save_result(solution, f"{case_name}_optimum.json")

        if config.output.write_tacs_solution and solution.get("last_evaluation") is not None:
            pipeline.structure.write_solution(pipeline.results_dir / "tacs_solution", case_name)
        if progress_writer is not None:
            progress_writer(
                {
                    "event": "optimization_complete",
                    "output_file": str(out_file),
                    "summary": solution["outputs"],
                    "optimizer": solution["optimizer"],
                }
            )

        print(f"optimization saved: {out_file}")
        print(
            "summary:",
            json.dumps(
                {
                    "optimizer": solution["optimizer"],
                    "objective_cd": solution["outputs"]["objective_cd"],
                    "cl": solution["outputs"]["cl"],
                    "cd": solution["outputs"]["cd"],
                    "l_over_d": solution["outputs"]["l_over_d"],
                    "mass_kg": solution["outputs"]["mass_kg"],
                },
                indent=2,
            ),
        )
        return 0
    except Exception as exc:
        if progress_writer is not None:
            progress_writer(
                {
                    "event": "run_error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
