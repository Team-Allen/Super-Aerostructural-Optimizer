#!/usr/bin/env python
"""
Generate a structural DOE for the VAM/MYSTRAN structural surrogate.

This script builds a parametric composite wing-box model, varies ply counts,
ply angles, thicknesses, and mechanical loads, runs each case through MYSTRAN,
and writes a CSV dataset for surrogate training.

Example::

    python scripts/generate_structural_doe.py \
        --n_samples 20 \
        --mystran_exe "Reference Docs/MYSTRAN/Masti/MYSTRAN.exe" \
        --output results/structural_doe/doe.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

# Add project root to path when running as a script
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piml_mdo.structures.composite_properties import CFRP_IM7_8552
from piml_mdo.structures.mystran_runner import build_wingbox_bdf, MystranRunner

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Latin-hypercube style sampling over the structural design space
# ---------------------------------------------------------------------------


def sample_laminate(rng: np.random.Generator) -> tuple[dict[float, int], float]:
    """Return angle-count dict and ply thickness for one sample."""
    angles = [0.0, 45.0, -45.0, 90.0]
    # Sample total ply count and allocate across angles
    total = rng.integers(8, 40)
    # Bias toward more 0 and +/-45 plies than 90 plies
    weights = np.array([0.35, 0.25, 0.25, 0.15])
    counts = rng.multinomial(total, weights / weights.sum())
    counts = dict(zip(angles, counts))
    ply_thickness = float(rng.uniform(0.000125, 0.00025))
    return counts, ply_thickness


def generate_samples(n_samples: int, seed: int = 42) -> list[dict]:
    """Generate a list of sample specifications."""
    rng = np.random.default_rng(seed)
    samples = []
    for i in range(n_samples):
        counts, ply_thickness = sample_laminate(rng)
        pressure = float(rng.uniform(-2e3, -2e2))  # suction pressure on upper skin [Pa]
        tip_force_z = float(rng.uniform(-5e3, -5e2))  # tip vertical force [N]
        tip_moment_y = float(rng.uniform(-2e3, 2e3))  # tip torque [N·m]
        samples.append(
            {
                "sample_id": i,
                "counts": counts,
                "ply_thickness": ply_thickness,
                "pressure": pressure,
                "tip_force_z": tip_force_z,
                "tip_moment_y": tip_moment_y,
            }
        )
    return samples


# ---------------------------------------------------------------------------
# MYSTRAN execution wrapper for one sample
# ---------------------------------------------------------------------------


def run_one_sample(
    sample: dict,
    workdir: Path,
    mystran_exe: Path | None,
    span: float,
    width: float,
    height: float,
    n_span: int,
    n_perim: int,
) -> dict:
    """Run MYSTRAN for a single sample and return feature/output dict."""
    sample_dir = workdir / f"sample_{sample['sample_id']:04d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    bdf_path = sample_dir / "wingbox.bdf"
    build_wingbox_bdf(
        bdf_path=bdf_path,
        span=span,
        width=width,
        height=height,
        n_span=n_span,
        n_perim=n_perim,
        pressure=sample["pressure"],
        tip_force=(0.0, 0.0, sample["tip_force_z"]),
        tip_moment=(0.0, sample["tip_moment_y"], 0.0),
    )

    runner = MystranRunner(
        bdf_path=bdf_path,
        workdir=sample_dir,
        mystran_exe=mystran_exe,
    )
    runner.set_laminate_from_counts(
        pid=1,
        ply_counts=sample["counts"],
        ply_thickness=sample["ply_thickness"],
    )

    result = runner.run(cleanup=True, timeout=600)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate structural DOE for surrogate training")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of DOE samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--span", type=float, default=1.0, help="Wing-box span [m]")
    parser.add_argument("--width", type=float, default=0.3, help="Wing-box width [m]")
    parser.add_argument("--height", type=float, default=0.05, help="Wing-box height [m]")
    parser.add_argument("--n_span", type=int, default=10, help="Spanwise mesh divisions")
    parser.add_argument("--n_perim", type=int, default=8, help="Perimeter mesh divisions")
    parser.add_argument("--mystran_exe", type=str, default=None, help="Path to MYSTRAN executable")
    parser.add_argument("--output", type=str, default="results/structural_doe/doe.csv", help="Output CSV path")
    args = parser.parse_args()

    workdir = Path(args.output).parent / "mystran_runs"
    workdir.mkdir(parents=True, exist_ok=True)
    mystran_exe = Path(args.mystran_exe) if args.mystran_exe else None

    samples = generate_samples(args.n_samples, args.seed)

    feature_rows = []
    output_rows = []

    start = time.time()
    for sample in samples:
        logger.info(
            f"Sample {sample['sample_id']}: counts={sample['counts']}, "
            f"t_ply={sample['ply_thickness']:.6f}, p={sample['pressure']:.1f}, "
            f"Fz={sample['tip_force_z']:.1f}, My={sample['tip_moment_y']:.1f}"
        )
        try:
            result = run_one_sample(
                sample,
                workdir,
                mystran_exe,
                args.span,
                args.width,
                args.height,
                args.n_span,
                args.n_perim,
            )
        except Exception as exc:
            logger.error(f"Sample {sample['sample_id']} failed: {exc}")
            continue

        upper_area = args.span * args.width
        total_force = sample["pressure"] * upper_area + sample["tip_force_z"]
        root_moment = total_force * args.span / 2.0 + sample["tip_moment_y"]
        root_torque = abs(sample["tip_moment_y"])

        features = [
            sample["counts"].get(0.0, 0),
            sample["counts"].get(45.0, 0),
            sample["counts"].get(-45.0, 0),
            sample["counts"].get(90.0, 0),
            sample["ply_thickness"],
            total_force,
            root_moment,
            root_torque,
        ]
        outputs = [
            result["max_vertical_displacement"],
            result["max_von_mises"],
            result["max_failure_index"],
            result["total_mass"],
        ]
        feature_rows.append(features)
        output_rows.append(outputs)

    elapsed = time.time() - start
    logger.info(f"Completed {len(feature_rows)} of {args.n_samples} samples in {elapsed:.1f}s")

    if not feature_rows:
        logger.error("No successful samples; aborting")
        return

    feature_names = ["n0", "n45", "n45_neg", "n90", "ply_thickness", "total_force", "root_moment", "root_torque"]
    output_names = ["max_vertical_displacement", "max_von_mises", "max_failure_index", "total_mass"]

    import pandas as pd

    df = pd.DataFrame(feature_rows, columns=[f"feat_{n}" for n in feature_names])
    for name, col in zip(output_names, np.asarray(output_rows).T):
        df[f"out_{name}"] = col

    meta = {
        "feature_names": [f"feat_{n}" for n in feature_names],
        "output_names": [f"out_{n}" for n in output_names],
        "material": CFRP_IM7_8552.name,
        "geometry": {"span": args.span, "width": args.width, "height": args.height},
        "n_samples_requested": args.n_samples,
        "n_samples_success": len(feature_rows),
        "wall_time_s": elapsed,
    }
    df.attrs["meta"] = json.dumps(meta)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Wrote DOE dataset to {out_path}")

    # Also write a metadata sidecar for non-pandas consumers
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
