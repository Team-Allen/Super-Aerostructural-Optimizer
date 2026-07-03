"""Show proof artifacts in a single matplotlib dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Display proof graphs with matplotlib")
    parser.add_argument(
        "--proof-dir",
        type=str,
        default="results/real_physics/proof_graphs",
        help="Directory containing proof graph PNGs",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="results/real_physics/proof_graphs/proof_dashboard_matplotlib.png",
        help="Output path for combined dashboard image",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Only save the dashboard image without opening a matplotlib window",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    proof_dir = Path(args.proof_dir)
    if not proof_dir.is_absolute():
        proof_dir = (repo_root / proof_dir).resolve()

    required = [
        "proof_coupling_handoff_metrics.png",
        "proof_eval_summary.png",
        "proof_cuda_backend_status.png",
        "proof_opt_outputs.png",
    ]
    missing = [name for name in required if not (proof_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing proof images: "
            + ", ".join(missing)
            + f". Expected under: {proof_dir}"
        )

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MDO Toolchain Proof Dashboard (Matplotlib)", fontsize=16, fontweight="bold")

    for ax, name in zip(axs.flatten(), required):
        img = mpimg.imread(proof_dir / name)
        ax.imshow(img)
        ax.set_title(name.replace(".png", ""))
        ax.axis("off")

    summary_path = proof_dir / "proof_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            note = (
                f"Evaluations: {summary.get('num_evaluations')} | "
                f"Coupling events: {summary.get('num_coupling_events')} | "
                f"All handoff finite: {summary.get('all_handoff_finite')}"
            )
            fig.text(0.01, 0.01, note, fontsize=10)
        except Exception:
            pass

    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    save_path = Path(args.save)
    if not save_path.is_absolute():
        save_path = (repo_root / save_path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=180)
    print(f"Saved dashboard: {save_path}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

