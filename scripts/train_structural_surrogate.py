#!/usr/bin/env python
"""
Train the structural neural surrogate from a MYSTRAN DOE dataset.

Example::

    python scripts/train_structural_surrogate.py \
        --doe results/structural_doe/doe.csv \
        --output results/structural_doe/surrogate.pt \
        --epochs 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from piml_mdo.structures.structural_surrogate import StructuralSurrogate, load_surrogate_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Train structural surrogate")
    parser.add_argument("--doe", type=str, default="results/structural_doe/doe.csv", help="DOE CSV path")
    parser.add_argument("--output", type=str, default="results/structural_doe/surrogate.pt", help="Output model path")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64, 32], help="Hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=3000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    X, y, feature_names, output_names = load_surrogate_dataset(args.doe)
    logger.info(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} outputs")
    logger.info(f"Features: {feature_names}")
    logger.info(f"Outputs: {output_names}")

    if X.shape[0] < 5:
        raise ValueError("Need at least 5 samples to train the surrogate")

    surrogate = StructuralSurrogate(
        input_dim=X.shape[1],
        output_dim=y.shape[1],
        hidden_layers=tuple(args.hidden),
        output_labels=[n.replace("out_", "") for n in output_names],
    )

    history = surrogate.fit(
        X,
        y,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        early_stopping_patience=300,
    )

    # Report accuracy on the full dataset
    y_pred = surrogate.predict(X)
    for i, name in enumerate(output_names):
        rmse = float(np.sqrt(np.mean((y[:, i] - y_pred[:, i]) ** 2)))
        logger.info(f"Output '{name}': RMSE = {rmse:.6e}")

    surrogate.save(args.output)

    # Save training metadata
    meta = {
        "doe_path": str(args.doe),
        "model_path": str(args.output),
        "feature_names": feature_names,
        "output_names": output_names,
        "hidden_layers": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "final_train_loss": history["train_loss"],
        "final_val_loss": history["val_loss"],
    }
    meta_path = Path(args.output).with_suffix(".train.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Training metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
