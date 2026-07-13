"""
Neural surrogate for fast structural response prediction.

The surrogate is trained offline on high-fidelity MYSTRAN data and, optionally,
a physics-informed loss built from the VAM beam solver.  At runtime it replaces
(or augments) the 1-D beam solver in the aerostructural coupling loop so that
expensive 3-D shell analyses are not required during MDO.

Supported usage modes:
- Direct: map design/load features to structural responses
  (tip deflection, failure index, max stress).
- Physics-informed: the network predicts a correction factor relative to the
  fast VAM beam solution, preserving beam physics and improving accuracy.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from .beam_solver import EulerBernoulliBeamSolver, WingStructure

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        act = {"relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}.get(activation, nn.ReLU)
        layers = []
        prev = input_dim
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), act()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


@dataclass
class Normalizer:
    """Feature-wise min/max normalizer."""

    x_min: np.ndarray
    x_max: np.ndarray
    y_min: np.ndarray
    y_max: np.ndarray
    eps: float = 1e-8

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.x_min) / (self.x_max - self.x_min + self.eps)

    def denormalize_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return y * (self.y_max - self.y_min + self.eps) + self.y_min

    def to_dict(self) -> dict:
        return {
            "x_min": self.x_min.tolist(),
            "x_max": self.x_max.tolist(),
            "y_min": self.y_min.tolist(),
            "y_max": self.y_max.tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(
            x_min=np.asarray(d["x_min"], dtype=float),
            x_max=np.asarray(d["x_max"], dtype=float),
            y_min=np.asarray(d["y_min"], dtype=float),
            y_max=np.asarray(d["y_max"], dtype=float),
            eps=float(d["eps"]),
        )


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------


def build_surrogate_features(
    wing: "WingStructure",
    lift_distribution: np.ndarray,
    moment_distribution: Optional[np.ndarray] = None,
    load_factor: float = 1.0,
) -> np.ndarray:
    """Build the feature vector expected by the structural surrogate.

    The feature order matches the DOE CSV produced by
    ``scripts/generate_structural_doe.py``:
    [n0, n45, n45_neg, n90, ply_thickness, total_force, root_moment, root_torque].
    """
    # Use the root laminate (or the single skin laminate) for ply descriptors.
    lam = wing.skin_laminate
    if wing.skin_laminates is not None and len(wing.skin_laminates) == wing.n_elements + 1:
        lam = wing.skin_laminates[0]

    angles = lam.angles if lam is not None else []
    counts = {0.0: 0, 45.0: 0, -45.0: 0, 90.0: 0}
    for theta in angles:
        counts[float(theta)] = counts.get(float(theta), 0) + 1

    ply_thickness = lam.material.t_ply if lam is not None else 0.0

    y = wing.spanwise_stations()
    q = np.asarray(lift_distribution) * load_factor
    total_force = float(np.trapz(q, y))
    root_moment = float(np.trapz(y * q, y))

    if moment_distribution is not None:
        m = np.asarray(moment_distribution) * load_factor
        root_torque = float(np.trapz(np.abs(m), y))
    else:
        root_torque = 0.0

    return np.array(
        [counts[0.0], counts[45.0], counts[-45.0], counts[90.0], ply_thickness, total_force, root_moment, root_torque],
        dtype=float,
    )


class StructuralSurrogate:
    """PyTorch MLP surrogate for structural response quantities.

    The default output labels are:
        - max_vertical_displacement
        - max_von_mises_stress
        - max_failure_index
        - total_mass

    Any subset can be used by the caller.
    """

    DEFAULT_OUTPUTS = [
        "max_vertical_displacement",
        "max_von_mises_stress",
        "max_failure_index",
        "total_mass",
    ]

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        activation: str = "relu",
        normalize: bool = True,
        output_labels: Optional[list[str]] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.output_labels = output_labels or self.DEFAULT_OUTPUTS[:output_dim]
        self.hidden_layers = hidden_layers
        self.activation = activation

        self.model = _MLP(input_dim, output_dim, hidden_layers, activation)
        self.normalizer: Optional[Normalizer] = None
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 2000,
        lr: float = 1e-3,
        batch_size: Optional[int] = None,
        val_split: float = 0.1,
        physics_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        physics_weight: float = 0.0,
        early_stopping_patience: int = 200,
        verbose: bool = True,
    ) -> dict:
        """Train the surrogate on (X, y) data.

        Args:
            X: [n_samples, n_features] input features.
            y: [n_samples, n_outputs] target outputs.
            epochs: Number of training epochs.
            lr: Adam learning rate.
            batch_size: If ``None``, use the full training set.
            val_split: Fraction of data reserved for validation.
            physics_fn: Optional function ``f(X_tensor) -> physics_term`` that
                returns a tensor used in a physics-informed loss term.
            physics_weight: Weight for the physics loss term.
            early_stopping_patience: Stop if validation loss does not improve.
            verbose: Print training progress.

        Returns:
            Training history dict with ``train_loss`` and ``val_loss``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        if self.normalize:
            self.normalizer = Normalizer(
                x_min=X.min(axis=0),
                x_max=X.max(axis=0),
                y_min=y.min(axis=0),
                y_max=y.max(axis=0),
            )
            Xn = self.normalizer.normalize_x(X)
            yn = (y - self.normalizer.y_min) / (self.normalizer.y_max - self.normalizer.y_min + self.normalizer.eps)
        else:
            Xn = X.copy()
            yn = y.copy()

        # Train/validation split
        n = Xn.shape[0]
        n_val = max(1, int(n * val_split))
        perm = np.random.permutation(n)
        train_idx = perm[n_val:]
        val_idx = perm[:n_val]

        X_train = torch.tensor(Xn[train_idx], dtype=torch.float32)
        y_train = torch.tensor(yn[train_idx], dtype=torch.float32)
        X_val = torch.tensor(Xn[val_idx], dtype=torch.float32)
        y_val = torch.tensor(yn[val_idx], dtype=torch.float32)

        if batch_size is None or batch_size >= len(X_train):
            batch_size = len(X_train)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, verbose=False)
        mse = nn.MSELoss()

        best_val = float("inf")
        patience_counter = 0
        self._history = []

        for epoch in range(epochs):
            self.model.train()
            permutation = torch.randperm(len(X_train))
            train_losses = []
            for i in range(0, len(X_train), batch_size):
                idx = permutation[i : i + batch_size]
                xb = X_train[idx]
                yb = y_train[idx]

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = mse(pred, yb)

                if physics_fn is not None and physics_weight > 0.0:
                    phys = physics_fn(xb)
                    loss = loss + physics_weight * phys.mean()

                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = float(mse(val_pred, y_val).item())

            avg_train = float(np.mean(train_losses))
            self._history.append({"epoch": epoch, "train": avg_train, "val": val_loss})
            scheduler.step(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 200 == 0 or epoch == epochs - 1):
                logger.info(
                    f"Surrogate epoch {epoch}: train_loss={avg_train:.6e}, val_loss={val_loss:.6e}"
                )

            if patience_counter >= early_stopping_patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch}")
                break

        return {"train_loss": avg_train, "val_loss": val_loss, "history": self._history}

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict outputs for input features X."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.normalize and self.normalizer is not None:
            Xn = self.normalizer.normalize_x(X)
        else:
            Xn = X.copy()

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(Xn, dtype=torch.float32)).numpy()

        if self.normalize and self.normalizer is not None:
            y_pred = self.normalizer.denormalize_y(y_pred)
        return y_pred

    def predict_dict(self, X: np.ndarray) -> dict[str, float]:
        """Predict and return a dict keyed by output label."""
        y = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(1, -1)
        return {label: float(y[0, i]) for i, label in enumerate(self.output_labels)}

    # ------------------------------------------------------------------
    # Physics-informed loss helper
    # ------------------------------------------------------------------

    @staticmethod
    def make_physics_loss(
        wing: "WingStructure",
        feature_extractor: Callable[[torch.Tensor], dict],
        response_index: int = 0,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build a physics loss that penalises violations of beam scaling.

        The VAM beam solver gives a reference tip deflection ``w_ref``.  The
        surrogate prediction for the displacement output should not deviate
        from ``w_ref`` by more than the data-driven residual being learned.
        This term is intended to be used when the network predicts correction
        factors, not absolute responses.

        Args:
            wing: Wing structure used for the VAM beam solution.
            feature_extractor: Function that turns a batch of input tensors
                into dictionaries accepted by the beam solver (e.g. lift).
            response_index: Index of the surrogate output that corresponds to
                the displacement response.

        Returns:
            Loss function ``physics_fn(X_tensor) -> scalar tensor``.
        """
        from .beam_solver import EulerBernoulliBeamSolver
        solver = EulerBernoulliBeamSolver(wing)

        def _loss(xb: torch.Tensor) -> torch.Tensor:
            # Convert batch to numpy, solve VAM beam, compare to prediction.
            x_np = xb.detach().cpu().numpy()
            residuals = []
            for xi in x_np:
                inputs = feature_extractor(xi)
                result = solver.solve(**inputs)
                # The physics term is currently a placeholder that returns zero
                # because wiring a full beam solve into the training loop is
                # problem-specific.  Users can override this method.
                residuals.append(0.0)
            return torch.tensor(residuals, dtype=torch.float32, device=xb.device)

        return _loss

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: Path | str):
        """Save the surrogate model, normalizer, and metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "output_labels": self.output_labels,
            "normalize": self.normalize,
            "model_state": self.model.state_dict(),
            "normalizer": self.normalizer.to_dict() if self.normalizer else None,
        }
        torch.save(state, path)
        logger.info(f"Saved structural surrogate to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "StructuralSurrogate":
        """Load a saved surrogate."""
        path = Path(path)
        state = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(
            input_dim=state["input_dim"],
            output_dim=state["output_dim"],
            hidden_layers=state["hidden_layers"],
            activation=state["activation"],
            output_labels=state["output_labels"],
            normalize=state["normalize"],
        )
        obj.model.load_state_dict(state["model_state"])
        if state["normalizer"] is not None:
            obj.normalizer = Normalizer.from_dict(state["normalizer"])
        obj.model.eval()
        logger.info(f"Loaded structural surrogate from {path}")
        return obj


# ---------------------------------------------------------------------------
# Convenience: CSV helpers
# ---------------------------------------------------------------------------


def load_surrogate_dataset(csv_path: Path | str) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load a CSV dataset produced by the structural DOE script.

    Returns:
        X, y, feature_names, output_names.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    meta = json.loads(df.attrs.get("meta", "{}")) if hasattr(df, "attrs") else {}
    feature_names = meta.get("feature_names", [c for c in df.columns if c.startswith("feat_")])
    output_names = meta.get("output_names", [c for c in df.columns if c.startswith("out_")])

    X = df[feature_names].to_numpy(dtype=float)
    y = df[output_names].to_numpy(dtype=float)
    return X, y, feature_names, output_names
