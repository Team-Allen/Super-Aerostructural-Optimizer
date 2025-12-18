"""Neural surrogate (NeuralFoil-style) for fast 2D drag prediction.

This module creates a small PyTorch MLP that predicts a scalar drag proxy
from a low-dimensional `latent` vector. If no dataset is present, it
constructs a synthetic dataset by mapping latent -> alpha_mask and labeling
with the JAX toy CFD `scalar_drag` function.
"""

from typing import Tuple
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag
import jax.numpy as jnp


class LatentDragMLP(nn.Module):
    def __init__(self, latent_dim=8, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def latent_to_alpha_mask(latent: np.ndarray, grid_size: int = 64) -> np.ndarray:
    """Map a latent vector to an alpha mask (Gaussian obstacle) on a grid.

    The mapping is synthetic and intended to generate varied shapes for labels.
    """
    xs = np.linspace(-1.0, 1.0, grid_size)
    ys = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # interpret first two dims as center, next as scale, next as amplitude
    cx = float(np.tanh(latent[0])) * 0.5
    cy = float(np.tanh(latent[1])) * 0.2
    sigma = 0.05 + 0.2 * (np.tanh(latent[2]) + 1) / 2
    amp = 5.0 * (0.5 + 0.5 * np.tanh(latent[3] if latent.shape[0] > 3 else 0.0))
    mask = amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    # clamp
    mask = np.clip(mask, 0.0, 50.0)
    return mask


def make_dataset(n_samples: int = 256, latent_dim: int = 8, grid_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    latents = np.random.randn(n_samples, latent_dim).astype(np.float32)
    labels = np.zeros((n_samples,), dtype=np.float32)
    # small grid coords
    xs = np.linspace(-1.0, 1.0, grid_size)
    ys = np.linspace(-1.0, 1.0, grid_size)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    mesh_coords = jnp.array(np.stack([X, Y], axis=-1))
    for i in range(n_samples):
        alpha = latent_to_alpha_mask(latents[i], grid_size=grid_size)
        p, vel = solve_steady_state_flow(mesh_coords, jnp.array([1.0, 0.0]), 1.0, jnp.array(alpha), num_steps=30)
        labels[i] = float(scalar_drag(p, vel))
    return latents, labels


def train_surrogate(model: LatentDragMLP,
                    latents: np.ndarray,
                    labels: np.ndarray,
                    epochs: int = 200,
                    lr: float = 1e-3,
                    batch_size: int = 64) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    X = torch.from_numpy(latents)
    y = torch.from_numpy(labels)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        t0 = time.time()
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().numpy())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"Epoch {epoch+1}/{epochs}, loss={total/len(loader):.6f}, time={time.time()-t0:.2f}s")


def predict_drag(model: LatentDragMLP, latent: np.ndarray) -> float:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(latent.astype(np.float32)).unsqueeze(0).to(device)
        out = model(z).cpu().numpy().squeeze()
    return float(out)


if __name__ == '__main__':
    # quick local train on synthetic dataset
    latents, labels = make_dataset(n_samples=128, latent_dim=8, grid_size=48)
    model = LatentDragMLP(latent_dim=8)
    train_surrogate(model, latents, labels, epochs=50)
    print('Sample predict:', predict_drag(model, latents[0]))
