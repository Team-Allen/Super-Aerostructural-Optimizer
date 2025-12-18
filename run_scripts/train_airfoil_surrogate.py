import os
import sys
import pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

proj_root = str(pathlib.Path(__file__).resolve().parents[1])
pkg_root = os.path.join(proj_root, 'Super_Aerostructural_Optimizer')
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from Super_Aerostructural_Optimizer.solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag

import torch
import torch.nn as nn

DATA_DIR = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]), 'airfoil_database')
OUT = os.path.join(os.path.dirname(__file__), 'ui_outputs')
os.makedirs(OUT, exist_ok=True)


def read_dat(path):
    pts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                x = float(parts[0])
                y = float(parts[1])
                pts.append((x, y))
            except Exception:
                continue
    return np.array(pts)


def coords_to_mask(coords, grid_size=64):
    # normalize coords to [ -1, 1 ] range
    xs = coords[:, 0]
    ys = coords[:, 1]
    if xs.max() - xs.min() == 0:
        return np.zeros((grid_size, grid_size))
    x_norm = 2 * (xs - xs.min()) / (xs.max() - xs.min()) - 1
    y_norm = 2 * (ys - ys.min()) / (ys.max() - ys.min()) - 1
    poly = np.stack([x_norm, y_norm], axis=-1)

    # rasterize using point-in-polygon test
    gx = np.linspace(-1, 1, grid_size)
    gy = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    pts = np.stack([X.ravel(), Y.ravel()], axis=-1)
    from matplotlib.path import Path
    path = Path(poly)
    mask = path.contains_points(pts).astype(float).reshape((grid_size, grid_size))
    # smooth / dilate a bit
    mask = np.clip(mask * 1.0, 0.0, 1.0)
    return mask


def build_dataset(max_files=120, grid_size=64):
    files = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.dat')])
    X = []
    y = []
    import jax.numpy as jnp
    for i, fpath in enumerate(files[:max_files]):
        coords = read_dat(fpath)
        if coords.shape[0] < 5:
            continue
        mask = coords_to_mask(coords, grid_size=grid_size)
        # compute JAX label
        xs = np.linspace(-1.0, 1.0, grid_size)
        ys = np.linspace(-1.0, 1.0, grid_size)
        Xc, Yc = np.meshgrid(xs, ys, indexing='ij')
        mesh_coords = jnp.array(np.stack([Xc, Yc], axis=-1))
        try:
            p, vel = solve_steady_state_flow(mesh_coords, jnp.array([1.0, 0.0]), 1.0, jnp.array(mask), num_steps=50)
            lbl = float(scalar_drag(p, vel))
        except Exception:
            lbl = float(np.sum(mask))
        X.append(mask.ravel().astype(np.float32))
        y.append(lbl)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.float32)
    return X, y


class MaskDragMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_eval():
    X, y = build_dataset(max_files=120, grid_size=64)
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    Xtr, ytr = X[idx[:split]], y[idx[:split]]
    Xte, yte = X[idx[split:]], y[idx[split:]]

    model = MaskDragMLP(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    for ep in range(80):
        model.train()
        opt.zero_grad()
        pred = model(Xtr_t)
        loss = loss_fn(pred, ytr_t)
        loss.backward()
        opt.step()
        if (ep + 1) % 20 == 0:
            print(f'Epoch {ep+1}/{80}, loss={loss.item():.4f}')

    # eval
    model.eval()
    with torch.no_grad():
        pred_tr = model(torch.from_numpy(Xtr)).numpy()
        pred_te = model(torch.from_numpy(Xte)).numpy()

    # plots
    plt.figure(figsize=(6, 6))
    plt.scatter(yte, pred_te)
    plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], 'r--')
    plt.xlabel('JAX drag')
    plt.ylabel('Surrogate prediction')
    plt.title('Surrogate vs JAX on test set')
    plt.savefig(os.path.join(OUT, 'airfoil_surrogate_scatter.png'))

    # save model
    try:
        torch.save(model.state_dict(), os.path.join(OUT, 'mask_surrogate.pt'))
    except Exception:
        pass

    print('Test MSE:', float(np.mean((pred_te - yte) ** 2)))


if __name__ == '__main__':
    train_and_eval()
