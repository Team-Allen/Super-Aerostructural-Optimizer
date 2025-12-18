import os
import time
import numpy as np
import sys
import pathlib

# Ensure project root is on sys.path so `generative` and `solvers` imports work
proj_root = str(pathlib.Path(__file__).resolve().parents[1])
# The project package lives in `Super_Aerostructural_Optimizer/` inside repo root
pkg_root = os.path.join(proj_root, 'Super_Aerostructural_Optimizer')
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from scipy.optimize import minimize

from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import make_dataset, LatentDragMLP, latent_to_alpha_mask
try:
    from Super_Aerostructural_Optimizer.solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag
    jax_available = True
except Exception:
    jax_available = False


OUT_DIR = os.path.join(os.path.dirname(__file__), "ui_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def train_surrogate(latent_dim=6, n=300, epochs=20, lr=1e-3):
    X, y = make_dataset(n_samples=n, latent_dim=latent_dim)
    model = LatentDragMLP(latent_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    losses = []
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    for ep in range(epochs):
        optim.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, y_t)
        loss.backward()
        optim.step()
        losses.append(float(loss.item()))
    return model, losses


def surrogate_predict(model, z):
    zt = torch.tensor(z, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        p = model(zt).item()
    return float(p)


def true_evaluate(z):
    # if JAX CFD available, evaluate real drag; otherwise use surrogate-ish fallback
    if jax_available:
        mask = latent_to_alpha_mask(z)
        # build simple mesh coords used by the toy solver
        Nx = mask.shape[0]
        xs = np.linspace(-1.0, 1.0, Nx)
        ys = np.linspace(-1.0, 1.0, Nx)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        try:
            import jax.numpy as jnp
            mesh_coords = jnp.array(np.stack([X, Y], axis=-1))
            p, vel = solve_steady_state_flow(mesh_coords, jnp.array([1.0, 0.0]), 1.0, jnp.array(mask), num_steps=30)
            return float(scalar_drag(p, vel))
        except Exception:
            return None
    else:
        # fallback: small perturbation of surrogate-like value (not accurate)
        return None


def run_ui_session():
    latent_dim = 6
    print('Training small surrogate...')
    model, losses = train_surrogate(latent_dim=latent_dim, n=300, epochs=25)

    # initial latent
    x0 = np.zeros(latent_dim)

    iter_history = []
    pred_history = []
    true_history = []

    def callback(xk):
        val = surrogate_predict(model, xk)
        iter_history.append(xk.copy())
        pred_history.append(val)
        tv = true_evaluate(xk)
        true_history.append(tv if tv is not None else np.nan)

        # save a quick plot per callback step
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        # show latest geometry mask
        try:
            mask = latent_to_alpha_mask(xk)
            ax1.imshow(mask, cmap='gray')
            ax1.set_title('Airfoil mask')
            ax1.axis('off')
        except Exception:
            ax1.text(0.5, 0.5, 'mask unavailable', ha='center')

        ax2.plot(pred_history, label='surrogate')
        if np.any(~np.isnan(true_history)):
            ax2.plot(true_history, label='true')
        ax2.set_title('Objective over iterations')
        ax2.set_xlabel('iteration')
        ax2.set_ylabel('drag')
        ax2.legend()

        fname = os.path.join(OUT_DIR, f'iter_{len(pred_history):03d}.png')
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)

    print('Running optimization on surrogate (Nelder-Mead)...')
    res = minimize(lambda z: surrogate_predict(model, z), x0, method='Nelder-Mead', callback=callback,
                   options={'maxiter': 40, 'xatol': 1e-3, 'fatol': 1e-3})

    print('Optimization finished. Result:', res.fun)

    # Save summary plots
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(losses)
    ax.set_title('Surrogate training loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE')
    fig.savefig(os.path.join(OUT_DIR, 'training_loss.png'))
    plt.close(fig)

    # Compose gif from iteration PNGs if available
    try:
        from PIL import Image
        pngs = sorted([p for p in os.listdir(OUT_DIR) if p.startswith('iter_') and p.endswith('.png')])
        imgs = [Image.open(os.path.join(OUT_DIR, p)) for p in pngs]
        if imgs:
            gif_path = os.path.join(OUT_DIR, 'optimization_progress.gif')
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=400, loop=0)
            print('Saved GIF:', gif_path)
    except Exception as e:
        print('Could not make GIF:', e)

    # Save final airfoil mask
    best = res.x
    try:
        mask = latent_to_alpha_mask(best)
        plt.imsave(os.path.join(OUT_DIR, 'final_airfoil.png'), mask, cmap='gray')
        print('Saved final airfoil mask to', os.path.join(OUT_DIR, 'final_airfoil.png'))
    except Exception:
        pass

    # If JAX available, compute true drag for best
    if jax_available:
        true_best = true_evaluate(best)
        with open(os.path.join(OUT_DIR, 'validation.txt'), 'w') as f:
            f.write(f'best_surrogate:{res.fun}\n')
            f.write(f'best_true:{true_best}\n')
        print('Validation saved to', os.path.join(OUT_DIR, 'validation.txt'))
    else:
        with open(os.path.join(OUT_DIR, 'validation.txt'), 'w') as f:
            f.write('JAX CFD not available; no true validation performed.\n')
        print('JAX not available; wrote note to validation.txt')


if __name__ == '__main__':
    start = time.time()
    run_ui_session()
    print('Done in %.1fs' % (time.time() - start))
