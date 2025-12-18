"""Train a NeuralFoil surrogate (if needed), optimize a latent vector using it,
and validate the found latent with the JAX CFD solver.
"""
import os
import numpy as np
from scipy.optimize import minimize

from ..generative.neuralfoil_surrogate import LatentDragMLP, make_dataset, train_surrogate, predict_drag
from ..solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag
import jax.numpy as jnp


def optimize_latent_with_surrogate(latent_dim=8):
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'neuralfoil.pth')
    # create model and dataset
    model = LatentDragMLP(latent_dim=latent_dim)
    # generate small dataset and train (fast)
    print('Generating synthetic dataset and training surrogate...')
    latents, labels = make_dataset(n_samples=128, latent_dim=latent_dim, grid_size=48)
    train_surrogate(model, latents, labels, epochs=80, lr=1e-3)

    # optimize latent vector using surrogate prediction
    z0 = np.zeros(latent_dim, dtype=np.float32)

    def obj(z):
        return predict_drag(model, z)

    print('Running SciPy optimization on surrogate...')
    res = minimize(obj, z0, method='Nelder-Mead', options={'maxiter': 200})
    print('Surrogate optimization result:', res.fun, res.x)

    # validate with JAX CFD
    from ..generative.neuralfoil_surrogate import latent_to_alpha_mask
    alpha = latent_to_alpha_mask(res.x, grid_size=64)
    xs = np.linspace(-1.0, 1.0, 64)
    ys = np.linspace(-1.0, 1.0, 64)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    mesh_coords = jnp.array(np.stack([X, Y], axis=-1))
    p, vel = solve_steady_state_flow(mesh_coords, jnp.array([1.0, 0.0]), 1.0, jnp.array(alpha), num_steps=60)
    true_drag = float(scalar_drag(p, vel))
    print('Validated drag from JAX CFD:', true_drag)
    return res.x, res.fun, true_drag


if __name__ == '__main__':
    z, pred, true = optimize_latent_with_surrogate()
    print('Done. surrogate predicted:', pred, 'validated:', true)
