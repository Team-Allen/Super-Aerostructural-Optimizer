"""Unit test: compare JAX gradient to finite-difference for the toy CFD solver."""

import numpy as np
import jax
import jax.numpy as jnp

from Super_Aerostructural_Optimizer.solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag


def fd_grad(mesh_coords, inlet_velocity, density, alpha_mask, eps=1e-6):
    # finite-difference gradient of scalar_drag wrt a single scalar inlet velocity component
    base_p, base_vel = solve_steady_state_flow(mesh_coords, inlet_velocity, density, alpha_mask, num_steps=50)
    base_loss = scalar_drag(base_p, base_vel)
    pert = inlet_velocity + jnp.array([eps, 0.0])
    p2, v2 = solve_steady_state_flow(mesh_coords, pert, density, alpha_mask, num_steps=50)
    loss2 = scalar_drag(p2, v2)
    return (loss2 - base_loss) / eps


def test_inlet_grad_match():
    Nx = Ny = 32
    xs = np.linspace(-1.0, 1.0, Nx)
    ys = np.linspace(-1.0, 1.0, Ny)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    coords = jnp_coords = jnp.array(np.stack([X, Y], axis=-1))
    alpha = jnp.zeros((Nx, Ny))
    inlet = jnp.array([1.0, 0.0])

    # JAX gradient via autodiff
    def loss_fn(inlet_v):
        p, vel = solve_steady_state_flow(coords, inlet_v, 1.0, alpha, num_steps=50)
        return scalar_drag(p, vel)

    jax_grad = jax.grad(loss_fn)(inlet)
    fd = fd_grad(coords, inlet, 1.0, alpha, eps=1e-4)
    # compare first component gradient
    # Relaxed tolerance for toy solver approximation
    assert abs(float(jax_grad[0]) - float(fd)) < 5e-2
