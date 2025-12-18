"""A minimal JAX-based 2D CFD prototype with Brinkman penalization.

This solver is intentionally small and educational: it implements a simple
time-stepping scheme for a Brinkman-penalized Navier-Stokes-like update that
is compatible with `jax.grad` and `jax.jit`.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


def _rhs_step(state, inputs):
    # state: (u, v, p) arrays stacked
    u, v, p = state
    mesh_coords, inlet_velocity, density, alpha = inputs
    # simple diffusion-like relaxation towards inlet_velocity where alpha==0
    vel = jnp.stack([u, v], axis=-1)
    target = inlet_velocity
    # Brinkman penalization: alpha * (vel - 0)
    penal = -alpha[..., None] * vel
    # simple advection/diffusion placeholder: relax toward target + penal
    vel_new = vel + 0.1 * (target - vel) + 0.01 * penal
    # pressure as divergence surrogate
    div = jnp.gradient(vel_new[..., 0], axis=0) + jnp.gradient(vel_new[..., 1], axis=1)
    p_new = p - 0.1 * div
    return (vel_new[..., 0], vel_new[..., 1], p_new), None


@jax.jit(static_argnums=(4,))
def solve_steady_state_flow(mesh_coords: jnp.ndarray,
                            inlet_velocity: jnp.ndarray,
                            density: float,
                            alpha_mask: jnp.ndarray,
                            num_steps: int = 200) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Solve a toy 2D Brinkman-penalized flow field.

    Args:
        mesh_coords: array (Nx, Ny, 2) of coordinates (unused in toy solver)
        inlet_velocity: array (2,) or (Nx,Ny,2) desired freestream
        density: scalar fluid density
        alpha_mask: array (Nx, Ny) with large values inside solid
        num_steps: number of pseudo-time steps

    Returns:
        pressure_field, velocity_field (stacked)
    """
    Nx, Ny, _ = mesh_coords.shape
    # initialize fields
    u0 = jnp.zeros((Nx, Ny))
    v0 = jnp.zeros((Nx, Ny))
    p0 = jnp.zeros((Nx, Ny))

    # broadcast inlet_velocity to grid if needed
    if inlet_velocity.ndim == 1:
        inlet_vel_grid = jnp.broadcast_to(inlet_velocity, (Nx, Ny, 2))
    else:
        inlet_vel_grid = inlet_velocity

    inputs = (mesh_coords, inlet_vel_grid, density, alpha_mask)

    def body_fun(state, _):
        new_state, _ = _rhs_step(state, inputs)
        return new_state, None

    init_state = (u0, v0, p0)
    final_state, _ = lax.scan(lambda s, _: (body_fun(s, None)[0], None), init_state, None, length=num_steps)
    u_final, v_final, p_final = final_state
    vel = jnp.stack([u_final, v_final], axis=-1)
    return p_final, vel


def scalar_drag(pressure_field: jnp.ndarray, velocity_field: jnp.ndarray) -> jnp.ndarray:
    """Toy drag proxy: integrate pressure along a nominal chord line."""
    # simple L2 integral
    return jnp.sum(jnp.abs(pressure_field)) + 1e-3 * jnp.sum(jnp.linalg.norm(velocity_field, axis=-1))


if __name__ == "__main__":
    import numpy as np

    Nx = Ny = 64
    xs = np.linspace(-1.0, 1.0, Nx)
    ys = np.linspace(-1.0, 1.0, Ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    coords = jnp.array(np.stack([X, Y], axis=-1))
    alpha = jnp.zeros((Nx, Ny))
    p, vel = solve_steady_state_flow(coords, jnp.array([1.0, 0.0]), 1.0, alpha, num_steps=50)
    print("p.shape", p.shape, "vel.shape", vel.shape)
