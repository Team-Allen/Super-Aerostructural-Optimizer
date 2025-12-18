"""A minimal JAX-based structural solver prototype with a Neo-Hookean placeholder.

This provides vectorized stiffness assembly and a linear solve wrapper compatible
with JAX autodiff. It uses a dense representation for clarity; replace with
proper sparse assembly for production.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import vmap


class HyperElasticSolver:
    """Prototype hyperelastic solver (Neo-Hookean-like) with vectorized assembly."""

    def __init__(self):
        pass

    def get_stiffness_matrix(self, mesh_nodes: jnp.ndarray, material_props: dict) -> jnp.ndarray:
        """Assemble a dense stiffness matrix for demonstration.

        Args:
            mesh_nodes: (N, 2) nodal coordinates
            material_props: dict with 'E' and 'nu'

        Returns:
            K: (2N, 2N) stiffness matrix (dense placeholder)
        """
        N = mesh_nodes.shape[0]
        E = material_props.get("E", 70e9)
        nu = material_props.get("nu", 0.33)
        # dummy stiffness: identity scaled by E/N
        K = jnp.eye(2 * N) * (E / max(1, N))
        return K

    def _cg_solve(self, K: jnp.ndarray, f: jnp.ndarray, maxiter: int = 1000, tol: float = 1e-8) -> jnp.ndarray:
        """Fallback CG implemented with JAX primitives for dense K (educational)."""
        def body(state):
            x, r, p, rsold, it = state
            Ap = K @ p
            alpha = rsold / (p @ Ap + 1e-12)
            x_new = x + alpha * p
            r_new = r - alpha * Ap
            rsnew = r_new @ r_new
            beta = rsnew / (rsold + 1e-12)
            p_new = r_new + beta * p
            return (x_new, r_new, p_new, rsnew, it + 1)

        x0 = jnp.zeros_like(f)
        r0 = f - K @ x0
        p0 = r0
        rs0 = r0 @ r0

        def cond(state):
            x, r, p, rsold, it = state
            return (it < maxiter) & (rsold > tol ** 2)

        init_state = (x0, r0, p0, rs0, 0)
        x_final = jax.lax.while_loop(cond, lambda s: body(s), init_state)[0]
        return x_final

    @jax.custom_vjp
    def solve_displacement(self, forces: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
        """Solve K u = f and provide a custom VJP for efficient adjoint.

        Args:
            forces: (2N,) load vector
            K: (2N,2N) stiffness matrix

        Returns:
            u: (2N,) displacement vector
        """
        # use a simple dense solve here for clarity
        return jnp.linalg.solve(K, forces)


    def solve_displacement_fwd(self, forces, K):
        u = jnp.linalg.solve(K, forces)
        return u, (u, K)


    def solve_displacement_bwd(self, res, g):
        u, K = res
        # gradient w.r.t. forces is solution of K^T x = g
        sol = jnp.linalg.solve(K.T, g)
        # gradient w.r.t. K is -outer(sol, u)
        gradK = -jnp.outer(sol, u)
        return (sol, gradK)


    solve_displacement.defvjp(solve_displacement_fwd, solve_displacement_bwd)


if __name__ == "__main__":
    import numpy as np

    nodes = jnp.array(np.random.randn(10, 2))
    solver = HyperElasticSolver()
    K = solver.get_stiffness_matrix(nodes, {"E": 1e7})
    f = jnp.ones(20) * 1.0
    u = solver.solve_displacement(f, K)
    print("u", u.shape)
