"""OpenMDAO MPhys wrappers for JAX components.

This file provides a `JaxAerostructGroup` that includes a `FluidsComp` ExplicitComponent
which delegates flow solves to the JAX solver and uses JAX jvp/vjp for matrix-free
derivative actions.
"""

from typing import Any

import numpy as np

import openmdao.api as om
import jax
import jax.numpy as jnp

from ..solvers.fluid.jax_cfd import solve_steady_state_flow, scalar_drag
import torch


class FluidsComp(om.ExplicitComponent):
    """OpenMDAO component wrapping the `solve_steady_state_flow` JAX function."""

    def setup(self):
        # Assume a flattened grid representation for inputs
        self.add_input('mesh_coords', val=np.zeros((64, 64, 2)))
        self.add_input('alpha_mask', val=np.zeros((64, 64)))
        self.add_input('inlet_velocity', val=np.array([1.0, 0.0]))
        # optional latent representation for surrogate fast path
        self.add_input('latent', val=np.zeros(6))
        self.add_input('use_surrogate', val=0.0)
        self.add_output('pressure', val=np.zeros((64, 64)))
        self.add_output('velocity', val=np.zeros((64, 64, 2)))
        # scalar drag output (either from surrogate or computed from JAX fields)
        self.add_output('drag', val=1.0)
        # declare partials for drag w.r.t latent (we provide analytic when possible)
        self.declare_partials('drag', 'latent')
        self.declare_partials('*', '*', method='cs')

    def compute(self, inputs, outputs):
        use_sur = float(inputs['use_surrogate'])
        mesh = jnp.array(inputs['mesh_coords'])
        alpha = jnp.array(inputs['alpha_mask'])
        inlet = jnp.array(inputs['inlet_velocity'])

        # If a surrogate callable is attached and requested, use it for fast drag
        if getattr(self, 'surrogate', None) is not None and use_sur > 0.5:
            latent = np.asarray(inputs['latent'], dtype=np.float32)
            try:
                # prefer raw torch model for gradient support if available
                if hasattr(self, 'torch_surrogate') and isinstance(self.torch_surrogate, torch.nn.Module):
                    # use callable wrapper if provided
                    try:
                        drag_val = float(self.torch_surrogate(torch.from_numpy(latent.astype(np.float32)).unsqueeze(0)).detach().cpu().numpy().squeeze())
                    except Exception:
                        drag_val = float(self.surrogate(latent))
                else:
                    drag_val = float(self.surrogate(latent))
                outputs['drag'] = drag_val
                # keep fields zeroed (surrogate only predicts scalar)
                outputs['pressure'] = np.zeros_like(outputs['pressure'])
                outputs['velocity'] = np.zeros_like(outputs['velocity'])
                return
            except Exception:
                # fallback to JAX solve if surrogate fails
                pass

        # default: run JAX solver and compute drag
        p, vel = solve_steady_state_flow(mesh, inlet, 1.0, alpha, num_steps=100)
        outputs['pressure'] = np.array(p)
        outputs['velocity'] = np.array(vel)
        outputs['drag'] = float(scalar_drag(p, vel))

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        # Matrix-free partials: use JAX jvp/vjp when OpenMDAO requests directional derivatives.
        mesh = jnp.array(inputs['mesh_coords'])
        alpha = jnp.array(inputs['alpha_mask'])
        inlet = jnp.array(inputs['inlet_velocity'])

        def forward_fn(mesh_coords, inlet_velocity, alpha_mask):
            p, vel = solve_steady_state_flow(mesh_coords, inlet_velocity, 1.0, alpha_mask, num_steps=100)
            # flatten outputs to a 1D vector for convenience
            return jnp.concatenate([p.ravel(), vel.ravel()])

        primals = (mesh, inlet, alpha)

        if mode == 'fwd':
            # build tangent vector
            t_mesh = jnp.array(d_inputs.get('mesh_coords', np.zeros_like(mesh)))
            t_inlet = jnp.array(d_inputs.get('inlet_velocity', np.zeros_like(inlet)))
            t_alpha = jnp.array(d_inputs.get('alpha_mask', np.zeros_like(alpha)))
            y_dot = jax.jvp(forward_fn, primals, (t_mesh, t_inlet, t_alpha))[1]
            # scatter into d_outputs
            n_p = mesh.shape[0] * mesh.shape[1]
            d_outputs['pressure'] = np.array(y_dot[:n_p].reshape(mesh.shape[:2]))
            d_outputs['velocity'] = np.array(y_dot[n_p:].reshape((*mesh.shape[:2], 2)))
        elif mode == 'rev':
            # transpose-mode: receive d_outputs and compute adjoint to inputs
            n_p = mesh.shape[0] * mesh.shape[1]
            adj_out = jnp.concatenate([jnp.array(d_outputs.get('pressure', np.zeros((mesh.shape[0], mesh.shape[1])))).ravel(),
                                      jnp.array(d_outputs.get('velocity', np.zeros((*mesh.shape[:2], 2)))).ravel()])
            vjp_fun = jax.vjp(forward_fn, *primals)[1]
            grads = vjp_fun(adj_out)
            d_inputs['mesh_coords'] = np.array(grads[0])
            d_inputs['inlet_velocity'] = np.array(grads[1])
            d_inputs['alpha_mask'] = np.array(grads[2])

        # Note: surrogate-forward derivatives handled by OpenMDAO FD declared for 'drag'->'latent'
    def compute_partials(self, inputs, partials):
        """Provide analytic partials for `drag` w.r.t `latent` when a torch surrogate is attached.

        Fills `partials['drag', 'latent']` with gradient values.
        """
        use_sur = float(inputs['use_surrogate'])
        if use_sur <= 0.5:
            # no surrogate path; leave partials as zeros (OpenMDAO will use defaults)
            return

        if not hasattr(self, 'torch_surrogate') or not isinstance(self.torch_surrogate, torch.nn.Module):
            return

        # compute gradient via torch autograd
        latent_np = np.asarray(inputs['latent'], dtype=np.float32)
        z = torch.tensor(latent_np, dtype=torch.float32, requires_grad=True)
        model = self.torch_surrogate
        model.eval()
        try:
            out = model(z.unsqueeze(0)).squeeze()
            if out.dim() != 0:
                out = out.squeeze()
            out.backward()
            grad = z.grad.detach().cpu().numpy().astype(float)
            partials['drag', 'latent'] = grad
        except Exception:
            # on failure, leave partials unset so OpenMDAO can fallback
            return

        


class JaxAerostructGroup(om.Group):
    def setup(self):
        self.add_subsystem('fluids', FluidsComp(), promotes=['*'])


if __name__ == "__main__":
    # quick smoke test: create a problem with the group
    prob = om.Problem()
    prob.model = JaxAerostructGroup()
    prob.setup(check=False)
    prob.set_val('mesh_coords', np.zeros((64, 64, 2)))
    prob.run_model()
    print('pressure', prob.get_val('pressure').shape)
