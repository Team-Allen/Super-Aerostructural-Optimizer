"""Optimization script wiring the OpenMDAO problem to the JAX-backed components.

This script is a runnable example that sets up an OpenMDAO Problem with the
`JaxAerostructGroup`, exposes a latent vector as a design variable, and runs
an SLSQP optimization. The objective and constraints are placeholders and
should be replaced with problem-appropriate metrics.
"""

import numpy as np
import openmdao.api as om

from ..mphys_interface.jax_component import JaxAerostructGroup
from ..generative.geometry_decoder import WingDecoder


def build_problem(grid_res: int = 64):
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('aero_group', JaxAerostructGroup(), promotes=['*'])

    # latent vector indep var
    ivc = om.IndepVarComp()
    ivc.add_output('latent_vector', val=np.zeros(64))
    model.add_subsystem('inputs', ivc, promotes=['*'])

    # small recorder
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'

    return prob


def main():
    prob = build_problem()
    prob.setup()

    # example: run a single model evaluation
    prob.set_val('mesh_coords', np.zeros((64, 64, 2)))
    prob.set_val('alpha_mask', np.zeros((64, 64)))
    prob.set_val('inlet_velocity', np.array([1.0, 0.0]))
    prob.run_model()
    print('Pressure shape:', prob.get_val('pressure').shape)


if __name__ == '__main__':
    main()
