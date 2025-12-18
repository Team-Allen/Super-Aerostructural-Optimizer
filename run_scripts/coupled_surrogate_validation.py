import os
import sys
import pathlib
import numpy as np

# ensure package imports work
proj_root = str(pathlib.Path(__file__).resolve().parents[1])
pkg_root = os.path.join(proj_root, 'Super_Aerostructural_Optimizer')
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import LatentDragMLP, make_dataset, predict_drag, latent_to_alpha_mask
from Super_Aerostructural_Optimizer.mphys_interface.jax_component import JaxAerostructGroup, FluidsComp

import openmdao.api as om

OUT = os.path.join(os.path.dirname(__file__), 'ui_outputs')
os.makedirs(OUT, exist_ok=True)


def train_surrogate_and_run():
    latent_dim = 6
    X, y = make_dataset(n_samples=300, latent_dim=latent_dim)
    model = LatentDragMLP(latent_dim=latent_dim)
    from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import train_surrogate
    train_surrogate(model, X, y, epochs=40, lr=1e-3)

    def model_callable(z: np.ndarray) -> float:
        return predict_drag(model, z)

    # Build OpenMDAO problem with a FluidsComp instance
    prob = om.Problem()
    fluids = FluidsComp()
    prob.model.add_subsystem('fluids', fluids, promotes=['*'])

    # expose latent and use_surrogate at top-level via IndepVarComp
    indep = om.IndepVarComp()
    indep.add_output('latent', val=np.zeros(latent_dim))
    indep.add_output('use_surrogate', val=1.0)
    prob.model.add_subsystem('indep', indep, promotes=['*'])

    # setup to create components
    prob.setup(check=False)
    fluids.surrogate = model_callable
    # attach raw torch model for analytic gradient support
    fluids.torch_surrogate = model

    # Add design var and objective
    prob.model.add_design_var = getattr(prob.model, 'add_design_var', None)
    # For simplicity, run a manual surrogate optimization loop using SciPy outside OpenMDAO
    from scipy.optimize import minimize

    def obj(z):
        return float(model_callable(z))

    res = minimize(obj, np.zeros(latent_dim), method='Nelder-Mead', options={'maxiter': 30})

    best = res.x
    # validate with JAX run by switching off surrogate and running model
    prob.set_val('latent', best)
    prob.set_val('use_surrogate', 0.0)
    prob.setup(check=False)
    prob.run_model()

    # collect JAX drag by calling the FluidsComp compute results
    drag = prob.get_val('drag')

    # save final mask
    mask = latent_to_alpha_mask(best)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.imsave(os.path.join(OUT, 'coupled_final_mask.png'), mask, cmap='gray')
    with open(os.path.join(OUT, 'coupled_validation.txt'), 'w') as f:
        f.write(f'surrogate_pred:{res.fun}\n')
        f.write(f'jax_validated:{float(drag)}\n')

    print('Surrogate pred, JAX validated:', res.fun, float(drag))


if __name__ == '__main__':
    train_surrogate_and_run()
