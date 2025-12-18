import os
import sys
import pathlib
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ensure project package import
proj_root = str(pathlib.Path(__file__).resolve().parents[1])
pkg_root = os.path.join(proj_root, 'Super_Aerostructural_Optimizer')
if pkg_root not in sys.path:
    sys.path.insert(0, pkg_root)

from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import LatentDragMLP, make_dataset, predict_drag
from Super_Aerostructural_Optimizer.mphys_interface.surrogate_component import SurrogateComp

from openmdao.api import Problem, IndepVarComp, ScipyOptimizeDriver, SqliteRecorder

OUT = os.path.join(os.path.dirname(__file__), 'ui_outputs')
os.makedirs(OUT, exist_ok=True)


def train_and_optimize(latent_dim=6, n_samples=400):
    # train surrogate
    X, y = make_dataset(n_samples=n_samples, latent_dim=latent_dim)
    model = LatentDragMLP(latent_dim=latent_dim)
    # use training helper defined in module
    from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import train_surrogate
    train_surrogate(model, X, y, epochs=60, lr=1e-3)

    # wrap model as a callable
    def model_callable(z: np.ndarray) -> float:
        return predict_drag(model, z)

    # build OpenMDAO problem
    prob = Problem()
    indep = IndepVarComp()
    indep.add_output('latent', val=np.zeros(latent_dim))
    prob.model.add_subsystem('indep', indep, promotes=['*'])
    comp = SurrogateComp(latent_dim=latent_dim)
    comp.model = model_callable
    # also attach raw torch model for analytic gradients if needed
    comp.torch_model = model
    prob.model.add_subsystem('sur', comp, promotes=['*'])

    prob.driver = ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-6
    prob.driver.options['maxiter'] = 100

    prob.model.add_design_var('latent', lower=-3.0, upper=3.0)
    prob.model.add_objective('drag')

    recorder = SqliteRecorder(os.path.join(OUT, 'openmdao_opt_cases.sql'))
    prob.driver.add_recorder(recorder)

    prob.setup()
    prob.run_driver()

    # read cases to build objective history
    try:
        from openmdao.api import CaseReader
        cr = CaseReader(os.path.join(OUT, 'openmdao_opt_cases.sql'))
        cases = cr.list_cases('driver')
        objs = []
        for c in cases:
            case = cr.get_case(c)
            try:
                objs.append(case.outputs['drag'])
            except Exception:
                pass
    except Exception:
        objs = []

    # plot objective history
    if objs:
        plt.figure()
        plt.plot(objs, marker='o')
        plt.title('OpenMDAO surrogate objective history')
        plt.xlabel('driver iteration')
        plt.ylabel('drag')
        plt.grid(True)
        plt.savefig(os.path.join(OUT, 'openmdao_objective_history.png'))
        plt.close()

    # save final latent mask if possible
    try:
        from Super_Aerostructural_Optimizer.generative.neuralfoil_surrogate import latent_to_alpha_mask
        best_latent = prob.get_val('latent')
        mask = latent_to_alpha_mask(np.asarray(best_latent))
        plt.imsave(os.path.join(OUT, 'openmdao_final_mask.png'), mask, cmap='gray')
    except Exception:
        pass

    return prob


if __name__ == '__main__':
    t0 = time.time()
    prob = train_and_optimize()
    print('Done in %.1fs' % (time.time() - t0))
