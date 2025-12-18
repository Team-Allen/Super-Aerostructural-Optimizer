from openmdao.api import ExplicitComponent
import numpy as np


class SurrogateComp(ExplicitComponent):
    """OpenMDAO component that wraps a PyTorch surrogate model.

    The `model` attribute should be set to a callable taking a numpy array
    (latent,) and returning a scalar drag prediction.
    """

    def initialize(self):
        self.options.declare('latent_dim', types=int, default=6)

    def setup(self):
        ld = self.options['latent_dim']
        self.add_input('latent', val=np.zeros(ld))
        self.add_output('drag', val=1.0)
        # use FD for now (surrogate may be non-jittable here)
        self.declare_partials('drag', 'latent', method='fd')

    def compute(self, inputs, outputs):
        latent = np.asarray(inputs['latent'], dtype=np.float32)
        model = getattr(self, 'model', None)
        if model is None:
            # fallback: simple quadratic objective
            outputs['drag'] = float(np.sum(latent ** 2))
        else:
            try:
                outputs['drag'] = float(model(latent))
            except Exception:
                outputs['drag'] = float(np.sum(latent ** 2))
