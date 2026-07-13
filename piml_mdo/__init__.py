"""
PIML-MDO Pipeline: Physics-Informed Machine Learning for Aerostructural MDO

A production-grade multidisciplinary design optimization pipeline combining:
- PINN-based aerodynamic solvers (2D RANS)
- NeuralFoil fast surrogate (0.002s/eval)
- Euler-Bernoulli beam structural solver with composite laminate properties
- Aerostructural coupling (load + displacement transfer)
- Gradient-based optimization (scipy L-BFGS-B / pyOptSparse)
- End-to-end differentiable pipeline orchestration

Architecture:
    PINN/NeuralFoil Aero → Load Transfer → Beam Structure → Displacement Transfer → Mesh Update
                    ↕                                          ↕
              Aero Gradients                            Structural Gradients
                    ↕                                          ↕
                         Coupled Adjoint → Optimizer
"""

__version__ = "0.1.0"
