"""
Optimizer wrapper for the MDO pipeline.

Supports:
- scipy L-BFGS-B (gradient-based, fast, local)
- scipy SLSQP (gradient-based, handles constraints)
- scipy differential_evolution (gradient-free, global)
- pyOptSparse interface (SNOPT, IPOPT — when available)
"""

import numpy as np
from typing import Optional
import logging
import time

from .mdo_problem import MDOProblem, MDOResult, MDOProblemSetup

logger = logging.getLogger(__name__)


class MDOOptimizer:
    """Unified optimizer interface for the MDO pipeline."""

    def __init__(
        self,
        method: str = "L-BFGS-B",
        max_iterations: int = 200,
        ftol: float = 1e-8,
        gtol: float = 1e-6,
        verbose: bool = True,
    ):
        self.method = method
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.gtol = gtol
        self.verbose = verbose

    def optimize(self, problem: MDOProblem) -> MDOResult:
        """Run the optimization.

        Args:
            problem: configured MDOProblem instance

        Returns:
            MDOResult with optimal design and performance
        """
        setup = problem.setup
        dvs = setup.create_design_variables()

        # Initial design vector
        x0 = np.array([dv.value for dv in dvs])
        bounds = [(dv.lower, dv.upper) for dv in dvs]

        logger.info(f"Starting optimization with {self.method}")
        logger.info(f"  Design variables: {len(x0)}")
        logger.info(f"  Bounds: {[(b[0], b[1]) for b in bounds[:3]]}...")
        logger.info(f"  Max iterations: {self.max_iterations}")

        start_time = time.time()
        problem._start_time = start_time

        if self.method in ("L-BFGS-B", "SLSQP", "COBYLA"):
            result = self._optimize_scipy(problem, x0, bounds)
        elif self.method == "differential_evolution":
            result = self._optimize_de(problem, x0, bounds)
        elif self.method == "pyoptsparse":
            result = self._optimize_pyoptsparse(problem, x0, bounds, dvs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        wall_time = time.time() - start_time
        result.wall_time = wall_time
        result.n_function_evals = problem._eval_count

        logger.info(f"\nOptimization complete in {wall_time:.1f}s")
        logger.info(f"  Method: {self.method}")
        logger.info(f"  Function evaluations: {problem._eval_count}")
        logger.info(f"  Objective: {result.objective_value:.6f}")
        logger.info(f"  CL: {result.cl:.4f}")
        logger.info(f"  CD: {result.cd:.6f}")
        logger.info(f"  L/D: {result.ld_ratio:.1f}")
        logger.info(f"  Structural mass: {result.structural_mass:.1f} kg")
        logger.info(f"  Failure index: {result.failure_index:.3f}")

        return result

    def _optimize_scipy(self, problem, x0, bounds):
        """Scipy L-BFGS-B, SLSQP, or COBYLA optimizer."""
        from scipy.optimize import minimize

        callback_data = {'iter': 0}

        def callback(xk):
            callback_data['iter'] += 1
            if self.verbose and callback_data['iter'] % 10 == 0:
                logger.info(f"  Iteration {callback_data['iter']}")

        options = {
            'disp': self.verbose,
        }
        if self.method in ("L-BFGS-B", "SLSQP"):
            options['maxiter'] = self.max_iterations
            options['ftol'] = self.ftol
        if self.method == "L-BFGS-B":
            options['gtol'] = self.gtol
        if self.method == "COBYLA":
            # COBYLA uses its own tolerance; keep it loose for cheap coupling.
            options['tol'] = max(self.ftol, 1e-4)
            options['rhobeg'] = 0.1
            # In scipy >=1.16 COBYLA treats maxiter as the function-evaluation
            # budget.  Cap it at a modest multiple of the requested iteration
            # limit so production runs do not run away.
            options['maxiter'] = max(self.max_iterations * 10, 200)

        kwargs = {
            'fun': problem.objective,
            'x0': x0,
            'method': self.method,
            'callback': callback,
            'options': options,
        }
        if self.method in ("L-BFGS-B", "SLSQP", "COBYLA"):
            kwargs['bounds'] = bounds
        if self.method == "L-BFGS-B":
            kwargs['jac'] = problem.gradient

        result = minimize(**kwargs)

        mdo_result = problem.get_result(result.x)
        mdo_result.success = result.success
        mdo_result.n_iterations = getattr(result, 'nit', None)
        mdo_result.message = result.message
        return mdo_result

    def _optimize_de(self, problem, x0, bounds):
        """Scipy differential evolution (global optimizer)."""
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            problem.objective,
            bounds,
            x0=x0,
            maxiter=self.max_iterations,
            tol=self.ftol,
            seed=42,
            disp=self.verbose,
            polish=True,  # L-BFGS-B polish at the end
        )

        mdo_result = problem.get_result(result.x)
        mdo_result.success = result.success
        mdo_result.n_iterations = result.nit
        mdo_result.message = result.message
        return mdo_result

    def _optimize_pyoptsparse(self, problem, x0, bounds, dvs):
        """pyOptSparse interface for SNOPT/IPOPT."""
        try:
            from pyoptsparse import Optimization, SNOPT, IPOPT, SLSQP as pySLSQP
        except ImportError:
            logger.warning("pyOptSparse not available, falling back to scipy L-BFGS-B")
            return self._optimize_scipy(problem, x0, bounds)

        # Build pyOptSparse problem
        opt_prob = Optimization("Aerostructural MDO", problem.objective)

        for i, dv in enumerate(dvs):
            opt_prob.addVar(dv.name, value=dv.value, lower=dv.lower, upper=dv.upper)

        opt_prob.addObj('objective')

        # Try SNOPT first, then IPOPT, then SLSQP
        for OptimizerClass in [SNOPT, IPOPT, pySLSQP]:
            try:
                optimizer = OptimizerClass()
                break
            except Exception:
                continue

        sol = optimizer(opt_prob)

        x_opt = np.array([sol.variables[dv.name].value for dv in dvs])
        mdo_result = problem.get_result(x_opt)
        return mdo_result
