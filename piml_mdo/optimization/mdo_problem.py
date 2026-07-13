"""
MDO Problem Definition for Aerostructural Wing Optimization.

Defines the design variables, objectives, and constraints for the
coupled aerostructural optimization problem.

Design Variables:
    - Airfoil shape (CST weights): 12 params
    - Wing twist distribution: n_twist params
    - Structural sizing (skin thickness scaling): n_struct params
    - Angle of attack: 1 param

Objective:
    - Minimize fuel burn ∝ (W_struct * CD / CL) at cruise
    OR
    - Maximize L/D at fixed structural weight constraint

Constraints:
    - CL = CL_required (trim)
    - Failure index < 1.0 (structural strength)
    - Tip deflection < limit
    - Thickness > minimum (manufacturing)
    - Moment coefficient bounds (stability)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import logging
import time

from ..structures.composite_properties import Laminate, CFRP_IM7_8552

logger = logging.getLogger(__name__)


@dataclass
class DesignVariable:
    """Single design variable with bounds."""
    name: str
    value: float
    lower: float
    upper: float
    scale: float = 1.0  # Scaling for optimizer

    @property
    def normalized(self) -> float:
        return (self.value - self.lower) / (self.upper - self.lower)


@dataclass
class Constraint:
    """Optimization constraint."""
    name: str
    type: str  # 'eq' or 'ineq'
    lower: Optional[float] = None
    upper: Optional[float] = None


@dataclass
class MDOProblemSetup:
    """Complete MDO problem setup."""
    # Airfoil CST parameters
    n_cst_upper: int = 6
    n_cst_lower: int = 6

    # Wing twist stations
    n_twist: int = 5

    # Structural sizing stations
    n_struct_sizing: int = 5

    # Composite laminate stackup design
    n_laminate_stations: int = 5
    n_ply_angles: int = 4
    ply_angles: tuple[float, ...] = (0.0, 45.0, -45.0, 90.0)
    optimize_layup: bool = True
    baseline_ply_counts: Optional[dict[float, float]] = None

    # Include alpha as design variable
    optimize_alpha: bool = True

    # Target conditions
    cl_target: float = 0.5       # Required CL (trim constraint)
    re_cruise: float = 5e6       # Cruise Reynolds number
    mach_cruise: float = 0.3     # Cruise Mach number

    # Constraints
    max_failure_index: float = 0.8   # Safety margin
    max_tip_deflection: float = 2.0  # [m]
    min_thickness: float = 0.08      # Minimum t/c
    max_thickness: float = 0.20      # Maximum t/c

    # Objective weights (for multi-objective)
    weight_drag: float = 1.0
    weight_mass: float = 0.1
    weight_ld: float = 0.0  # If > 0, maximize L/D instead

    def __post_init__(self):
        """Ensure ply_angles is consistent with n_ply_angles."""
        if len(self.ply_angles) != self.n_ply_angles:
            base = [0.0, 45.0, -45.0, 90.0]
            n = self.n_ply_angles
            self.ply_angles = tuple((base * ((n // len(base)) + 1))[:n])

    @property
    def n_design_vars(self) -> int:
        n = self.n_cst_upper + self.n_cst_lower + self.n_twist + self.n_struct_sizing
        if self.optimize_layup:
            n += self.n_laminate_stations * self.n_ply_angles
        if self.optimize_alpha:
            n += 1
        return n

    def create_design_variables(self) -> list[DesignVariable]:
        """Create the full set of design variables with bounds."""
        dvs = []

        # CST upper surface weights
        for i in range(self.n_cst_upper):
            dvs.append(DesignVariable(
                name=f'cst_upper_{i}',
                value=0.15 + 0.02 * i,  # Initial: reasonable airfoil shape
                lower=-0.5, upper=0.8,
                scale=1.0,
            ))

        # CST lower surface weights
        for i in range(self.n_cst_lower):
            dvs.append(DesignVariable(
                name=f'cst_lower_{i}',
                value=-0.15 + 0.01 * i,
                lower=-0.8, upper=0.5,
                scale=1.0,
            ))

        # Wing twist at stations
        for i in range(self.n_twist):
            dvs.append(DesignVariable(
                name=f'twist_{i}',
                value=-2.0 + i * 0.5,  # Mild washout
                lower=-10.0, upper=10.0,
                scale=0.1,
            ))

        # Structural sizing (thickness multiplier). Start slightly above unity
        # so the optimizer begins near the strength-feasible region rather than
        # having to discover the whole way there from an under-sized wing.
        for i in range(self.n_struct_sizing):
            dvs.append(DesignVariable(
                name=f'struct_scale_{i}',
                value=1.3,
                lower=0.3, upper=3.0,
                scale=1.0,
            ))

        # Composite laminate ply counts per station / per angle
        # These are integer-like continuous variables that are rounded inside
        # the objective.  Counts define one half of a symmetric laminate; the
        # full stack is mirrored about the midplane.
        if self.optimize_layup:
            default_counts = self.baseline_ply_counts or {a: 1.0 for a in self.ply_angles}
            for i in range(self.n_laminate_stations):
                for angle in self.ply_angles:
                    init_count = float(default_counts.get(angle, 1.0))
                    dvs.append(DesignVariable(
                        name=f'ply_count_{int(angle)}_{i}',
                        value=init_count,
                        lower=0.0, upper=16.0,
                        scale=1.0,
                    ))

        # Angle of attack
        if self.optimize_alpha:
            dvs.append(DesignVariable(
                name='alpha',
                value=3.0,
                lower=-2.0, upper=12.0,
                scale=0.1,
            ))

        return dvs

    def design_vector_to_dict(self, x: np.ndarray) -> dict:
        """Unpack flat design vector into named groups."""
        idx = 0
        result = {}

        result['cst_upper'] = x[idx:idx + self.n_cst_upper]
        idx += self.n_cst_upper

        result['cst_lower'] = x[idx:idx + self.n_cst_lower]
        idx += self.n_cst_lower

        result['twist'] = x[idx:idx + self.n_twist]
        idx += self.n_twist

        result['struct_scale'] = x[idx:idx + self.n_struct_sizing]
        idx += self.n_struct_sizing

        if self.optimize_layup:
            n_lam = self.n_laminate_stations * self.n_ply_angles
            result['ply_counts'] = x[idx:idx + n_lam].reshape(
                (self.n_laminate_stations, self.n_ply_angles)
            )
            idx += n_lam

        if self.optimize_alpha:
            result['alpha'] = x[idx]
            idx += 1

        return result

    def dict_to_design_vector(self, d: dict) -> np.ndarray:
        """Pack named groups into flat design vector."""
        parts = [
            d['cst_upper'],
            d['cst_lower'],
            d['twist'],
            d['struct_scale'],
        ]
        if self.optimize_layup:
            parts.append(d['ply_counts'].flatten())
        if self.optimize_alpha:
            parts.append(np.array([d['alpha']]))
        return np.concatenate(parts)


@dataclass
class MDOResult:
    """Result from MDO optimization."""
    success: bool
    x_optimal: np.ndarray
    design_dict: dict
    objective_value: float
    cl: float
    cd: float
    ld_ratio: float
    structural_mass: float
    failure_index: float
    tip_deflection: float
    n_iterations: int
    n_function_evals: int
    wall_time: float
    history: list[dict] = field(default_factory=list)
    message: str = ""


class MDOProblem:
    """Aerostructural MDO problem that couples aero + structure + optimizer.

    This is the core of the pipeline: it defines the objective function
    that the optimizer calls, which internally runs the full
    aerostructural coupling loop.
    """

    def __init__(
        self,
        setup: MDOProblemSetup,
        aero_solver,
        beam_solver,
        coupler,
        airfoil_geometry_class,
        wing_structure,
        flight_condition=None,
    ):
        self.setup = setup
        self.aero_solver = aero_solver
        self.beam_solver = beam_solver
        self.coupler = coupler
        self.airfoil_geometry_class = airfoil_geometry_class
        self.wing = wing_structure
        self.flight_condition = flight_condition

        self._eval_count = 0
        self._best_objective = np.inf
        self._history = []
        self._start_time = None
        self._cache = {}
        self._cache_hits = 0

    def _apply_design_to_wing(self, d: dict) -> float:
        """Apply structural sizing and laminate DVs to the wing.

        Returns a manufacturing/symmetry penalty for invalid layups.
        """
        beam_stations = self.wing.spanwise_stations()

        # Existing thickness-scale design variables
        struct_stations = np.linspace(0, self.wing.span, self.setup.n_struct_sizing)
        struct_scales = np.interp(beam_stations, struct_stations, d['struct_scale'])
        self.wing.set_skin_thickness_scales(struct_scales)

        penalty = 0.0
        if not self.setup.optimize_layup:
            self.wing.set_skin_laminates(None)
            return penalty

        base_material = (
            self.wing.skin_laminate.material
            if self.wing.skin_laminate is not None
            else CFRP_IM7_8552
        )

        ply_counts = np.asarray(d['ply_counts'])
        n_stations = self.setup.n_laminate_stations
        n_angles = self.setup.n_ply_angles
        if ply_counts.shape != (n_stations, n_angles):
            raise ValueError(
                f"ply_counts shape {ply_counts.shape} != "
                f"({n_stations}, {n_angles})"
            )

        # Interpolate ply counts from design stations to beam nodes
        laminate_stations = np.linspace(0, self.wing.span, n_stations)
        node_counts = np.zeros((len(beam_stations), n_angles))
        for j in range(n_angles):
            node_counts[:, j] = np.interp(
                beam_stations, laminate_stations, ply_counts[:, j]
            )

        # Round to integer plies and clip to design bounds
        node_counts = np.clip(np.rint(node_counts), 0.0, 16.0).astype(int)

        # Enforce a minimum number of plies (manufacturing / matrix singularity)
        min_total = 2
        total_counts = np.sum(node_counts, axis=1)
        insufficient = total_counts < min_total
        if np.any(insufficient):
            penalty += 100.0 * np.sum((min_total - total_counts[insufficient]) ** 2)
            for i in np.where(insufficient)[0]:
                node_counts[i, :] = np.maximum(node_counts[i, :], 1)

        # Build a symmetric laminate for each node.  The ply counts define one
        # half of the stack; symmetric=True mirrors it about the midplane.
        laminates = []
        for i in range(len(beam_stations)):
            angles_half = []
            for j, angle in enumerate(self.setup.ply_angles):
                angles_half.extend([angle] * int(node_counts[i, j]))
            laminates.append(
                Laminate(
                    material=base_material,
                    angles=angles_half,
                    symmetric=True,
                )
            )
        self.wing.set_skin_laminates(laminates)

        return penalty

    def objective(self, x: np.ndarray) -> float:
        """Evaluate the MDO objective function.

        This is what the optimizer calls. It:
        1. Unpacks design variables
        2. Creates airfoil geometry
        3. Runs aerostructural coupling
        4. Evaluates objective + constraints
        5. Returns penalized objective

        Returns:
            objective value (lower is better)
        """
        if self._start_time is None:
            self._start_time = time.time()

        # Cache design-vector evaluations (common during line searches and when
        # the optimizer revisits nearly-identical points).  Rounding collapses
        # numerically equivalent vectors so we do not re-run the coupling.
        key = np.round(x, 8).tobytes()
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._eval_count += 1
        d = self.setup.design_vector_to_dict(x)

        try:
            # Create airfoil from CST parameters
            from ..aero.airfoil_geometry import AirfoilGeometry
            geom = AirfoilGeometry(
                upper_weights=d['cst_upper'],
                lower_weights=d['cst_lower'],
            )
            coords = geom.full_coordinates(150)

            # Twist distribution (interpolate to beam stations)
            beam_stations = self.wing.spanwise_stations()
            twist_stations = np.linspace(0, self.wing.span, self.setup.n_twist)
            twist_interp = np.interp(beam_stations, twist_stations, d['twist'])

            # Apply structural sizing and laminate design variables
            layup_penalty = self._apply_design_to_wing(d)

            # Flight condition
            from ..coupling.load_transfer import FlightCondition
            alpha = d.get('alpha', self.setup.cl_target * 10)  # rough estimate if not optimizing
            if self.flight_condition is not None:
                flight = FlightCondition(
                    velocity=self.flight_condition.velocity,
                    altitude=self.flight_condition.altitude,
                    alpha=float(alpha),
                    load_factor=self.flight_condition.load_factor,
                )
            else:
                flight = FlightCondition(
                    velocity=100.0,
                    altitude=3000.0,
                    alpha=float(alpha),
                    load_factor=1.0,
                )

            # Run aerostructural coupling
            coupling_result = self.coupler.solve(
                airfoil_coords=coords,
                initial_twist=twist_interp,
                flight_condition=flight,
            )

            aero = coupling_result['aero']
            struct = coupling_result['structure']

            # Integrated aerodynamic coefficients over the semi-span
            q = flight.dynamic_pressure
            wing_area = np.trapz(self.wing.chord_distribution(), self.wing.spanwise_stations())
            total_lift = np.trapz(aero['lift'], self.wing.spanwise_stations())
            total_drag = np.trapz(aero['drag'], self.wing.spanwise_stations())
            cl_mean = float(total_lift / (q * wing_area)) if wing_area > 1e-10 else 0.0
            cd_mean = float(total_drag / (q * wing_area)) if wing_area > 1e-10 else 0.0
            ld_ratio = cl_mean / cd_mean if cd_mean > 1e-10 else 0.0

            # Objective: minimize drag coefficient (or fuel burn proxy)
            obj = cd_mean * self.setup.weight_drag

            # Add structural mass penalty
            if self.setup.weight_mass > 0:
                obj += self.setup.weight_mass * struct.total_mass / 1000.0  # normalize

            # If maximizing L/D
            if self.setup.weight_ld > 0:
                obj -= self.setup.weight_ld * ld_ratio / 100.0

            # Constraint penalties (exterior penalty method)
            penalty = layup_penalty
            # Weight the strength/deflection penalties strongly so the gradient-
            # free optimizer is driven to feasibility even under the higher 3-D
            # VLM loads (a weak weight lets COBYLA park at an infeasible design).
            penalty_weight = 1000.0
            # CL target is the primary flight constraint; weight it heavily so
            # the optimizer cannot collapse lift to avoid structural penalties.
            cl_penalty_weight = 10000.0

            # CL target constraint
            cl_error = abs(cl_mean - self.setup.cl_target)
            if cl_error > 0.01:
                penalty += cl_penalty_weight * cl_error**2

            # Structural failure constraint
            if struct.failure_index > self.setup.max_failure_index:
                penalty += penalty_weight * (struct.failure_index - self.setup.max_failure_index)**2

            # Tip deflection constraint
            tip_def = abs(struct.tip_deflection)
            if tip_def > self.setup.max_tip_deflection:
                penalty += penalty_weight * (tip_def - self.setup.max_tip_deflection)**2

            # Thickness constraint
            tc = geom.max_thickness()
            if tc < self.setup.min_thickness:
                penalty += penalty_weight * (self.setup.min_thickness - tc)**2
            if tc > self.setup.max_thickness:
                penalty += penalty_weight * (tc - self.setup.max_thickness)**2

            total = obj + penalty

            # Track history
            record = {
                'eval': self._eval_count,
                'objective': total,
                'cl': cl_mean,
                'cd': cd_mean,
                'ld': ld_ratio,
                'mass': struct.total_mass,
                'failure_index': struct.failure_index,
                'tip_deflection': struct.tip_deflection,
                'penalty': penalty,
                'time': time.time() - self._start_time,
            }
            self._history.append(record)

            if total < self._best_objective:
                self._best_objective = total
                logger.info(
                    f"Eval {self._eval_count}: obj={total:.6f} "
                    f"CL={cl_mean:.4f} CD={cd_mean:.6f} L/D={ld_ratio:.1f} "
                    f"mass={struct.total_mass:.1f}kg FI={struct.failure_index:.3f} "
                    f"tip={struct.tip_deflection:.3f}m"
                )

            self._cache[key] = total
            return total

        except Exception as e:
            logger.warning(f"Eval {self._eval_count} failed: {e}")
            return 1e6  # Large penalty for failed evaluations

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient via finite differences (central differences)."""
        n = len(x)
        grad = np.zeros(n)
        h = 1e-5

        f0 = self.objective(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            f_plus = self.objective(x_plus)
            grad[i] = (f_plus - f0) / h

        return grad

    def get_result(self, x_optimal: np.ndarray) -> MDOResult:
        """Build final MDOResult from optimal design."""
        d = self.setup.design_vector_to_dict(x_optimal)

        # Final evaluation
        from ..aero.airfoil_geometry import AirfoilGeometry
        geom = AirfoilGeometry(
            upper_weights=d['cst_upper'],
            lower_weights=d['cst_lower'],
        )
        coords = geom.full_coordinates(150)

        beam_stations = self.wing.spanwise_stations()
        twist_stations = np.linspace(0, self.wing.span, self.setup.n_twist)
        twist_interp = np.interp(beam_stations, twist_stations, d['twist'])

        # Apply structural sizing and laminate design variables
        self._apply_design_to_wing(d)

        from ..coupling.load_transfer import FlightCondition
        if self.flight_condition is not None:
            flight = FlightCondition(
                velocity=self.flight_condition.velocity,
                altitude=self.flight_condition.altitude,
                alpha=float(d.get('alpha', 3.0)),
                load_factor=self.flight_condition.load_factor,
            )
        else:
            flight = FlightCondition(
                velocity=100.0, altitude=3000.0,
                alpha=float(d.get('alpha', 3.0)),
                load_factor=1.0,
            )

        coupling_result = self.coupler.solve(coords, twist_interp, flight)
        aero = coupling_result['aero']
        struct = coupling_result['structure']

        q = flight.dynamic_pressure
        wing_area = np.trapz(self.wing.chord_distribution(), self.wing.spanwise_stations())
        total_lift = np.trapz(aero['lift'], self.wing.spanwise_stations())
        total_drag = np.trapz(aero['drag'], self.wing.spanwise_stations())
        cl_mean = float(total_lift / (q * wing_area)) if wing_area > 1e-10 else 0.0
        cd_mean = float(total_drag / (q * wing_area)) if wing_area > 1e-10 else 0.0

        wall_time = time.time() - self._start_time if self._start_time else 0.0

        return MDOResult(
            success=True,
            x_optimal=x_optimal,
            design_dict=d,
            objective_value=self._best_objective,
            cl=cl_mean,
            cd=cd_mean,
            ld_ratio=cl_mean / cd_mean if cd_mean > 1e-10 else 0.0,
            structural_mass=struct.total_mass,
            failure_index=struct.failure_index,
            tip_deflection=struct.tip_deflection,
            n_iterations=0,  # Set by optimizer
            n_function_evals=self._eval_count,
            wall_time=wall_time,
            history=self._history,
        )
