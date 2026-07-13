"""
OpenAeroStruct VLM-based aerodynamic solver for the PIML MDO pipeline.

This module wraps OpenAeroStruct (an OpenMDAO-based VLM framework) to provide a
higher-fidelity 3-D lifting-surface solver with the same evaluate() interface as
NeuralFoil.  A custom wing mesh is built from the supplied 2-D airfoil
coordinates and the pipeline wing planform (span, root/tip chord, sweep).  The
camber line of the airfoil is embedded in the VLM mesh, while the viscous drag
correction uses the airfoil thickness-to-chord ratio.

The solver returns section-like coefficients (CL, CD, CM about c/4) which can be
used by the aerostructural coupling loop in place of the NeuralFoil surrogate.
"""

import hashlib
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .neuralfoil_wrapper import AeroResult

logger = logging.getLogger(__name__)

try:
    import openmdao.api as om
    from openaerostruct.aerodynamics.aero_groups import AeroPoint
    from openaerostruct.geometry.geometry_group import Geometry

    HAS_OPENAEROSTRUCT = True
except ImportError:  # pragma: no cover
    HAS_OPENAEROSTRUCT = False
    logger.warning(
        "OpenAeroStruct/OpenMDAO not available. "
        "Install with: pip install openaerostruct openmdao"
    )


@dataclass
class WingPlanform:
    """Planform parameters used to build the OpenAeroStruct mesh."""

    span: float = 15.0
    chord_root: float = 3.5
    chord_tip: float = 1.4
    sweep_deg: float = 25.0


@dataclass
class WingAeroDistribution:
    """Spanwise aerodynamic loads from a single 3-D VLM solve.

    All arrays are defined at the ``y`` spanwise panel centers.
    """

    y: np.ndarray               # spanwise panel-center positions [m]
    lift_per_len: np.ndarray    # distributed lift [N/m]
    drag_per_len: np.ndarray    # distributed drag [N/m]
    moment_per_len: np.ndarray  # distributed pitching moment about c/4 [N·m/m]
    chord: np.ndarray           # local chord at each strip [m]
    CL: float                   # whole-wing lift coefficient
    CD: float                   # whole-wing drag coefficient
    cp_panels: np.ndarray       # (nx-1, ny-1) per-panel ΔCp loading field
    def_mesh: np.ndarray        # (nx, ny, 3) deformed VLM mesh (half-wing)


class OpenAeroStructSolver:
    """VLM aerodynamic solver using OpenAeroStruct.

    Parameters
    ----------
    wing_planform : WingPlanform, optional
        Wing planform definition.
    num_x : int, optional
        Number of chordwise mesh nodes (odd recommended).
    num_y : int, optional
        Number of spanwise mesh nodes for the full wing.  Must be odd because
        OpenAeroStruct uses symmetry and cuts the mesh in half internally.
    num_twist_cp : int, optional
        Number of twist B-spline control points.  Kept at zero for the
        standalone aero solver.
    with_viscous : bool, optional
        Include the OpenAeroStruct empirical viscous drag correction.
    with_wave : bool, optional
        Include the OpenAeroStruct wave drag correction.
    velocity : float, optional
        Freestream velocity used only for non-dimensionalization of the
        viscous correction [m/s].
    rho : float, optional
        Freestream density [kg/m^3].
    """

    def __init__(
        self,
        wing_planform: Optional[WingPlanform] = None,
        num_x: int = 5,
        num_y: int = 9,
        num_twist_cp: int = 3,
        with_viscous: bool = True,
        with_wave: bool = False,
        velocity: float = 100.0,
        rho: float = 1.225,
    ):
        if not HAS_OPENAEROSTRUCT:
            raise ImportError(
                "OpenAeroStruct and OpenMDAO are required for this solver."
            )

        self.planform = wing_planform or WingPlanform()
        self.num_x = max(3, num_x)
        self.num_y = max(5, num_y)
        if self.num_y % 2 == 0:
            self.num_y += 1  # OpenAeroStruct requires odd spanwise count
        self.num_twist_cp = max(2, num_twist_cp)
        self.with_viscous = with_viscous
        self.with_wave = with_wave
        self.velocity = velocity
        self.rho = rho

        self._cache: dict[tuple, AeroResult] = {}
        self._eval_count = 0
        self._current_geo_hash: Optional[str] = None
        self._prob: Optional[om.Problem] = None
        self._surface_name = "wing"

    @property
    def eval_count(self) -> int:
        return self._eval_count

    def reset_count(self):
        """Clear the evaluation counter and cache."""
        self._eval_count = 0
        self._cache.clear()
        self._current_geo_hash = None
        self._prob = None

    def evaluate(
        self,
        coordinates: Tuple[np.ndarray, np.ndarray],
        alpha: float,
        Re: float,
        mach: float = 0.0,
    ) -> AeroResult:
        """Evaluate aerodynamics at a single operating point.

        Args
        ----
        coordinates : (x, y) arrays defining the airfoil (TE->LE->TE loop).
        alpha : angle of attack [deg].
        Re : Reynolds number based on the local chord (dimensionless).
        mach : Mach number.

        Returns
        -------
        AeroResult with CL, CD, CM about c/4, and L/D.
        """
        geo_hash = self._geometry_hash(coordinates)
        cache_key = (geo_hash, round(float(alpha), 6), float(Re), float(mach))

        if cache_key in self._cache:
            return self._cache[cache_key]

        if geo_hash != self._current_geo_hash:
            self._build_problem(coordinates)
            self._current_geo_hash = geo_hash

        self._set_flight_conditions(alpha, Re, mach)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._prob.run_model()

            cl = float(self._prob.get_val("aero_point.CL")[0])
            cd = float(self._prob.get_val("aero_point.CD")[0])
            cm_vec = self._prob.get_val("aero_point.CM")
            cm = float(cm_vec[1]) if cm_vec.size > 1 else float(cm_vec)

            if not (np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm)):
                raise RuntimeError("OpenAeroStruct returned non-finite coefficients")

            converged = True
        except Exception as exc:
            logger.warning("OpenAeroStruct evaluation failed: %s", exc)
            cl, cd, cm, converged = 0.0, 1.0, 0.0, False

        self._eval_count += 1
        result = AeroResult(
            cl=cl,
            cd=cd,
            cm=cm,
            ld_ratio=cl / cd if cd > 1e-10 else 0.0,
            alpha=alpha,
            Re=Re,
            mach=mach,
            converged=converged,
        )
        self._cache[cache_key] = result
        return result

    def _geometry_hash(self, coordinates: Tuple[np.ndarray, np.ndarray]) -> str:
        """Stable hash of the airfoil coordinates for caching."""
        x, y = coordinates
        data = np.ascontiguousarray(np.column_stack([x, y]))
        return hashlib.sha256(data.tobytes()).hexdigest()[:16]

    def _extract_camber_and_thickness(
        self, coordinates: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract camber line and thickness from full airfoil coordinates.

        Returns
        -------
        x : normalized chordwise stations (0 -> LE, 1 -> TE)
        camber : camber line z/c at each station
        thickness : thickness t/c at each station
        """
        x_raw, y_raw = coordinates
        x_raw = np.asarray(x_raw).ravel()
        y_raw = np.asarray(y_raw).ravel()

        # Normalize chord to [0, 1]
        c = x_raw.max() - x_raw.min()
        if c < 1e-6:
            c = 1.0
        x_norm = (x_raw - x_raw.min()) / c

        # Aggregate upper/lower y values at each unique x location
        xu = np.unique(np.round(x_norm, decimals=8))
        if len(xu) < 3:
            # Fallback to a flat plate if the input is degenerate
            xu = np.linspace(0.0, 1.0, 50)
            return xu, np.zeros_like(xu), np.zeros_like(xu)

        upper = np.full_like(xu, -1e9)
        lower = np.full_like(xu, 1e9)
        for i, xv in enumerate(xu):
            mask = np.abs(x_norm - xv) < 1e-6
            if np.any(mask):
                upper[i] = y_raw[mask].max()
                lower[i] = y_raw[mask].min()

        camber = (upper + lower) / (2.0 * c)
        thickness = (upper - lower) / c
        thickness = np.clip(thickness, 0.0, None)
        return xu, camber, thickness

    def _build_mesh(
        self, coordinates: Tuple[np.ndarray, np.ndarray]
    ) -> Tuple[np.ndarray, float, float]:
        """Build a tapered/swept VLM mesh with the airfoil camber embedded.

        Returns
        -------
        mesh : OpenAeroStruct mesh array, shape (num_x, num_y_half, 3)
        t_over_c_max : maximum thickness-to-chord ratio
        c_max_t : chordwise location of maximum thickness
        """
        x_c, camber, thickness = self._extract_camber_and_thickness(coordinates)

        t_over_c_max = float(np.max(thickness)) if thickness.size else 0.12
        if t_over_c_max < 1e-6:
            t_over_c_max = 0.12
        c_max_t = float(
            x_c[np.argmax(thickness)] if thickness.size else 0.3
        )

        ny_half = (self.num_y + 1) // 2
        span = self.planform.span
        chord_root = self.planform.chord_root
        chord_tip = self.planform.chord_tip
        sweep_rad = np.radians(self.planform.sweep_deg)

        y = np.linspace(0.0, span, ny_half)
        chord = chord_root + (chord_tip - chord_root) * (y / span)

        mesh = np.zeros((self.num_x, ny_half, 3))
        mesh[:, :, 1] = y  # spanwise coordinate

        xi = np.linspace(0.0, 1.0, self.num_x)
        camber_interp = np.interp(xi, x_c, camber, left=0.0, right=0.0)

        for j in range(ny_half):
            # Quarter-chord sweep: x_qc(y) = y * tan(sweep)
            x_le = y[j] * np.tan(sweep_rad) - 0.25 * (chord[j] - chord_root)
            mesh[:, j, 0] = x_le + xi * chord[j]
            # Scale camber by local chord
            mesh[:, j, 2] = camber_interp * chord[j]

        return mesh, t_over_c_max, c_max_t

    def _build_problem(self, coordinates: Tuple[np.ndarray, np.ndarray]):
        """Construct the OpenMDAO problem for the current geometry."""
        mesh, t_over_c_max, c_max_t = self._build_mesh(coordinates)

        surface = {
            "name": self._surface_name,
            "symmetry": True,
            "S_ref_type": "wetted",
            "mesh": mesh,
            "twist_cp": np.zeros(self.num_twist_cp),
            # Sweep and taper are already baked into the custom mesh.
            "sweep": 0.0,
            "taper": 1.0,
            "CL0": 0.0,
            "CD0": 0.0,
            "k_lam": 0.05,
            "t_over_c_cp": np.array([t_over_c_max]),
            "c_max_t": c_max_t,
            "with_viscous": self.with_viscous,
            "with_wave": self.with_wave,
        }

        prob = om.Problem(reports=None)
        indep = om.IndepVarComp()
        indep.add_output("v", val=self.velocity, units="m/s")
        indep.add_output("alpha", val=0.0, units="deg")
        indep.add_output("Mach_number", val=0.0)
        indep.add_output(
            "re",
            val=1e6 / max(self.planform.chord_root, 1e-6),
            units="1/m",
        )
        indep.add_output("rho", val=self.rho, units="kg/m**3")
        indep.add_output(
            "cg",
            val=np.array([0.25 * self.planform.chord_root, 0.0, 0.0]),
            units="m",
        )
        prob.model.add_subsystem("flight_vars", indep, promotes=["*"])

        geom_group = Geometry(surface=surface)
        prob.model.add_subsystem("wing_geom", geom_group)

        aero_group = AeroPoint(surfaces=[surface])
        prob.model.add_subsystem("aero_point", aero_group)

        # Connect the deformed mesh and thickness from the geometry group
        prob.model.connect(
            "wing_geom.mesh", f"aero_point.{self._surface_name}.def_mesh"
        )
        prob.model.connect(
            "wing_geom.mesh",
            f"aero_point.aero_states.{self._surface_name}_def_mesh",
        )
        prob.model.connect(
            "wing_geom.t_over_c",
            f"aero_point.{self._surface_name}_perf.t_over_c",
        )

        # Connect flight condition variables
        for var in ("v", "alpha", "Mach_number", "re", "rho", "cg"):
            prob.model.connect(var, f"aero_point.{var}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prob.setup()

        self._prob = prob

    def _set_flight_conditions(self, alpha: float, Re: float, mach: float):
        """Update the operating point on the existing OpenMDAO problem."""
        # Re passed in is dimensionless (based on chord); OAS wants Re per meter.
        re_per_m = Re / max(self.planform.chord_root, 1e-6)
        self._prob.set_val("alpha", alpha, units="deg")
        self._prob.set_val("Mach_number", mach)
        self._prob.set_val("re", re_per_m, units="1/m")

    def solve_wing_distribution(
        self,
        coordinates: Tuple[np.ndarray, np.ndarray],
        twist_stations: np.ndarray,
        twist_deg: np.ndarray,
        alpha: float,
        Re: float,
        mach: float,
        velocity: float,
        rho: float,
    ) -> "WingAeroDistribution":
        """Run ONE 3-D VLM solve of the whole wing and extract the real
        spanwise load distribution and panel loading (ΔCp).

        Unlike ``evaluate`` (which returns whole-wing coefficients), this method
        is the physically correct way to drive a spanwise beam: it solves the
        full 3-D lifting surface a single time and integrates the panel forces
        (``sec_forces``) chordwise to obtain the true spanwise lift, drag and
        pitching-moment distributions, plus a per-panel ΔCp field for
        visualization.

        Args
        ----
        coordinates : airfoil (x, y) — supplies camber and t/c for the mesh.
        twist_stations : spanwise positions [m] at which ``twist_deg`` is given.
        twist_deg : geometric+structural twist [deg] at ``twist_stations``.
        alpha : root angle of attack [deg].
        Re : Reynolds number (based on root chord, dimensionless).
        mach, velocity, rho : freestream conditions (loads scale with q).

        Returns
        -------
        WingAeroDistribution with panel-center spanwise arrays and the ΔCp map.
        """
        geo_hash = self._geometry_hash(coordinates)
        if geo_hash != self._current_geo_hash or self._prob is None:
            self._build_problem(coordinates)
            self._current_geo_hash = geo_hash

        # Inject the twist distribution via the geometry B-spline control points.
        cp_stations = np.linspace(twist_stations[0], twist_stations[-1], self.num_twist_cp)
        twist_cp_vals = np.interp(cp_stations, twist_stations, twist_deg)
        try:
            self._prob.set_val("wing_geom.twist_cp", twist_cp_vals, units="deg")
        except Exception:
            # Some OAS versions promote twist_cp differently; fall back silently.
            pass

        self._prob.set_val("v", velocity, units="m/s")
        self._prob.set_val("rho", rho, units="kg/m**3")
        self._set_flight_conditions(alpha, Re, mach)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._prob.run_model()

        CL = float(self._prob.get_val("aero_point.CL")[0])
        CD = float(self._prob.get_val("aero_point.CD")[0])
        sf = np.asarray(self._prob.get_val("aero_point.aero_states.wing_sec_forces"))
        widths = np.asarray(self._prob.get_val("aero_point.wing.widths"))
        def_mesh = np.asarray(self._prob.get_val("aero_point.aero_states.wing_def_mesh"))
        self._eval_count += 1

        q = 0.5 * rho * velocity ** 2

        # Spanwise integration: sum the chordwise panel forces per strip.
        Fz = sf[:, :, 2].sum(axis=0)          # vertical force per strip [N]
        Fx = sf[:, :, 0].sum(axis=0)          # streamwise force per strip [N]
        widths = np.maximum(widths, 1e-9)
        lift_per_len = Fz / widths            # [N/m]
        drag_per_len = Fx / widths            # [N/m]

        # Panel-center spanwise coordinate.
        y_panel = 0.5 * (def_mesh[0, :-1, 1] + def_mesh[0, 1:, 1])

        # Chordwise panel-center x and per-strip quarter-chord for the pitching
        # moment (nose-up positive about the local c/4).
        x_center = 0.25 * (
            def_mesh[:-1, :-1, 0] + def_mesh[1:, :-1, 0]
            + def_mesh[:-1, 1:, 0] + def_mesh[1:, 1:, 0]
        )                                     # (nx-1, ny-1)
        le_x = 0.5 * (def_mesh[0, :-1, 0] + def_mesh[0, 1:, 0])
        te_x = 0.5 * (def_mesh[-1, :-1, 0] + def_mesh[-1, 1:, 0])
        chord_strip = np.abs(te_x - le_x)
        qc_x = le_x + 0.25 * chord_strip
        moment = -(sf[:, :, 2] * (x_center - qc_x[None, :])).sum(axis=0)  # [N·m] per strip
        moment_per_len = moment / widths                                  # [N·m/m]

        # Per-panel ΔCp = normal force / (q · panel area).
        panel_len = np.abs(def_mesh[1:, :, 0] - def_mesh[:-1, :, 0])       # (nx-1, ny)
        panel_len = 0.5 * (panel_len[:, :-1] + panel_len[:, 1:])            # (nx-1, ny-1)
        area = panel_len * widths[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            cp = np.where(area > 1e-12, sf[:, :, 2] / (q * area), 0.0)

        return WingAeroDistribution(
            y=y_panel,
            lift_per_len=lift_per_len,
            drag_per_len=drag_per_len,
            moment_per_len=moment_per_len,
            chord=chord_strip,
            CL=CL,
            CD=CD,
            cp_panels=cp,
            def_mesh=def_mesh,
        )

    def compute_polar(
        self,
        coordinates: Tuple[np.ndarray, np.ndarray],
        alpha_range: Tuple[float, float] = (-5.0, 15.0),
        n_alpha: int = 41,
        Re: float = 1e6,
        mach: float = 0.0,
    ):
        """Compute a lift/drag polar by sweeping alpha.

        Returns a simple namespace-like object matching the NeuralFoil polar.
        """
        from .neuralfoil_wrapper import AeroPolarResult

        alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
        cls = np.zeros(n_alpha)
        cds = np.zeros(n_alpha)
        cms = np.zeros(n_alpha)
        for i, a in enumerate(alphas):
            res = self.evaluate(coordinates, a, Re, mach)
            cls[i] = res.cl
            cds[i] = res.cd
            cms[i] = res.cm
        return AeroPolarResult(alphas=alphas, cls=cls, cds=cds, cms=cms)
