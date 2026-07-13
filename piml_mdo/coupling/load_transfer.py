"""
Aerostructural coupling: aerodynamic loads → structural loads.

Converts 2D airfoil aerodynamic coefficients (CL, CD, CM per section)
into distributed forces and moments along the wing span for the beam solver.

The coupling accounts for:
- Spanwise lift distribution (elliptic + twist correction)
- Pitching moment distribution
- Wing sweep effects on load direction
- Dynamic pressure variation (if needed)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class FlightCondition:
    """Flight condition for load computation."""
    velocity: float      # Freestream velocity [m/s]
    altitude: float      # Altitude [m]
    alpha: float         # Angle of attack [deg]
    load_factor: float = 1.0  # Load factor (n)

    @property
    def mach(self) -> float:
        """Mach number at altitude."""
        T = 288.15 - 0.0065 * self.altitude  # ISA temperature
        a = np.sqrt(1.4 * 287.058 * max(T, 200.0))
        return self.velocity / a

    @property
    def density(self) -> float:
        """Air density at altitude [kg/m³] (ISA model)."""
        T = 288.15 - 0.0065 * self.altitude
        p = 101325.0 * (T / 288.15) ** 5.2561
        return p / (287.058 * T)

    @property
    def dynamic_pressure(self) -> float:
        """Dynamic pressure q = 0.5 * rho * V² [Pa]."""
        return 0.5 * self.density * self.velocity**2

    @property
    def reynolds(self) -> float:
        """Reynolds number per unit chord."""
        T = 288.15 - 0.0065 * self.altitude
        mu = 1.458e-6 * T**1.5 / (T + 110.4)  # Sutherland's law
        return self.density * self.velocity / mu


class LoadTransfer:
    """Transfers aerodynamic section data to spanwise beam loads."""

    def __init__(self, wing_span: float, n_stations: int):
        self.wing_span = wing_span
        self.n_stations = n_stations
        self.y = np.linspace(0, wing_span, n_stations)

    def elliptic_lift_distribution(
        self,
        total_lift: float,
    ) -> np.ndarray:
        """Classical elliptic lift distribution (optimal for minimum induced drag).

        L'(y) = L0 * sqrt(1 - (2y/b)²)

        Args:
            total_lift: Total lift force on semi-span [N]

        Returns:
            lift_per_length: [n_stations] distributed lift [N/m]
        """
        eta = self.y / self.wing_span
        # Elliptic distribution: L'(eta) = L0 * sqrt(1 - eta²)
        L_dist = np.sqrt(np.maximum(1.0 - eta**2, 0.0))

        # Normalize to match total lift
        integral = np.trapz(L_dist, self.y)
        if integral > 1e-10:
            L_dist *= total_lift / integral

        return L_dist

    def section_cl_to_lift_distribution(
        self,
        section_cls: np.ndarray,
        chord_distribution: np.ndarray,
        flight_condition: FlightCondition,
    ) -> np.ndarray:
        """Convert section CL values to distributed lift.

        L'(y) = q * c(y) * Cl(y)

        Args:
            section_cls: [n_stations] section lift coefficients
            chord_distribution: [n_stations] chord at each station [m]
            flight_condition: flight condition

        Returns:
            lift_per_length: [n_stations] distributed lift [N/m]
        """
        q = flight_condition.dynamic_pressure
        return q * chord_distribution * section_cls

    def section_cm_to_moment_distribution(
        self,
        section_cms: np.ndarray,
        chord_distribution: np.ndarray,
        flight_condition: FlightCondition,
    ) -> np.ndarray:
        """Convert section CM values to distributed pitching moment.

        M'(y) = q * c(y)² * Cm(y)

        Args:
            section_cms: [n_stations] section moment coefficients
            chord_distribution: [n_stations] chord [m]
            flight_condition: flight condition

        Returns:
            moment_per_length: [n_stations] distributed moment [N·m/m]
        """
        q = flight_condition.dynamic_pressure
        return q * chord_distribution**2 * section_cms

    def compute_section_aero(
        self,
        aero_solver,
        airfoil_coords: tuple[np.ndarray, np.ndarray],
        chord_distribution: np.ndarray,
        twist_distribution: np.ndarray,
        flight_condition: FlightCondition,
    ) -> dict[str, np.ndarray]:
        """Evaluate aerodynamics at each spanwise section.

        Calls the aero solver (NeuralFoil or PINN) at each section with
        local angle of attack = flight alpha + geometric twist. An induced-drag
        correction is added to the drag distribution via lifting-line theory.

        Args:
            aero_solver: solver with evaluate(coords, alpha, Re, mach) method
            airfoil_coords: (x, y) full airfoil coordinates
            chord_distribution: [n_stations] chord [m]
            twist_distribution: [n_stations] geometric twist [deg]
            flight_condition: flight condition

        Returns:
            dict with 'cl', 'cd', 'cm', 'lift', 'drag', 'moment' arrays [n_stations]

        Note:
            This uses the raw 2-D section drag from the aero solver and adds
            an induced-drag estimate from lifting-line theory. It does not
            re-converge the local angle of attack for the induced downwash;
            that would require an additional aero evaluation per station and
            is left as a documented limitation for this simplified coupling.
        """
        n = self.n_stations
        cls = np.zeros(n)
        cds = np.zeros(n)
        cms = np.zeros(n)

        for i in range(n):
            # Local angle of attack (geometric only; induced downwash is
            # approximated via the induced-drag term below)
            alpha_local = flight_condition.alpha + twist_distribution[i]

            # Local Reynolds number
            Re_local = flight_condition.reynolds * chord_distribution[i]

            result = aero_solver.evaluate(
                airfoil_coords,
                alpha=alpha_local,
                Re=Re_local,
                mach=flight_condition.mach,
            )
            cls[i] = result.cl
            cds[i] = result.cd
            cms[i] = result.cm

        # Convert to distributed loads
        q = flight_condition.dynamic_pressure
        lift = q * chord_distribution * cls
        moment = q * chord_distribution**2 * cms

        # Induced-drag estimate from lifting-line theory:
        #   cd_i = cl^2 / (pi * AR)
        # This captures the spanwise induced drag without a second aero solve.
        mean_chord = 0.5 * (chord_distribution[0] + chord_distribution[-1])
        full_span = 2.0 * self.wing_span
        AR = full_span / mean_chord if mean_chord > 1e-10 else 10.0
        cd_induced = cls**2 / (np.pi * max(AR, 1.0))
        drag = q * chord_distribution * (cds + cd_induced)

        return {
            'cl': cls,
            'cd': cds,
            'cm': cms,
            'lift': lift,
            'drag': drag,
            'moment': moment,
            'AR': float(AR),
            'cd_induced': cd_induced,
        }

    def compute_total_forces(
        self,
        lift_distribution: np.ndarray,
        drag_distribution: np.ndarray,
    ) -> dict[str, float]:
        """Integrate spanwise distributions to get total forces."""
        total_lift = float(np.trapz(lift_distribution, self.y))
        total_drag = float(np.trapz(drag_distribution, self.y))

        return {
            'total_lift': total_lift,
            'total_drag': total_drag,
            'L/D': total_lift / total_drag if abs(total_drag) > 1e-10 else 0.0,
        }


class DisplacementTransfer:
    """Transfers structural displacements back to aerodynamic mesh.

    After the structure deforms under load, the aerodynamic shape changes:
    - Deflection changes the effective dihedral
    - Twist changes the local angle of attack
    - This feeds back into the aero solver for the next coupling iteration
    """

    def __init__(self, n_stations: int):
        self.n_stations = n_stations

    def update_twist(
        self,
        initial_twist: np.ndarray,
        structural_twist: np.ndarray,
    ) -> np.ndarray:
        """Update aerodynamic twist with structural deformation.

        Args:
            initial_twist: [n_stations] initial geometric twist [deg]
            structural_twist: [n_stations] twist from beam solver [rad]

        Returns:
            updated_twist: [n_stations] total twist [deg]
        """
        return initial_twist + np.degrees(structural_twist)

    def update_effective_alpha(
        self,
        base_alpha: float,
        deflection: np.ndarray,
        span_stations: np.ndarray,
    ) -> np.ndarray:
        """Compute effective alpha change from wing deflection (dihedral effect).

        Small deflection slope dw/dy changes effective angle of attack.
        """
        dw_dy = np.gradient(deflection, span_stations)
        # Deflection slope reduces effective alpha
        delta_alpha = -np.degrees(np.arctan(dw_dy))
        return delta_alpha


class AerostructuralCoupler:
    """Manages the aerostructural coupling iteration (Gauss-Seidel).

    Iterates between:
    1. Aero: compute loads from current shape + deformation
    2. Structure: compute deformation from current loads
    3. Update: modify aero shape with structural deformation

    Until convergence (displacement change < tolerance).
    """

    def __init__(
        self,
        aero_solver,
        beam_solver,
        wing_structure,
        n_stations: int,
        max_iterations: int = 20,
        tolerance: float = 1e-4,
        relaxation: float = 0.3,
    ):
        self.aero_solver = aero_solver
        self.beam_solver = beam_solver
        self.wing = wing_structure

        self.load_transfer = LoadTransfer(wing_structure.span, n_stations)
        self.disp_transfer = DisplacementTransfer(n_stations)

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.relaxation = relaxation
        self.n_stations = n_stations

        self.history = []
        # Most recent 3-D VLM load distribution (set when the aero solver
        # supports solve_wing_distribution); used for VTK/ParaView ΔCp export.
        self._last_vlm_distribution = None

    def _compute_aero(self, airfoil_coords, twist, flight_condition) -> dict:
        """Compute distributed aerodynamic loads for the current wing shape.

        Uses a single 3-D VLM solve when the aero solver supports
        ``solve_wing_distribution`` (OpenAeroStruct) — the physically correct
        way to obtain a spanwise load distribution — otherwise falls back to
        the per-strip NeuralFoil path.
        """
        chord_dist = self.wing.chord_distribution()
        beam_y = self.load_transfer.y

        if hasattr(self.aero_solver, "solve_wing_distribution"):
            Re_root = flight_condition.reynolds * self.wing.chord_root
            dist = self.aero_solver.solve_wing_distribution(
                coordinates=airfoil_coords,
                twist_stations=beam_y,
                twist_deg=twist,
                alpha=flight_condition.alpha,
                Re=Re_root,
                mach=flight_condition.mach,
                velocity=flight_condition.velocity,
                rho=flight_condition.density,
            )
            self._last_vlm_distribution = dist

            # Interpolate the spanwise lift and moment (from panel forces) onto
            # the beam stations. The lift comes directly from the vertical panel
            # forces (validated against CL).
            lift = np.interp(beam_y, dist.y, dist.lift_per_len)
            moment = np.interp(beam_y, dist.y, dist.moment_per_len)

            q = flight_condition.dynamic_pressure
            # Drag is taken from the whole-wing VLM CD (induced + viscous, from
            # the Trefftz-plane calculation) — NOT from the body-axis streamwise
            # panel force, which is dominated by leading-edge suction and is not
            # the aerodynamic drag. It is distributed proportionally to local
            # chord (area) so it integrates back to the correct total drag.
            cd = np.full_like(chord_dist, dist.CD)
            drag = cd * q * chord_dist
            with np.errstate(divide="ignore", invalid="ignore"):
                cl = np.where(chord_dist > 1e-9, lift / (q * chord_dist), 0.0)
                cm = np.where(chord_dist ** 2 > 1e-9,
                              moment / (q * chord_dist ** 2), 0.0)

            return {
                'cl': cl, 'cd': cd, 'cm': cm,
                'lift': lift, 'drag': drag, 'moment': moment,
                'AR': float(2.0 * self.wing.span / max(np.mean(chord_dist), 1e-9)),
                'cd_induced': np.zeros_like(cd),
                'CL': dist.CL, 'CD': dist.CD,
            }

        # Default: per-strip NeuralFoil path.
        return self.load_transfer.compute_section_aero(
            self.aero_solver, airfoil_coords, chord_dist, twist, flight_condition,
        )

    def solve(
        self,
        airfoil_coords: tuple[np.ndarray, np.ndarray],
        initial_twist: np.ndarray,
        flight_condition: FlightCondition,
    ) -> dict:
        """Run aerostructural coupling iteration to convergence.

        Returns:
            dict with converged aero results, structural results, and iteration history
        """
        chord_dist = self.wing.chord_distribution()
        twist = initial_twist.copy()
        prev_deflection = np.zeros(self.n_stations)

        converged = False
        self.history = []

        for iteration in range(self.max_iterations):
            # Step 1: Aerodynamic analysis (3-D VLM if available, else strip)
            aero_data = self._compute_aero(airfoil_coords, twist, flight_condition)

            # Step 2: Structural analysis
            struct_result = self.beam_solver.solve(
                lift_distribution=aero_data['lift'],
                moment_distribution=aero_data['moment'],
                load_factor=flight_condition.load_factor,
            )

            # Step 3: Check convergence
            deflection_change = np.max(np.abs(struct_result.deflection - prev_deflection))
            self.history.append({
                'iteration': iteration,
                'deflection_change': deflection_change,
                'tip_deflection': struct_result.tip_deflection,
                'total_lift': np.trapz(aero_data['lift'], self.load_transfer.y),
                'total_drag': float(np.trapz(aero_data['drag'], self.load_transfer.y)),
                'L/D': (np.trapz(aero_data['lift'], self.load_transfer.y) /
                        np.trapz(aero_data['drag'], self.load_transfer.y)
                        if np.trapz(aero_data['drag'], self.load_transfer.y) > 1e-10 else 0.0),
                'failure_index': struct_result.failure_index,
            })

            logger.info(
                f"Coupling iter {iteration}: Δw={deflection_change:.6f}, "
                f"tip_def={struct_result.tip_deflection:.4f}m, "
                f"FI={struct_result.failure_index:.4f}"
            )

            if deflection_change < self.tolerance and iteration > 0:
                converged = True
                logger.info(f"Converged in {iteration + 1} iterations")
                break

            # Step 4: Update for next iteration (with relaxation)
            new_deflection = (self.relaxation * struct_result.deflection +
                            (1 - self.relaxation) * prev_deflection)

            # Update twist from structural deformation (also relaxed)
            new_twist = self.disp_transfer.update_twist(initial_twist, struct_result.twist)
            twist = self.relaxation * new_twist + (1 - self.relaxation) * twist

            prev_deflection = new_deflection

        return {
            'converged': converged,
            'iterations': iteration + 1,
            'aero': aero_data,
            'structure': struct_result,
            'history': self.history,
            'final_twist': twist,
            'initial_twist': initial_twist.copy(),
            'airfoil_coords': airfoil_coords,
            # 3-D VLM load distribution (None for the strip-theory path); carries
            # the real per-panel ΔCp field for VTK/ParaView export.
            'vlm': self._last_vlm_distribution,
        }
