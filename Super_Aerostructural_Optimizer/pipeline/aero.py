"""Physics-based aerodynamic model for the coupled MDO workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FlightConditionConfig
from .geometry import SpanwiseAeroGrid


def _isa_density_viscosity(altitude_m: float) -> tuple[float, float]:
    """Return ISA density (kg/m^3) and dynamic viscosity (Pa*s)."""
    g0 = 9.80665
    r_air = 287.05
    t0 = 288.15
    p0 = 101325.0
    lapse = -0.0065

    h = float(max(0.0, altitude_m))
    if h <= 11000.0:
        t = t0 + lapse * h
        p = p0 * (t / t0) ** (-g0 / (lapse * r_air))
    else:
        t = 216.65
        p11 = p0 * (t / t0) ** (-g0 / (lapse * r_air))
        p = p11 * np.exp(-g0 * (h - 11000.0) / (r_air * t))

    rho = p / (r_air * t)
    mu = 1.458e-6 * t ** 1.5 / (t + 110.4)
    return float(rho), float(mu)


@dataclass
class AerodynamicResult:
    """Aerodynamic loads and coefficients for one coupled iteration."""

    alpha_deg: float
    trim_converged: bool
    cl: float
    cd: float
    cdi: float
    cd0: float
    lift_n: float
    drag_n: float
    l_over_d: float
    reynolds_mean: float
    span_efficiency: float
    panel_lift_n: np.ndarray
    panel_drag_n: np.ndarray
    panel_forces_xyz: np.ndarray


class LiftingLineAeroSolver:
    """Spanwise aerodynamic solver with finite-wing and viscous corrections."""

    def __init__(
        self,
        flight_config: FlightConditionConfig,
        thickness_to_chord: float = 0.12,
        alpha_l0_deg: float = -2.0,
        cl_alpha_rad: float = 2.0 * np.pi,
    ) -> None:
        self.flight = flight_config
        self.thickness_to_chord = float(thickness_to_chord)
        self.alpha_l0_deg = float(alpha_l0_deg)
        self.cl_alpha_rad = float(cl_alpha_rad)

    def _evaluate_at_alpha(
        self,
        span_grid: SpanwiseAeroGrid,
        alpha_deg: float,
    ) -> AerodynamicResult:
        rho, mu = _isa_density_viscosity(self.flight.altitude_m)
        v_inf = float(self.flight.velocity_ms)
        q_inf = 0.5 * rho * v_inf ** 2
        area_half = float(span_grid.area_half_m2)

        b_full = 2.0 * span_grid.half_span_m
        area_full = 2.0 * area_half
        ar = b_full ** 2 / max(area_full, 1.0e-12)

        alpha_local = np.radians(alpha_deg + span_grid.twist_panel_deg - self.alpha_l0_deg)
        finite_factor = ar / (ar + 2.0)
        cl_2d = self.cl_alpha_rad * alpha_local
        cl_3d = np.clip(cl_2d * finite_factor, -1.5, 1.8)

        lift_per_span = q_inf * span_grid.chord_panel * cl_3d
        panel_lift = lift_per_span * span_grid.dy_panel
        lift_total = float(np.sum(panel_lift))
        cl_total = lift_total / max(q_inf * area_half, 1.0e-12)

        # Approximate span efficiency from loading-shape deviation from elliptical.
        y_hat = span_grid.y_panel_mid / max(span_grid.half_span_m, 1.0e-12)
        ellip = np.sqrt(np.clip(1.0 - y_hat ** 2, 0.0, 1.0))
        ellip_norm = ellip / max(np.mean(np.abs(ellip)), 1.0e-12)
        load_norm = lift_per_span / max(np.mean(np.abs(lift_per_span)), 1.0e-12)
        shape_err = float(np.sqrt(np.mean((load_norm - ellip_norm) ** 2)))
        span_efficiency = float(np.clip(0.96 - 0.16 * shape_err, 0.65, 0.97))

        cdi = cl_total ** 2 / max(np.pi * ar * span_efficiency, 1.0e-12)

        reynolds = rho * v_inf * span_grid.chord_panel / max(mu, 1.0e-12)
        reynolds = np.maximum(reynolds, 5.0e4)
        cf = 0.455 / np.maximum(np.log10(reynolds), 1.0) ** 2.58
        form_factor = 1.0 + 2.0 * self.thickness_to_chord + 60.0 * self.thickness_to_chord ** 4
        cd0_local = cf * form_factor
        panel_drag_profile = q_inf * span_grid.chord_panel * cd0_local * span_grid.dy_panel
        cd0 = float(np.sum(panel_drag_profile) / max(q_inf * area_half, 1.0e-12))

        drag_induced_total = q_inf * area_half * cdi
        lift_abs_sum = float(np.sum(np.abs(panel_lift)))
        induced_weight = np.abs(panel_lift) / max(lift_abs_sum, 1.0e-12)
        panel_drag_induced = drag_induced_total * induced_weight
        panel_drag_total = panel_drag_profile + panel_drag_induced
        drag_total = float(np.sum(panel_drag_total))

        cd_total = cd0 + cdi
        lod = lift_total / max(drag_total, 1.0e-12)

        panel_forces = np.zeros((span_grid.y_panel_mid.size, 3), dtype=float)
        panel_forces[:, 0] = -panel_drag_total
        panel_forces[:, 2] = panel_lift

        return AerodynamicResult(
            alpha_deg=float(alpha_deg),
            trim_converged=True,
            cl=float(cl_total),
            cd=float(cd_total),
            cdi=float(cdi),
            cd0=float(cd0),
            lift_n=float(lift_total),
            drag_n=float(drag_total),
            l_over_d=float(lod),
            reynolds_mean=float(np.mean(reynolds)),
            span_efficiency=span_efficiency,
            panel_lift_n=panel_lift,
            panel_drag_n=panel_drag_total,
            panel_forces_xyz=panel_forces,
        )

    def solve_trimmed(self, span_grid: SpanwiseAeroGrid) -> AerodynamicResult:
        """Solve for angle of attack that matches configured target CL."""
        target_cl = float(self.flight.target_cl)
        alpha_low = float(self.flight.trim_alpha_min_deg)
        alpha_high = float(self.flight.trim_alpha_max_deg)

        low = self._evaluate_at_alpha(span_grid, alpha_low)
        high = self._evaluate_at_alpha(span_grid, alpha_high)
        f_low = low.cl - target_cl
        f_high = high.cl - target_cl

        # If no bracket, pick closest bound and continue.
        if f_low * f_high > 0.0:
            winner = low if abs(f_low) <= abs(f_high) else high
            winner.trim_converged = False
            return winner

        best = low
        for _ in range(int(self.flight.trim_max_iter)):
            alpha_mid = 0.5 * (alpha_low + alpha_high)
            mid = self._evaluate_at_alpha(span_grid, alpha_mid)
            f_mid = mid.cl - target_cl
            best = mid
            if abs(f_mid) <= float(self.flight.trim_tol):
                best.trim_converged = True
                return best
            if f_low * f_mid <= 0.0:
                alpha_high = alpha_mid
                high = mid
                f_high = f_mid
            else:
                alpha_low = alpha_mid
                low = mid
                f_low = f_mid

        best.trim_converged = False
        return best

    def solve_fixed_alpha(
        self,
        span_grid: SpanwiseAeroGrid,
        alpha_deg: float | None = None,
    ) -> AerodynamicResult:
        """Evaluate loads at a specified angle of attack."""
        alpha = self.flight.alpha_deg if alpha_deg is None else float(alpha_deg)
        return self._evaluate_at_alpha(span_grid, alpha)

