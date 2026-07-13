"""
Euler-Bernoulli beam structural solver for wing analysis.

Solves for deflection, twist, bending stress, and shear stress along the wing span
under aerodynamic loading. Coupled with composite laminate properties from CLT.

The wing is modeled as a cantilevered beam (root fixed, tip free) with:
- Spanwise varying EI(y), GJ(y) from composite laminate properties
- Distributed aerodynamic lift q(y) and pitching moment m(y)
- Concentrated loads (engine, fuel)
- Geometric nonlinearity (optional: follower forces)

Finite element formulation with Hermite cubic shape functions for bending
and linear shape functions for torsion.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import logging

from .composite_properties import Laminate, quasi_isotropic
from .vam_section import compute_vam_beam_properties

logger = logging.getLogger(__name__)


@dataclass
class WingStructure:
    """Wing structural definition along the span."""
    span: float                    # Semi-span [m]
    n_elements: int = 20           # Number of beam elements
    chord_root: float = 3.0        # Root chord [m]
    chord_tip: float = 1.2         # Tip chord [m]
    sweep_deg: float = 25.0        # Quarter-chord sweep [deg]
    taper_ratio: float = 0.4       # Tip/root chord ratio

    # Structural sizing
    skin_laminate: Optional[Laminate] = None
    spar_laminate: Optional[Laminate] = None
    skin_laminates: Optional[list[Laminate]] = None  # Per-station skin laminates
    box_width_fraction: float = 0.5  # Wing box width as fraction of local chord
    box_height_fraction: float = 0.12  # Wing box height as fraction of local chord
    skin_thickness_scales: Optional[np.ndarray] = None  # Per-station thickness multiplier

    # VAM / beam-model switches
    use_vam: bool = False  # Use VAM cross-sectional stiffness module

    # Material override for simple analyses
    E: float = 70e9       # Young's modulus [Pa] (default: aluminum)
    G: float = 26e9       # Shear modulus [Pa]
    rho: float = 2700.0   # Density [kg/m³]

    def __post_init__(self):
        if self.skin_laminate is None:
            self.skin_laminate = quasi_isotropic()
        if self.skin_thickness_scales is None:
            self.skin_thickness_scales = np.ones(self.n_elements + 1)
        # VAM is automatically enabled when per-station laminates are used
        # so that the cross-sectional stiffness is recomputed for each station.
        if self.skin_laminates is not None and len(self.skin_laminates) == self.n_elements + 1:
            self.use_vam = True

    def set_skin_thickness_scale(self, station_idx: int, scale: float):
        """Set the skin thickness scale at a single spanwise station."""
        if self.skin_thickness_scales is None:
            self.skin_thickness_scales = np.ones(self.n_elements + 1)
        self.skin_thickness_scales[station_idx] = float(scale)

    def set_skin_thickness_scales(self, scales: np.ndarray):
        """Set skin thickness scales for all spanwise stations.

        Args:
            scales: [n_nodes] array of thickness multipliers (>= 0).
        """
        scales = np.asarray(scales, dtype=float)
        n_nodes = self.n_elements + 1
        if len(scales) != n_nodes:
            raise ValueError(
                f"Expected {n_nodes} thickness scales, got {len(scales)}"
            )
        if np.any(scales <= 0):
            raise ValueError("Thickness scales must be positive")
        self.skin_thickness_scales = scales.copy()

    def set_laminate_at_station(self, station_idx: int, laminate: Laminate):
        """Set the skin laminate at a single spanwise station."""
        if self.skin_laminates is None:
            self.skin_laminates = [self.skin_laminate] * (self.n_elements + 1)
        self.skin_laminates[station_idx] = laminate

    def set_skin_laminates(self, laminates: Optional[list[Laminate]]):
        """Set skin laminates for all spanwise stations.

        Args:
            laminates: [n_nodes] list of Laminate objects, or None to reset.
        """
        if laminates is None:
            self.skin_laminates = None
            return
        n_nodes = self.n_elements + 1
        if len(laminates) != n_nodes:
            raise ValueError(
                f"Expected {n_nodes} laminates, got {len(laminates)}"
            )
        self.skin_laminates = laminates.copy()

    @property
    def element_length(self) -> float:
        return self.span / self.n_elements

    def spanwise_stations(self) -> np.ndarray:
        """Node positions along the span [0, span]."""
        return np.linspace(0, self.span, self.n_elements + 1)

    def chord_distribution(self) -> np.ndarray:
        """Chord at each spanwise station."""
        y = self.spanwise_stations()
        eta = y / self.span  # normalized span
        return self.chord_root * (1.0 - eta * (1.0 - self.taper_ratio))

    def section_properties(self) -> dict[str, np.ndarray]:
        """Compute EI, GJ, EA, mass/length at each station.

        Uses composite laminate properties if available, otherwise
        falls back to simple rectangular box with given E, G.

        For the composite wing box the dominant stiffness comes from the
        extensional stiffness (A matrix) of the top and bottom skins acting
        at a distance from the neutral axis, rather than from the skin's own
        bending stiffness (D matrix).
        """
        y = self.spanwise_stations()
        n = len(y)
        chords = self.chord_distribution()

        if self.skin_thickness_scales is None or len(self.skin_thickness_scales) != n:
            self.skin_thickness_scales = np.ones(n)
        scales = self.skin_thickness_scales

        use_per_station = (
            self.skin_laminates is not None
            and len(self.skin_laminates) == n
        )

        # Use VAM cross-sectional stiffness module when requested or when
        # per-station laminates are present (the crude formula cannot represent
        # varying ply counts along the span).
        if self.use_vam or use_per_station:
            vam_props = compute_vam_beam_properties(
                span=self.span,
                n_stations=n,
                chord_distribution=chords,
                box_width_fraction=self.box_width_fraction,
                box_height_fraction=self.box_height_fraction,
                skin_laminate=self.skin_laminate,
                skin_thickness_scales=scales,
                skin_laminates=self.skin_laminates if use_per_station else None,
                spar_laminate=self.spar_laminate,
            )
            return {
                'y': y,
                'EI': vam_props.EI,
                'GJ': vam_props.GJ,
                'EA': vam_props.EA,
                'mass_per_length': vam_props.mass_per_length,
                'chord': chords,
                'skin_thickness': vam_props.skin_thickness,
                'bend_twist_coupling': vam_props.bend_twist_coupling,
            }

        EI = np.zeros(n)
        GJ = np.zeros(n)
        EA = np.zeros(n)
        mass_per_length = np.zeros(n)
        skin_thickness = np.zeros(n)

        for i in range(n):
            c = chords[i]
            box_w = c * self.box_width_fraction
            box_h = c * self.box_height_fraction
            scale = scales[i]

            lam = (
                self.skin_laminates[i]
                if use_per_station and self.skin_laminates[i] is not None
                else self.skin_laminate
            )

            if lam is not None:
                # Use CLT-derived extensional stiffness (A matrix) for the box.
                # The bending stiffness is dominated by the upper and lower skins
                # acting at ±box_h/2 from the neutral axis.
                ABD = lam.ABD_matrix()
                A = ABD[:3, :3]
                A11 = A[0, 0] * scale  # thickness scales linearly with scale
                A66 = A[2, 2] * scale

                EI[i] = 0.5 * A11 * box_w * box_h**2
                GJ[i] = 2.0 * A66 * box_w * box_h**2 / (box_w + box_h)
                EA[i] = A11 * box_w

                # Mass: scaled laminate mass per area times width (upper + lower skin)
                mass_per_length[i] = (
                    lam.total_mass_per_area * scale * 2.0 * box_w
                )
                # Add spar contributions (front + rear)
                mass_per_length[i] += (
                    lam.material.rho
                    * lam.total_thickness
                    * scale
                    * box_h
                    * 2.0
                )
                skin_thickness[i] = lam.total_thickness * scale
            else:
                # Simple rectangular box
                t_skin = 0.003 * scale  # 3mm skin baseline
                I = box_w * box_h**3 / 12.0 - (box_w - 2*t_skin) * (box_h - 2*t_skin)**3 / 12.0
                J = 2.0 * t_skin * (box_w * box_h)**2 / (box_w + box_h)
                EI[i] = self.E * I
                GJ[i] = self.G * J
                EA[i] = self.E * 2.0 * (box_w + box_h) * t_skin
                mass_per_length[i] = self.rho * 2.0 * (box_w + box_h) * t_skin
                skin_thickness[i] = t_skin

        return {
            'y': y,
            'EI': EI,
            'GJ': GJ,
            'EA': EA,
            'mass_per_length': mass_per_length,
            'chord': chords,
            'skin_thickness': skin_thickness,
        }


@dataclass
class BeamResult:
    """Results from beam structural analysis."""
    y: np.ndarray                  # Spanwise stations [m]
    deflection: np.ndarray         # Vertical deflection [m]
    twist: np.ndarray              # Twist angle [rad]
    bending_moment: np.ndarray     # Bending moment [N·m]
    shear_force: np.ndarray        # Shear force [N]
    torque: np.ndarray             # Torque [N·m]
    bending_stress: np.ndarray     # Max bending stress [Pa]
    shear_stress: np.ndarray       # Max shear stress [Pa]
    total_mass: float              # Wing structural mass [kg]
    tip_deflection: float          # Tip deflection [m]
    max_stress: float              # Maximum von Mises stress [Pa]
    failure_index: float           # Max failure index (>1 = failure)


class EulerBernoulliBeamSolver:
    """FEM-based Euler-Bernoulli beam solver for wing structures.

    Solves the coupled bending-torsion system:
        d²/dy²[EI(y) d²w/dy²] = q(y)     (bending)
        d/dy[GJ(y) dθ/dy] = m(y)           (torsion)

    with boundary conditions:
        Root (y=0): w=0, dw/dy=0, θ=0 (clamped)
        Tip (y=b): M=0, V=0, T=0 (free)

    If a trained ``StructuralSurrogate`` is supplied, the solver uses it to
    replace the low-fidelity tip-deflection and failure-index estimates with
    MYSTRAN-corrected predictions while still returning the full spanwise
    displacement/twist distributions from the VAM beam model.
    """

    def __init__(self, wing: WingStructure, surrogate=None):
        self.wing = wing
        self.n_elem = wing.n_elements
        self.n_nodes = wing.n_elements + 1
        self.surrogate = surrogate

        # DOFs per node: w (deflection), θ_bend (bending rotation), θ_twist (twist)
        self.n_dof_per_node = 3
        self.n_dof = self.n_nodes * self.n_dof_per_node

    def solve(
        self,
        lift_distribution: np.ndarray,
        moment_distribution: Optional[np.ndarray] = None,
        point_loads: Optional[dict] = None,
        load_factor: float = 2.5,
    ) -> BeamResult:
        """Solve the beam bending-torsion problem.

        Args:
            lift_distribution: [n_nodes] distributed lift [N/m] at each station
            moment_distribution: [n_nodes] distributed pitching moment [N·m/m]
            point_loads: dict of {station_index: (force, moment)}
            load_factor: ultimate load factor (default 2.5g)
        """
        props = self.wing.section_properties()
        EI = props['EI']
        GJ = props['GJ']
        y = props['y']
        L = self.wing.element_length

        # Apply load factor
        q = lift_distribution * load_factor
        if moment_distribution is not None:
            m = moment_distribution * load_factor
        else:
            m = np.zeros(self.n_nodes)

        # Assemble global stiffness matrix and load vector
        K = np.zeros((self.n_dof, self.n_dof))
        F = np.zeros(self.n_dof)

        for e in range(self.n_elem):
            # Average element properties
            EI_e = 0.5 * (EI[e] + EI[e + 1])
            GJ_e = 0.5 * (GJ[e] + GJ[e + 1])

            # Element stiffness matrix (bending: Hermite cubic)
            k_bend = (EI_e / L**3) * np.array([
                [12,    6*L,   -12,    6*L],
                [6*L,   4*L**2, -6*L,  2*L**2],
                [-12,  -6*L,    12,   -6*L],
                [6*L,   2*L**2, -6*L,  4*L**2],
            ])

            # Element stiffness matrix (torsion: linear)
            k_tors = (GJ_e / L) * np.array([
                [1,  -1],
                [-1,  1],
            ])

            # Element load vector (consistent loads)
            q_e = 0.5 * (q[e] + q[e + 1])
            m_e = 0.5 * (m[e] + m[e + 1])

            f_bend = q_e * L * np.array([0.5, L/12.0, 0.5, -L/12.0])
            f_tors = m_e * L * np.array([0.5, 0.5])

            # DOF mapping: node i → [w_i, θ_bend_i, θ_twist_i]
            dofs_bend = [
                e * 3, e * 3 + 1,
                (e + 1) * 3, (e + 1) * 3 + 1,
            ]
            dofs_tors = [e * 3 + 2, (e + 1) * 3 + 2]

            # Assemble bending
            for ii, di in enumerate(dofs_bend):
                F[di] += f_bend[ii]
                for jj, dj in enumerate(dofs_bend):
                    K[di, dj] += k_bend[ii, jj]

            # Assemble torsion
            for ii, di in enumerate(dofs_tors):
                F[di] += f_tors[ii]
                for jj, dj in enumerate(dofs_tors):
                    K[di, dj] += k_tors[ii, jj]

        # Apply point loads
        if point_loads is not None:
            for node_idx, (force, moment) in point_loads.items():
                F[node_idx * 3] += force * load_factor
                F[node_idx * 3 + 2] += moment * load_factor

        # Apply boundary conditions (clamped root: w=0, θ_bend=0, θ_twist=0)
        bc_dofs = [0, 1, 2]  # Root node DOFs
        free_dofs = [i for i in range(self.n_dof) if i not in bc_dofs]

        K_free = K[np.ix_(free_dofs, free_dofs)]
        F_free = F[free_dofs]

        # Solve
        try:
            u_free = np.linalg.solve(K_free, F_free)
        except np.linalg.LinAlgError:
            logger.warning("Singular stiffness matrix — adding regularization")
            K_free += np.eye(len(free_dofs)) * 1e-10
            u_free = np.linalg.solve(K_free, F_free)

        # Reconstruct full displacement vector
        u = np.zeros(self.n_dof)
        for i, dof in enumerate(free_dofs):
            u[dof] = u_free[i]

        # Extract deflection, rotation, twist
        deflection = u[0::3]
        bending_rotation = u[1::3]
        twist = u[2::3]

        # Compute internal forces
        bending_moment = np.zeros(self.n_nodes)
        shear_force = np.zeros(self.n_nodes)
        torque = np.zeros(self.n_nodes)

        for e in range(self.n_elem):
            EI_e = 0.5 * (EI[e] + EI[e + 1])
            GJ_e = 0.5 * (GJ[e] + GJ[e + 1])

            # Bending moment: M = EI * d²w/dy² ≈ EI * Δθ_bend / L
            bending_moment[e] = EI_e * abs(bending_rotation[e + 1] - bending_rotation[e]) / L

            # Torque: T = GJ * dθ_twist/dy
            torque[e] = GJ_e * abs(twist[e + 1] - twist[e]) / L

        # Shear force from vertical equilibrium: V(y) = ∫_y^L q(s) ds.
        # This is more robust than differentiating the numerically computed
        # bending moment, which gave spuriously large shear values.
        for e in range(self.n_nodes - 1):
            shear_force[e] = abs(np.trapz(q[e:], y[e:]))
        shear_force[-1] = 0.0

        # Compute stresses
        chords = props['chord']
        box_h = chords * self.wing.box_height_fraction
        box_w = chords * self.wing.box_width_fraction
        scales = self.wing.skin_thickness_scales
        if scales is None or len(scales) != len(box_h):
            scales = np.ones(len(box_h))
        skin_thickness = props.get(
            'skin_thickness',
            np.full_like(
                chords,
                self.wing.skin_laminate.total_thickness if self.wing.skin_laminate else 0.003,
            ),
        )
        t_skin = skin_thickness * scales

        # Thin-walled box section modulus and shear area.
        # Bending is carried by the upper and lower skins at ±box_h/2.
        # Shear is carried by the full perimeter of the box.
        Z_bending = t_skin * box_w * box_h
        A_shear = 2.0 * t_skin * (box_w + box_h)

        bending_stress = bending_moment / (Z_bending + 1e-20)
        shear_stress = shear_force / (A_shear + 1e-20)

        # Total mass
        total_mass = float(np.sum(props['mass_per_length'][:-1] * L))

        # Failure index: use Tsai-Wu for composites, von Mises for isotropic
        if self.wing.skin_laminate or self.wing.skin_laminates:
            laminates = self.wing.skin_laminates if self.wing.skin_laminates else None
            failure_index = self._compute_tsai_wu_failure(
                bending_moment, shear_force, torque, box_w, box_h, scales, laminates
            )
            sigma_allow = self.wing.skin_laminate.material.Xt
        else:
            sigma_allow = 400e6  # Al yield
            failure_index = float(
                np.max(np.sqrt(bending_stress**2 + 3 * shear_stress**2)) / sigma_allow
            )

        max_stress = float(np.max(np.sqrt(bending_stress**2 + 3 * shear_stress**2)))

        result = BeamResult(
            y=y,
            deflection=deflection,
            twist=twist,
            bending_moment=bending_moment,
            shear_force=shear_force,
            torque=torque,
            bending_stress=bending_stress,
            shear_stress=shear_stress,
            total_mass=total_mass,
            tip_deflection=float(deflection[-1]),
            max_stress=max_stress,
            failure_index=failure_index,
        )

        # If a MYSTRAN-trained surrogate is available, override the
        # low-fidelity global responses with the neural prediction.  The
        # spanwise fields are still supplied by the VAM beam model.
        if self.surrogate is not None:
            try:
                from .structural_surrogate import build_surrogate_features
                features = build_surrogate_features(
                    self.wing,
                    lift_distribution,
                    moment_distribution,
                    load_factor,
                )
                pred = self.surrogate.predict_dict(features.reshape(1, -1))
                # Map surrogate outputs to BeamResult fields.  The surrogate
                # was trained on a 1 m wing-box; the deflection is therefore
                # interpreted as a corrected global displacement proxy and the
                # failure index is used directly.
                if "max_vertical_displacement" in pred:
                    result.tip_deflection = float(pred["max_vertical_displacement"])
                if "max_failure_index" in pred:
                    result.failure_index = float(pred["max_failure_index"])
                if "max_von_mises" in pred:
                    result.max_stress = float(pred["max_von_mises"])
                if "max_von_mises_stress" in pred:
                    result.max_stress = float(pred["max_von_mises_stress"])
            except Exception as exc:
                logger.warning(f"Surrogate prediction failed: {exc}; using beam result")

        return result

    def compute_mass(self) -> float:
        """Compute total structural mass [kg]."""
        props = self.wing.section_properties()
        L = self.wing.element_length
        return float(np.sum(props['mass_per_length'][:-1] * L))

    def _compute_tsai_wu_failure(
        self,
        bending_moment: np.ndarray,
        shear_force: np.ndarray,
        torque: np.ndarray,
        box_w: np.ndarray,
        box_h: np.ndarray,
        thickness_scales: np.ndarray,
        laminates: Optional[list[Laminate]] = None,
    ) -> float:
        """Compute the maximum composite failure index across stations and plies.

        Uses the maximum-stress (Max-Stress) failure criterion per ply, which is
        robust for preliminary wing-box sizing and avoids the well-known
        numerical pathologies of the Tsai-Wu interaction term when tensile and
        compressive strengths differ strongly.

        Treats the wing-box skins as membrane laminates carrying the axial
        force resultant from bending (N_x = M / h / w) plus the shear flow from
        vertical shear and torsion. The laminate mid-plane strain is obtained
        from [N] = [A][ε⁰], and each ply stress is computed in material
        coordinates via its transformed stiffness matrix Q_bar.

        Both tension and compression outer-fiber cases are checked because the
        simplified beam model loses the sign of the bending moment.
        """
        base_laminate = self.wing.skin_laminate
        n_stations = len(bending_moment)

        max_fi = 0.0

        for i in range(n_stations):
            M = bending_moment[i]
            V = shear_force[i]
            T = torque[i]
            w = box_w[i]
            h = box_h[i]
            scale = thickness_scales[i] if i < len(thickness_scales) else 1.0

            if w <= 0.0 or h <= 0.0 or scale <= 0.0:
                continue

            laminate = (
                laminates[i]
                if laminates and laminates[i] is not None
                else base_laminate
            )
            if laminate is None:
                continue

            angles = laminate.full_angles
            A = laminate.ABD_matrix()[:3, :3]
            try:
                A_inv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                A_inv = np.linalg.pinv(A)

            # Axial force resultant in the skins from bending.
            # The couple formed by the upper and lower skins gives F = M / h
            # in each skin; distributed over the skin width w gives N_x [N/m].
            Nx = M / (h * w)

            # Shear flow from vertical shear (two webs) plus torsion.
            # These are approximate membrane resultants for the laminate.
            Nxy_shear = V / (2.0 * h) if h > 1e-10 else 0.0
            Nxy_torsion = T / (2.0 * w * h) if w * h > 1e-10 else 0.0
            Nxy = Nxy_shear + Nxy_torsion

            # Scale the A matrix inverse by 1/scale because [A] is linear in
            # thickness; same N produces half the strain when thickness doubles.
            A_inv_scaled = A_inv / scale

            mat = laminate.material

            # Check both tension and compression on the outer fibers.
            for sign in (1.0, -1.0):
                N = np.array([sign * Nx, 0.0, Nxy])
                epsilon = A_inv_scaled @ N

                ply_stresses = np.zeros((len(angles), 3))
                for j, theta_deg in enumerate(angles):
                    Qb = laminate.Q_bar(theta_deg)
                    ply_stresses[j] = Qb @ epsilon

                # Maximum-stress failure index per ply (conservative, robust)
                s1 = ply_stresses[:, 0]
                s2 = ply_stresses[:, 1]
                t12 = ply_stresses[:, 2]
                fi_ply = np.maximum(
                    np.maximum(
                        np.where(s1 >= 0, s1 / (mat.Xt + 1e-20), -s1 / (mat.Xc + 1e-20)),
                        np.where(s2 >= 0, s2 / (mat.Yt + 1e-20), -s2 / (mat.Yc + 1e-20)),
                    ),
                    np.abs(t12) / (mat.S12 + 1e-20),
                )
                max_fi = max(max_fi, float(np.max(fi_ply)))

        return max_fi

    def gradient_fd(
        self,
        lift_distribution: np.ndarray,
        design_vars: np.ndarray,
        apply_design: callable,
        objectives: list[str] = ['tip_deflection', 'total_mass', 'failure_index'],
        fd_step: float = 1e-6,
    ) -> dict[str, np.ndarray]:
        """Compute structural gradients via finite differences.

        Args:
            lift_distribution: [n_nodes] lift loading
            design_vars: flat design vector (e.g., skin thicknesses)
            apply_design: function that applies design_vars to self.wing
            objectives: list of BeamResult attribute names
        """
        # Baseline
        apply_design(design_vars)
        base_result = self.solve(lift_distribution)
        base_vals = {obj: getattr(base_result, obj) for obj in objectives}

        grads = {obj: np.zeros(len(design_vars)) for obj in objectives}

        for i in range(len(design_vars)):
            x_pert = design_vars.copy()
            x_pert[i] += fd_step
            apply_design(x_pert)
            pert_result = self.solve(lift_distribution)

            for obj in objectives:
                grads[obj][i] = (getattr(pert_result, obj) - base_vals[obj]) / fd_step

        # Restore baseline
        apply_design(design_vars)
        return grads
