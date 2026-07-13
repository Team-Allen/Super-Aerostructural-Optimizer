"""
VAM-aligned thin-walled composite cross-sectional stiffness module.

This module replaces the crude CLT box-beam stiffness estimate with a more
rigorous thin-walled beam formulation consistent with the Variational
Asymptotic Method (VAM).  The cross-section is described by a single-cell
rectangular wing box made of composite laminates.  For each wall we use the
full Classical Lamination Theory (CLT) ABD stiffness matrix and integrate the
membrane and local-bending contributions around the perimeter to obtain a
beam stiffness matrix.

References:
- Hodges, D. H. "Nonlinear Composite Beam Theory" (VABS/VAM).
- Jung, S. N., Nagaraj, V. T., & Chopra, I. "Assessment of composite rotor
  blade modeling techniques", Journal of the American Helicopter Society, 1999.
- Smith, E. C., & Chopra, I. "Formulation and evaluation of an analytical
  model for composite box beams", Journal of the American Helicopter Society, 1990.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .composite_properties import Laminate


@dataclass
class WallSegment:
    """A straight segment of the thin-walled cross-section."""

    name: str
    x0: float          # Start x [m]
    z0: float          # Start z [m]
    x1: float          # End x [m]
    z1: float          # End z [m]
    laminate: Laminate

    @property
    def length(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.z1 - self.z0))

    @property
    def is_horizontal(self) -> bool:
        return abs(self.z1 - self.z0) < 1e-12

    @property
    def is_vertical(self) -> bool:
        return abs(self.x1 - self.x0) < 1e-12

    def point(self, s: float) -> tuple[float, float]:
        """Linear interpolation along the segment (s in [0,1])."""
        x = self.x0 + s * (self.x1 - self.x0)
        z = self.z0 + s * (self.z1 - self.z0)
        return float(x), float(z)


class VAMSection:
    """Cross-sectional stiffness of a composite wing box via VAM-like integrals.

    Coordinate system:
        y : beam axis (spanwise), positive toward tip
        x : chordwise, origin at box centroid
        z : vertical, positive up

    The local laminate reference axis (1) is assumed to be aligned with the
    beam axis y for all walls (typical wing skin/spar layup).  Ply angles are
    therefore interpreted as measured from the spanwise direction.
    """

    def __init__(
        self,
        width: float,
        height: float,
        skin_laminate: Laminate,
        spar_laminate: Optional[Laminate] = None,
        n_integration_points: int = 16,
    ):
        """
        Args:
            width: Wing-box width (chordwise) [m].
            height: Wing-box height (vertical) [m].
            skin_laminate: Laminate used for top/bottom skins.
            spar_laminate: Laminate used for front/rear spars. If ``None``,
                the skin laminate is used for all walls.
            n_integration_points: Number of Gauss points per wall for the
                membrane integrals.
        """
        self.width = float(width)
        self.height = float(height)
        self.skin_laminate = skin_laminate
        self.spar_laminate = spar_laminate or skin_laminate
        self.n_points = max(4, n_integration_points)

        w2 = 0.5 * self.width
        h2 = 0.5 * self.height

        # Perimeter order: bottom, left, top, right
        self.walls = [
            WallSegment("bottom", w2, -h2, -w2, -h2, self.skin_laminate),
            WallSegment("left", -w2, -h2, -w2, h2, self.spar_laminate),
            WallSegment("top", -w2, h2, w2, h2, self.skin_laminate),
            WallSegment("right", w2, h2, w2, -h2, self.spar_laminate),
        ]

        # The ABD matrix of a wall is constant along the wall (it depends only
        # on the wall's laminate, not on the integration point). Cache it per
        # laminate so the expensive CLT assembly runs once per unique laminate
        # instead of once per Gauss point per integral (~500x fewer calls).
        self._abd_cache: dict[int, np.ndarray] = {}

        # Gauss-Legendre points/weights on [0, 1] are the same for every wall
        # and every integral; precompute once.
        s, w = np.polynomial.legendre.leggauss(self.n_points)
        self._gauss_s = 0.5 * (s + 1.0)
        self._gauss_w = w

    # ------------------------------------------------------------------
    # Core stiffness integrals
    # ------------------------------------------------------------------

    def _wall_abd(self, wall: WallSegment) -> np.ndarray:
        """Return the 6x6 ABD matrix for a wall (cached per laminate).

        The ABD matrix depends only on the wall's laminate, so it is assembled
        once per unique laminate and reused across every integration point.
        """
        key = id(wall.laminate)
        abd = self._abd_cache.get(key)
        if abd is None:
            abd = wall.laminate.ABD_matrix()
            self._abd_cache[key] = abd
        return abd

    def _integrate_membrane(self, func) -> float:
        """Integrate a scalar function over all wall mid-surfaces."""
        total = 0.0
        for wall in self.walls:
            L = wall.length
            # Gauss-Legendre points/weights precomputed in __init__
            for si, wi in zip(self._gauss_s, self._gauss_w):
                x, z = wall.point(si)
                val = func(wall, x, z)
                total += 0.5 * L * wi * val
        return float(total)

    def stiffness_matrix(self) -> np.ndarray:
        """Compute the 6x6 cross-sectional stiffness matrix S.

        Ordering of generalized strains / stress resultants:
            [ε_y, γ_xy, γ_yz, κ_z, κ_x, κ_y]

        where:
            ε_y  : axial strain along beam axis
            γ_xy : transverse shear in the chordwise plane
            γ_yz : transverse shear in the vertical plane
            κ_z  : bending curvature about the z-axis (chordwise bending)
            κ_x  : bending curvature about the x-axis (vertical bending)
            κ_y  : twist rate about the beam axis

        The matrix is symmetric and contains the usual EA, shear, EI, and GJ
        terms plus extension-bending and bend-twist couplings that arise from
        anisotropic laminates.
        """
        S = np.zeros((6, 6))

        # Membrane contributions to EA, shear, and bending
        # A11 is extensional stiffness along the beam axis (local 1 = y)
        # A66 is in-plane shear stiffness
        # A16/A26 are extension-shear coupling terms
        # D11/D22 are local bending stiffnesses of the wall
        # D16/D26 are bend-twist coupling terms

        # EA : integral of A11 around perimeter
        S[0, 0] = self._integrate_membrane(
            lambda wall, x, z: self._wall_abd(wall)[0, 0]
        )

        # Extension-shear coupling
        S[0, 1] = self._integrate_membrane(
            lambda wall, x, z: self._wall_abd(wall)[0, 2]
        )
        S[0, 2] = S[0, 1]

        # Shear stiffness (approximate)
        S[1, 1] = self._integrate_membrane(
            lambda wall, x, z: self._wall_abd(wall)[2, 2]
        )
        S[2, 2] = S[1, 1]

        # Bending stiffness about x-axis (vertical bending)
        # Contributions: A11 * z^2 (membrane couple) + local D terms
        def ei_x_contrib(wall, x, z):
            ABD = self._wall_abd(wall)
            A11 = ABD[0, 0]
            if wall.is_horizontal:
                D_local = ABD[3, 3]  # D11, bending along wall s direction
            elif wall.is_vertical:
                D_local = ABD[4, 4]  # D22
            else:
                # Diagonal average for slanted walls (not used for box)
                D_local = 0.5 * (ABD[3, 3] + ABD[4, 4])
            return A11 * z**2 + D_local

        S[4, 4] = self._integrate_membrane(ei_x_contrib)

        # Bending stiffness about z-axis (chordwise bending)
        def ei_z_contrib(wall, x, z):
            ABD = self._wall_abd(wall)
            A11 = ABD[0, 0]
            if wall.is_horizontal:
                D_local = ABD[4, 4]  # D22
            elif wall.is_vertical:
                D_local = ABD[3, 3]  # D11
            else:
                D_local = 0.5 * (ABD[3, 3] + ABD[4, 4])
            return A11 * x**2 + D_local

        S[3, 3] = self._integrate_membrane(ei_z_contrib)

        # Bending-torsion coupling (D16/D26 integrated over perimeter)
        # This captures the bend-twist coupling that is central to VAM.
        def bend_torsion_couple(wall, x, z):
            ABD = self._wall_abd(wall)
            if wall.is_horizontal:
                return ABD[3, 5]  # D16
            elif wall.is_vertical:
                return ABD[4, 5]  # D26
            return 0.5 * (ABD[3, 5] + ABD[4, 5])

        S[4, 5] = self._integrate_membrane(bend_torsion_couple)
        S[3, 5] = S[4, 5]

        # Torsional stiffness of a closed thin-walled cell.
        # GJ = 4 * A_m^2 / ∮ (1 / A66) ds
        # where A66 is the laminate in-plane shear stiffness [N/m].
        enclosed_area = self.width * self.height
        torsion_integral = 0.0
        for wall in self.walls:
            A66 = self._wall_abd(wall)[2, 2]
            if A66 > 1e-12:
                torsion_integral += wall.length / A66
        if torsion_integral > 1e-12:
            S[5, 5] = 4.0 * enclosed_area**2 / torsion_integral

        # Cross-coupling between vertical and chordwise bending (zero for a
        # symmetric box, but retained for completeness)
        S[3, 4] = self._integrate_membrane(
            lambda wall, x, z: self._wall_abd(wall)[0, 0] * x * z
        )

        # Symmetrize
        S = 0.5 * (S + S.T)
        return S

    def equivalent_beam_properties(self) -> dict[str, float]:
        """Extract conventional beam properties from the 6x6 stiffness matrix.

        Returns:
            dict with EA, EI_x, EI_z, GJ, bend_twist_coupling, mass_per_length.
        """
        S = self.stiffness_matrix()
        mass_per_length = self.mass_per_length()
        return {
            "EA": float(S[0, 0]),
            "EI_x": float(S[4, 4]),
            "EI_z": float(S[3, 3]),
            "GJ": float(S[5, 5]),
            "bend_twist_coupling": float(S[4, 5]),
            "extension_bending_coupling": float(np.linalg.norm(S[0, 3:5])),
            "mass_per_length": float(mass_per_length),
        }

    def mass_per_length(self) -> float:
        """Structural mass per unit span [kg/m]."""
        total = 0.0
        for wall in self.walls:
            total += wall.length * wall.laminate.total_mass_per_area
        return float(total)


@dataclass
class VAMBeamProperties:
    """Spanwise beam properties computed with VAM section theory."""

    y: np.ndarray
    EI: np.ndarray
    GJ: np.ndarray
    EA: np.ndarray
    mass_per_length: np.ndarray
    chord: np.ndarray
    skin_thickness: np.ndarray
    bend_twist_coupling: np.ndarray


def compute_vam_beam_properties(
    span: float,
    n_stations: int,
    chord_distribution: np.ndarray,
    box_width_fraction: float,
    box_height_fraction: float,
    skin_laminate: Laminate,
    skin_thickness_scales: Optional[np.ndarray] = None,
    skin_laminates: Optional[list[Laminate]] = None,
    spar_laminate: Optional[Laminate] = None,
) -> VAMBeamProperties:
    """Compute spanwise VAM beam properties for a wing structure.

    Args:
        span: Semi-span [m].
        n_stations: Number of spanwise stations (nodes).
        chord_distribution: Chord at each station [m].
        box_width_fraction: Wing-box width as fraction of local chord.
        box_height_fraction: Wing-box height as fraction of local chord.
        skin_laminate: Baseline skin laminate.
        skin_thickness_scales: Optional per-station thickness multiplier.
        skin_laminates: Optional per-station skin laminates.
        spar_laminate: Optional spar laminate.

    Returns:
        ``VAMBeamProperties`` arrays.
    """
    y = np.linspace(0.0, span, n_stations)
    EI = np.zeros(n_stations)
    GJ = np.zeros(n_stations)
    EA = np.zeros(n_stations)
    mass_per_length = np.zeros(n_stations)
    skin_thickness = np.zeros(n_stations)
    bend_twist = np.zeros(n_stations)

    if skin_thickness_scales is None:
        skin_thickness_scales = np.ones(n_stations)

    use_per_station = skin_laminates is not None and len(skin_laminates) == n_stations

    for i, c in enumerate(chord_distribution):
        width = c * box_width_fraction
        height = c * box_height_fraction
        scale = skin_thickness_scales[i]

        lam = skin_laminates[i] if use_per_station and skin_laminates[i] is not None else skin_laminate
        if lam is None:
            continue

        # Scale the laminate by adjusting the effective ply thickness.
        # For a symmetric laminate we scale all plies by the same factor.
        if scale != 1.0:
            from .composite_properties import Laminate
            scaled = Laminate(
                material=lam.material,
                angles=lam.angles,
                symmetric=lam.symmetric,
            )
            # Increase ply thickness uniformly.  This is approximate but keeps
            # the stacking sequence unchanged.
            scaled.material = lam.material  # same material
            # We cannot mutate material.t_ply directly without side effects, so
            # build an equivalent material with scaled ply thickness.
            from copy import deepcopy
            mat_scaled = deepcopy(lam.material)
            mat_scaled.t_ply *= scale
            scaled.material = mat_scaled
            lam = scaled

        section = VAMSection(
            width=width,
            height=height,
            skin_laminate=lam,
            spar_laminate=spar_laminate,
        )
        props = section.equivalent_beam_properties()
        EI[i] = props["EI_x"]
        GJ[i] = props["GJ"]
        EA[i] = props["EA"]
        mass_per_length[i] = props["mass_per_length"]
        skin_thickness[i] = lam.total_thickness
        bend_twist[i] = props["bend_twist_coupling"]

    return VAMBeamProperties(
        y=y,
        EI=EI,
        GJ=GJ,
        EA=EA,
        mass_per_length=mass_per_length,
        chord=chord_distribution,
        skin_thickness=skin_thickness,
        bend_twist_coupling=bend_twist,
    )
