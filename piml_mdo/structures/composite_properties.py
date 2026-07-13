"""
Composite laminate properties using Classical Lamination Theory (CLT).

Computes the ABD stiffness matrix for composite laminates from:
- Ply material properties (E1, E2, G12, nu12)
- Ply angles (stacking sequence)
- Ply thicknesses

This gives the equivalent beam stiffness (EI, GJ) for the wing box,
accounting for ply orientation effects like bend-twist coupling.

Reference: Jones, R.M. "Mechanics of Composite Materials"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlyMaterial:
    """Unidirectional ply material properties."""
    name: str
    E1: float     # Longitudinal modulus [Pa]
    E2: float     # Transverse modulus [Pa]
    G12: float    # In-plane shear modulus [Pa]
    nu12: float   # Major Poisson's ratio
    rho: float    # Density [kg/m³]
    t_ply: float  # Ply thickness [m]

    # Strength allowables
    Xt: float = 0.0   # Longitudinal tensile strength [Pa]
    Xc: float = 0.0   # Longitudinal compressive strength [Pa]
    Yt: float = 0.0   # Transverse tensile strength [Pa]
    Yc: float = 0.0   # Transverse compressive strength [Pa]
    S12: float = 0.0   # In-plane shear strength [Pa]

    @property
    def nu21(self) -> float:
        return self.nu12 * self.E2 / self.E1

    @property
    def Q(self) -> np.ndarray:
        """Reduced stiffness matrix Q in material coordinates [3x3]."""
        nu21 = self.nu21
        denom = 1.0 - self.nu12 * nu21
        Q = np.array([
            [self.E1 / denom,         self.nu12 * self.E2 / denom, 0.0],
            [self.nu12 * self.E2 / denom, self.E2 / denom,         0.0],
            [0.0,                      0.0,                      self.G12],
        ])
        return Q


# Common aerospace materials
CFRP_T300_5208 = PlyMaterial(
    name="T300/5208 CFRP",
    E1=181e9, E2=10.3e9, G12=7.17e9, nu12=0.28,
    rho=1600.0, t_ply=0.000125,
    Xt=1500e6, Xc=1500e6, Yt=40e6, Yc=246e6, S12=68e6,
)

CFRP_IM7_8552 = PlyMaterial(
    name="IM7/8552 CFRP",
    E1=171e9, E2=9.08e9, G12=5.29e9, nu12=0.32,
    rho=1580.0, t_ply=0.000131,
    Xt=2326e6, Xc=1200e6, Yt=62.3e6, Yc=199.8e6, S12=92.3e6,
)

AL_7075_T6 = PlyMaterial(
    name="Al 7075-T6 (isotropic)",
    E1=71.7e9, E2=71.7e9, G12=26.9e9, nu12=0.33,
    rho=2810.0, t_ply=0.001,
    Xt=572e6, Xc=572e6, Yt=572e6, Yc=572e6, S12=331e6,
)


@dataclass
class Laminate:
    """Composite laminate defined by stacking sequence and material."""
    material: PlyMaterial
    angles: list[float]  # Ply angles in degrees [θ₁, θ₂, ...]
    symmetric: bool = True  # If True, angles define only the top half

    @property
    def full_angles(self) -> list[float]:
        if self.symmetric:
            return self.angles + self.angles[::-1]
        return self.angles

    @property
    def n_plies(self) -> int:
        return len(self.full_angles)

    @property
    def total_thickness(self) -> float:
        return self.n_plies * self.material.t_ply

    @property
    def total_mass_per_area(self) -> float:
        """Mass per unit area [kg/m²]."""
        return self.total_thickness * self.material.rho

    def Q_bar(self, theta_deg: float) -> np.ndarray:
        """Transformed reduced stiffness matrix for ply at angle theta."""
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)

        Q = self.material.Q

        # Transformation matrix T
        T = np.array([
            [c**2,    s**2,     2*s*c],
            [s**2,    c**2,    -2*s*c],
            [-s*c,    s*c,      c**2 - s**2],
        ])

        T_inv = np.array([
            [c**2,    s**2,    -2*s*c],
            [s**2,    c**2,     2*s*c],
            [s*c,    -s*c,      c**2 - s**2],
        ])

        # Reuter matrix for engineering strain conversion
        R = np.diag([1.0, 1.0, 2.0])
        R_inv = np.diag([1.0, 1.0, 0.5])

        Q_bar = T_inv @ Q @ R @ T @ R_inv
        return Q_bar

    def ABD_matrix(self) -> np.ndarray:
        """Compute the full 6x6 ABD stiffness matrix.

        [N]   [A  B] [ε⁰]
        [M] = [B  D] [κ ]

        A: extensional stiffness (in-plane)
        B: coupling stiffness (extension-bending coupling)
        D: bending stiffness

        Returns:
            ABD: [6, 6] stiffness matrix
        """
        angles = self.full_angles
        n = len(angles)
        t = self.material.t_ply
        h_total = n * t

        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for k, theta in enumerate(angles):
            Qb = self.Q_bar(theta)

            # z-coordinates of ply boundaries (measured from midplane)
            z_bot = -h_total / 2.0 + k * t
            z_top = z_bot + t

            A += Qb * (z_top - z_bot)
            B += 0.5 * Qb * (z_top**2 - z_bot**2)
            D += (1.0 / 3.0) * Qb * (z_top**3 - z_bot**3)

        ABD = np.zeros((6, 6))
        ABD[:3, :3] = A
        ABD[:3, 3:] = B
        ABD[3:, :3] = B
        ABD[3:, 3:] = D
        return ABD

    def equivalent_beam_properties(self, width: float) -> dict[str, float]:
        """Compute equivalent beam EI, GJ, EA for a wing box cross-section.

        Assumes a rectangular box section of given width and laminate thickness.

        Args:
            width: wing box width (chord fraction * chord) [m]

        Returns:
            dict with EI, GJ, EA, EI_coupled (bend-twist coupling)
        """
        ABD = self.ABD_matrix()
        A = ABD[:3, :3]
        B = ABD[:3, 3:]
        D = ABD[3:, 3:]

        # Effective bending stiffness per unit width: D11
        # For beam: EI = D11 * width
        EI = float(D[0, 0] * width)

        # Effective torsional stiffness: D66
        GJ = float(D[2, 2] * width)

        # Extensional stiffness
        EA = float(A[0, 0] * width)

        # Bend-twist coupling (B16 or D16)
        EI_coupled = float(D[0, 2] * width)

        return {
            'EI': EI,           # Bending stiffness [N·m²]
            'GJ': GJ,           # Torsional stiffness [N·m²]
            'EA': EA,           # Extensional stiffness [N]
            'EI_coupled': EI_coupled,  # Bend-twist coupling [N·m²]
            'mass_per_length': self.total_mass_per_area * width,  # [kg/m]
        }

    def tsai_wu_failure(self, stress: np.ndarray) -> np.ndarray:
        """Tsai-Wu failure criterion for each ply.

        Args:
            stress: [n_plies, 3] stress in material coords (σ₁, σ₂, τ₁₂)

        Returns:
            failure_index: [n_plies] — values > 1.0 indicate failure
        """
        mat = self.material
        F1 = 1.0 / mat.Xt - 1.0 / mat.Xc
        F2 = 1.0 / mat.Yt - 1.0 / mat.Yc
        F11 = 1.0 / (mat.Xt * mat.Xc)
        F22 = 1.0 / (mat.Yt * mat.Yc)
        F66 = 1.0 / mat.S12**2
        F12 = -0.5 * np.sqrt(F11 * F22)  # Tsai-Wu interaction term

        s1 = stress[:, 0]
        s2 = stress[:, 1]
        t12 = stress[:, 2]

        fi = (F1 * s1 + F2 * s2 +
              F11 * s1**2 + F22 * s2**2 + F66 * t12**2 +
              2.0 * F12 * s1 * s2)
        return fi


# Common layup patterns
def quasi_isotropic(material: PlyMaterial = CFRP_T300_5208) -> Laminate:
    """[0/45/-45/90]s quasi-isotropic layup."""
    return Laminate(material=material, angles=[0, 45, -45, 90], symmetric=True)


def optimized_wing_skin(material: PlyMaterial = CFRP_IM7_8552) -> Laminate:
    """[±45/0₂/90/0₂/±45]s — typical wing skin layup."""
    return Laminate(
        material=material,
        angles=[45, -45, 0, 0, 90, 0, 0, 45, -45],
        symmetric=True
    )


def spar_cap(material: PlyMaterial = CFRP_IM7_8552) -> Laminate:
    """[0₄/±45/0₂]s — spar cap (bending-dominated)."""
    return Laminate(
        material=material,
        angles=[0, 0, 0, 0, 45, -45, 0, 0],
        symmetric=True
    )


def thick_wing_skin(material: PlyMaterial = CFRP_IM7_8552) -> Laminate:
    """[±45/0₄/90/0₄/±45]s — thick wing skin for high-load 15 m semi-span wing."""
    return Laminate(
        material=material,
        angles=[45, -45, 0, 0, 0, 0, 90, 0, 0, 0, 0, 45, -45],
        symmetric=True
    )
