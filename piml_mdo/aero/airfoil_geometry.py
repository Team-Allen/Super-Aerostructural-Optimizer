"""
Airfoil geometry parameterization using Class-Shape Transformation (CST) method.

CST is the standard parameterization for gradient-based airfoil optimization:
- Compact: 8-16 parameters define the full airfoil shape
- Smooth: C2-continuous shapes guaranteed
- Differentiable: analytical gradients w.r.t. shape parameters
- Physically meaningful: thickness, camber, LE radius all controllable

Reference: Kulfan, B.M. (2008) "Universal Parametric Geometry Representation Method"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AirfoilGeometry:
    """CST-parameterized airfoil geometry."""

    # CST shape parameters (Bernstein polynomial weights)
    upper_weights: np.ndarray  # Upper surface CST weights [N_upper]
    lower_weights: np.ndarray  # Lower surface CST weights [N_lower]

    # Trailing edge thickness (half, applied symmetrically)
    dte_upper: float = 0.0
    dte_lower: float = 0.0

    # Leading edge class function exponents
    n1: float = 0.5  # LE radius control
    n2: float = 1.0  # TE angle control

    # Chord length [m]
    chord: float = 1.0

    name: str = "cst_airfoil"

    @property
    def n_params(self) -> int:
        return len(self.upper_weights) + len(self.lower_weights)

    @property
    def design_vector(self) -> np.ndarray:
        """Flat design vector for optimizer."""
        return np.concatenate([self.upper_weights, self.lower_weights])

    @classmethod
    def from_design_vector(cls, x: np.ndarray, n_upper: int = 6, **kwargs) -> "AirfoilGeometry":
        """Create from flat optimizer design vector."""
        n_lower = len(x) - n_upper
        return cls(
            upper_weights=x[:n_upper].copy(),
            lower_weights=x[n_upper:].copy(),
            **kwargs
        )

    @classmethod
    def naca4(cls, code: str = "2412", n_weights: int = 6) -> "AirfoilGeometry":
        """Create CST approximation of a NACA 4-digit airfoil."""
        m = int(code[0]) / 100.0  # max camber
        p = int(code[1]) / 10.0   # camber location
        t = int(code[2:]) / 100.0  # max thickness

        # Generate NACA coordinates
        x = _cosine_spacing(100)
        yt = _naca4_thickness(x, t)
        yc = _naca4_camber(x, m, p)
        dyc = _naca4_camber_gradient(x, m, p)

        theta = np.arctan(dyc)
        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Fit CST weights to NACA coordinates
        upper_w = _fit_cst_weights(xu, yu, n_weights)
        lower_w = _fit_cst_weights(xl, yl, n_weights)

        return cls(
            upper_weights=upper_w,
            lower_weights=lower_w,
            name=f"NACA{code}_cst"
        )

    def coordinates(self, n_points: int = 150) -> tuple[np.ndarray, np.ndarray]:
        """Generate upper and lower surface coordinates.

        Returns:
            x: [n_points] chordwise coordinates (0 to 1)
            y_upper: [n_points] upper surface y
            y_lower: [n_points] lower surface y
        """
        x = _cosine_spacing(n_points)
        y_upper = self._cst_surface(x, self.upper_weights, self.dte_upper)
        y_lower = self._cst_surface(x, self.lower_weights, self.dte_lower)
        return x, y_upper, y_lower

    def full_coordinates(self, n_points: int = 150) -> tuple[np.ndarray, np.ndarray]:
        """Generate full airfoil loop (TE → LE via upper, LE → TE via lower)."""
        x, y_upper, y_lower = self.coordinates(n_points)
        x_full = np.concatenate([x[::-1], x[1:]])
        y_full = np.concatenate([y_upper[::-1], y_lower[1:]])
        return x_full * self.chord, y_full * self.chord

    def thickness_at(self, x: np.ndarray) -> np.ndarray:
        """Thickness distribution t(x) = y_upper(x) - y_lower(x)."""
        y_u = self._cst_surface(x, self.upper_weights, self.dte_upper)
        y_l = self._cst_surface(x, self.lower_weights, self.dte_lower)
        return y_u - y_l

    def max_thickness(self) -> float:
        """Maximum thickness ratio t/c."""
        x = np.linspace(0, 1, 200)
        return float(np.max(self.thickness_at(x)))

    def camber_at(self, x: np.ndarray) -> np.ndarray:
        """Camber line y_c(x) = (y_upper + y_lower) / 2."""
        y_u = self._cst_surface(x, self.upper_weights, self.dte_upper)
        y_l = self._cst_surface(x, self.lower_weights, self.dte_lower)
        return (y_u + y_l) / 2.0

    def max_camber(self) -> float:
        """Maximum camber ratio."""
        x = np.linspace(0, 1, 200)
        return float(np.max(np.abs(self.camber_at(x))))

    def _cst_surface(self, x: np.ndarray, weights: np.ndarray, dte: float) -> np.ndarray:
        """Evaluate CST surface: y(x) = C(x) * S(x) + x * dte."""
        # Class function: C(x) = x^n1 * (1-x)^n2
        C = np.power(x, self.n1) * np.power(1.0 - x, self.n2)

        # Shape function: S(x) = sum(w_i * B_i(x))  (Bernstein basis)
        S = _bernstein_shape(x, weights)

        return C * S + x * dte

    def cst_gradient(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Analytical gradient dy/dw for each CST weight.

        Returns:
            dy_upper_dw: [n_upper, n_points] gradient of upper surface w.r.t. upper weights
            dy_lower_dw: [n_lower, n_points] gradient of lower surface w.r.t. lower weights
        """
        C = np.power(x, self.n1) * np.power(1.0 - x, self.n2)
        n_u = len(self.upper_weights)
        n_l = len(self.lower_weights)

        # dy/dw_i = C(x) * B_i(x)  (Bernstein basis polynomial i)
        dy_upper_dw = np.zeros((n_u, len(x)))
        for i in range(n_u):
            dy_upper_dw[i] = C * _bernstein_basis(x, i, n_u - 1)

        dy_lower_dw = np.zeros((n_l, len(x)))
        for i in range(n_l):
            dy_lower_dw[i] = C * _bernstein_basis(x, i, n_l - 1)

        return dy_upper_dw, dy_lower_dw


def _cosine_spacing(n: int) -> np.ndarray:
    """Cosine-spaced points from 0 to 1 (denser at LE and TE)."""
    beta = np.linspace(0, np.pi, n)
    return (1.0 - np.cos(beta)) / 2.0


def _bernstein_basis(x: np.ndarray, k: int, n: int) -> np.ndarray:
    """Bernstein basis polynomial B_{k,n}(x)."""
    from math import comb
    return comb(n, k) * np.power(x, k) * np.power(1.0 - x, n - k)


def _bernstein_shape(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Bernstein polynomial shape function S(x) = sum(w_i * B_i(x))."""
    n = len(weights) - 1
    S = np.zeros_like(x)
    for i, w in enumerate(weights):
        S += w * _bernstein_basis(x, i, n)
    return S


def _fit_cst_weights(x_data: np.ndarray, y_data: np.ndarray, n_weights: int) -> np.ndarray:
    """Fit CST weights to given airfoil coordinates via least squares."""
    # Class function
    C = np.power(np.clip(x_data, 1e-10, 1.0 - 1e-10), 0.5) * \
        np.power(1.0 - np.clip(x_data, 1e-10, 1.0 - 1e-10), 1.0)

    # Build Bernstein basis matrix
    n = n_weights - 1
    A = np.zeros((len(x_data), n_weights))
    for i in range(n_weights):
        A[:, i] = C * _bernstein_basis(x_data, i, n)

    # Least squares fit
    weights, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
    return weights


def _naca4_thickness(x: np.ndarray, t: float) -> np.ndarray:
    """NACA 4-digit thickness distribution."""
    return 5.0 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
        0.2843 * x**3 - 0.1015 * x**4
    )


def _naca4_camber(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """NACA 4-digit camber line."""
    if m == 0 or p == 0:
        return np.zeros_like(x)
    yc = np.where(
        x <= p,
        (m / p**2) * (2.0 * p * x - x**2),
        (m / (1.0 - p)**2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2)
    )
    return yc


def _naca4_camber_gradient(x: np.ndarray, m: float, p: float) -> np.ndarray:
    """Gradient of NACA 4-digit camber line."""
    if m == 0 or p == 0:
        return np.zeros_like(x)
    dyc = np.where(
        x <= p,
        (2.0 * m / p**2) * (p - x),
        (2.0 * m / (1.0 - p)**2) * (p - x)
    )
    return dyc
