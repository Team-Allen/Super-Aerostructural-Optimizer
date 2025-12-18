"""
Utility to build a Wing3D from two airfoil coordinate files (root and tip)
and simple helpers to save section coordinates.

This script is a lightweight bridge: it reads optimized airfoil `.dat` files
or falls back to the internal NACA generator in `Wing3D`.
"""
from pathlib import Path
import numpy as np
from typing import Optional

from .wing_geometry import WingParameters, Wing3D


def load_airfoil_dat(path: Path) -> np.ndarray:
    """Load a simple two-column x,y airfoil `.dat` file into an array.

    Expects columns separated by whitespace or commas. Returns [N,2] array.
    """
    if not path.exists():
        raise FileNotFoundError(f"Airfoil file not found: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(-1, 2)
    return data


def create_wing_from_airfoil_files(
    root_dat: Optional[str],
    tip_dat: Optional[str],
    params: Optional[WingParameters] = None
):
    """Create a Wing3D object using provided airfoil files.

    If an airfoil path is None, the internal NACA fallback will be used.
    """
    wp = params if params is not None else WingParameters()

    # If files provided, place them into the project airfoil database folder
    # and set the names so Wing3D._load_airfoil can find them. For simplicity
    # we'll override Wing3D._load_airfoil by temporarily monkey-patching
    # the Wing3D instance to return the provided coordinates.

    wing = Wing3D(wp)

    # If root_dat/tip_dat provided, replace airfoil_coords on generated sections
    if root_dat:
        root_coords = load_airfoil_dat(Path(root_dat))
        # replace first half of sections with root, second half with tip if available
        for s in wing.sections:
            s.airfoil_coords = root_coords

    if tip_dat:
        tip_coords = load_airfoil_dat(Path(tip_dat))
        # for outer sections (near tip) replace with tip_coords
        n = len(wing.sections)
        for i, s in enumerate(wing.sections):
            eta = s.y_position / (wing.params.span / 2)
            if eta > 0.6:
                s.airfoil_coords = tip_coords

    return wing


if __name__ == "__main__":
    print("Building wing from airfoil files (example)")
    params = WingParameters(span=8.0, root_chord=1.5, taper_ratio=0.6, n_sections=20)
    wing = create_wing_from_airfoil_files(None, None, params)
    print(f"Created wing with {len(wing.sections)} sections")
