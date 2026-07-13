"""
VTK export utilities for PIML-MDO aerostructural results.

This module writes optimized wing geometry (airfoil + spanwise stations) as VTK
datasets that can be opened in ParaView.  Both undeformed and deformed wings are
exported, together with scalar fields for pressure coefficient, failure index,
laminate thickness and ply angles.

The implementation does **not** require the ``vtk`` Python package.  It writes
legacy ASCII ``.vtk`` files directly, which ParaView reads natively.  If the
``vtk`` package is available and functional it can optionally be used for XML
``.vtu`` output, but the legacy writer is the default because it is the most
portable (especially on Windows where system VTK packages are often incomplete).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import VTK, but do not fail if it is missing or broken.
try:
    import vtk

    HAS_VTK = True
except Exception:  # pragma: no cover - VTK is optional
    HAS_VTK = False
    vtk = None  # type: ignore


def _has_pvpython() -> bool:
    """Return True if a ParaView python executable is on PATH."""
    return shutil.which("pvpython") is not None or shutil.which("pvbatch") is not None


def _safe_mkdir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_airfoil_coords(
    airfoil_coords: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized chordwise coordinates in [0, 1] and a reference chord."""
    x_raw, y_raw = airfoil_coords
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    c_ref = float(np.max(x_raw) - np.min(x_raw))
    if c_ref < 1e-10:
        c_ref = 1.0
    x_norm = (x_raw - np.min(x_raw)) / c_ref
    y_norm = y_raw / c_ref
    return x_norm, y_norm, c_ref


def _expand_station_scalar(
    scalar: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Broadcast a per-station scalar to all points of a (nx, ny) structured grid."""
    arr = np.asarray(scalar, dtype=float)
    if arr.size == 1:
        return np.full(nx * ny, float(arr.flat[0]))
    if arr.size == ny:
        # x varies fastest in a C-ordered (nx, ny) grid -> repeat each station value nx times
        return np.repeat(arr, nx)
    if arr.size == nx * ny:
        return arr.flatten(order="C")
    # Fallback: scalar is constant over the whole surface
    logger.warning(
        "Scalar of size %d does not match nx=%d, ny=%d; using constant field.",
        arr.size,
        nx,
        ny,
    )
    return np.full(nx * ny, float(np.mean(arr)))


def _build_wing_surface_points(
    airfoil_coords: tuple[np.ndarray, np.ndarray],
    wing: Any,
    stations: np.ndarray,
    twist_deg: np.ndarray,
    deflection: np.ndarray,
) -> np.ndarray:
    """Build a (nx, ny, 3) point array for a wing structured surface."""
    x_norm, y_norm, _ = _normalize_airfoil_coords(airfoil_coords)
    nx = len(x_norm)
    ny = len(stations)

    chords = np.asarray(wing.chord_distribution())
    if len(chords) != ny:
        chords = np.interp(stations, wing.spanwise_stations(), chords)

    sweep_rad = np.radians(float(wing.sweep_deg))
    tan_sweep = np.tan(sweep_rad)

    twist_deg = np.asarray(twist_deg, dtype=float)
    if len(twist_deg) != ny:
        twist_deg = np.interp(stations, np.linspace(0, float(wing.span), len(twist_deg)), twist_deg)

    deflection = np.asarray(deflection, dtype=float)
    if len(deflection) != ny:
        deflection = np.interp(stations, np.linspace(0, float(wing.span), len(deflection)), deflection)

    points = np.zeros((nx, ny, 3), dtype=float)

    for j, y in enumerate(stations):
        chord_j = chords[j]
        x_qc = -y * tan_sweep

        # Place quarter-chord at origin of local airfoil section
        x_local = x_qc + (x_norm - 0.25) * chord_j
        z_local = y_norm * chord_j

        theta = np.radians(twist_deg[j])
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        points[:, j, 0] = x_local * cos_t + z_local * sin_t
        points[:, j, 1] = y
        points[:, j, 2] = -x_local * sin_t + z_local * cos_t + deflection[j]

    return points


def _scalar_fields(
    coupling_result: dict[str, Any],
    wing: Any,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    nx: int,
    ny: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return point scalar and vector fields for the wing surface."""
    aero = coupling_result.get("aero", {})
    struct = coupling_result.get("structure", None)

    stations = wing.spanwise_stations()
    if struct is not None:
        stations = np.asarray(struct.y)

    # Per-station aerodynamic coefficients
    cl = np.asarray(aero.get("cl", np.zeros(ny)))
    cd = np.asarray(aero.get("cd", np.zeros(ny)))
    cm = np.asarray(aero.get("cm", np.zeros(ny)))
    if len(cl) != ny:
        cl = _expand_to_ny(cl, ny)
    if len(cd) != ny:
        cd = _expand_to_ny(cd, ny)
    if len(cm) != ny:
        cm = _expand_to_ny(cm, ny)

    # Pressure coefficient.
    vlm = coupling_result.get("vlm", None)
    if vlm is not None and getattr(vlm, "cp_panels", None) is not None:
        # Real mapping from the 3-D VLM per-panel loading ΔCp. The loading is
        # ΔCp = Cp_lower - Cp_upper on the lifting surface; we split it onto the
        # airfoil surfaces as -ΔCp/2 (upper) and +ΔCp/2 (lower).
        cp = _vlm_surface_cp(vlm, x_norm, y_norm, stations)
    else:
        # Fallback (strip-theory path): synthetic distribution using section CL.
        # Upper surface -> negative Cp, lower surface -> positive Cp.
        sin_factor = np.sin(np.clip(x_norm, 0.0, 1.0) * np.pi)
        cp_profile = np.where(y_norm >= 0.0, -sin_factor, sin_factor)
        cp = np.outer(cp_profile, cl).flatten(order="C")

    # Failure index per station (von Mises-like stress / allowable)
    sigma_allow = 400e6
    if wing.skin_laminate is not None and hasattr(wing.skin_laminate.material, "Xt"):
        sigma_allow = wing.skin_laminate.material.Xt

    if struct is not None:
        bending_stress = np.asarray(struct.bending_stress)
        shear_stress = np.asarray(struct.shear_stress)
        vm = np.sqrt(bending_stress**2 + 3.0 * shear_stress**2)
        fi_station = vm / sigma_allow
    else:
        fi_station = np.zeros(ny)
    failure_index = _expand_station_scalar(fi_station, nx, ny)

    # Laminate / skin thickness per station
    if wing.skin_laminate is not None and hasattr(wing.skin_laminate, "total_thickness"):
        thickness_station = wing.skin_laminate.total_thickness
    else:
        thickness_station = 0.003
    laminate_thickness = _expand_station_scalar(np.full(ny, float(thickness_station)), nx, ny)

    # Ply angle: mean absolute ply angle if a laminate exists
    if wing.skin_laminate is not None and hasattr(wing.skin_laminate, "full_angles"):
        ply_angles = np.asarray(wing.skin_laminate.full_angles, dtype=float)
        ply_angle_station = np.mean(np.abs(ply_angles))
    else:
        ply_angle_station = 0.0
    ply_angle = _expand_station_scalar(np.full(ny, float(ply_angle_station)), nx, ny)

    scalars = {
        "pressure_coefficient": cp,
        "failure_index": failure_index,
        "laminate_thickness": laminate_thickness,
        "ply_angle": ply_angle,
        "cl": _expand_station_scalar(cl, nx, ny),
        "cd": _expand_station_scalar(cd, nx, ny),
        "cm": _expand_station_scalar(cm, nx, ny),
    }

    return scalars, stations


def _vlm_surface_cp(
    vlm: Any,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    stations: np.ndarray,
) -> np.ndarray:
    """Map the real 3-D VLM per-panel loading (ΔCp) onto the wing surface.

    ``vlm.cp_panels`` has shape ``(n_chord_panels, n_span_panels)`` and is the
    lifting-surface loading ΔCp = Cp_lower − Cp_upper. For each surface point we
    interpolate ΔCp spanwise (to the point's span station) and chordwise (to the
    point's x/c), then split it as −ΔCp/2 on the upper surface and +ΔCp/2 on the
    lower surface. Returns a flat (nx*ny,) array in C order matching the grid.
    """
    cp_panels = np.asarray(vlm.cp_panels, dtype=float)   # (nc, ns)
    nc, ns = cp_panels.shape

    # Chordwise panel-center positions in x/c and spanwise panel-center y.
    xi_edges = np.linspace(0.0, 1.0, nc + 1)
    xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])
    y_panel = np.asarray(vlm.y, dtype=float)
    if y_panel.size != ns:
        y_panel = np.linspace(float(stations[0]), float(stations[-1]), ns)

    nx = len(x_norm)
    ny = len(stations)
    cp = np.zeros((nx, ny), dtype=float)

    for j, y in enumerate(stations):
        # Spanwise interpolation: ΔCp(x_panel) at this station.
        dcp_chord = np.array([
            np.interp(y, y_panel, cp_panels[k, :]) for k in range(nc)
        ])
        # Chordwise interpolation onto the airfoil x/c stations.
        dcp = np.interp(np.clip(x_norm, 0.0, 1.0), xi_centers, dcp_chord)
        # Split loading onto upper (−ΔCp/2) and lower (+ΔCp/2) surfaces.
        cp[:, j] = np.where(y_norm >= 0.0, -0.5 * dcp, 0.5 * dcp)

    return cp.flatten(order="C")


def _expand_to_ny(arr: np.ndarray, ny: int) -> np.ndarray:
    """Interpolate or repeat a 1-D array to length ny."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == ny:
        return arr
    if arr.size == 1:
        return np.full(ny, float(arr.flat[0]))
    old_y = np.linspace(0.0, 1.0, arr.size)
    new_y = np.linspace(0.0, 1.0, ny)
    return np.interp(new_y, old_y, arr)


def _write_unstructured_grid_legacy(
    path: Path,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
    vectors: Optional[dict[str, np.ndarray]] = None,
    title: str = "Wing surface",
) -> Path:
    """Write a legacy VTK unstructured grid file (quadrilateral wing surface)."""
    nx, ny, _ = points.shape
    npoints = nx * ny
    ncells = (nx - 1) * (ny - 1)

    with open(path, "w", newline="") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{title}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {npoints} float\n")

        pts = points.reshape(-1, 3)
        for p in pts:
            f.write(f"{p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n")

        # Quadrilateral cells
        f.write(f"\nCELLS {ncells} {ncells * 5}\n")
        for j in range(ny - 1):
            for i in range(nx - 1):
                i0 = i + j * nx
                i1 = i0 + 1
                i2 = i1 + nx
                i3 = i0 + nx
                f.write(f"4 {i0} {i1} {i2} {i3}\n")

        f.write(f"\nCELL_TYPES {ncells}\n")
        for _ in range(ncells):
            f.write("9 ")
        f.write("\n")

        f.write(f"\nPOINT_DATA {npoints}\n")

        if vectors:
            for name, arr in vectors.items():
                safe_name = name.replace(" ", "_")
                f.write(f"VECTORS {safe_name} float\n")
                vec = arr.reshape(-1, 3)
                for v in vec:
                    f.write(f"{v[0]:.8e} {v[1]:.8e} {v[2]:.8e}\n")
                f.write("\n")

        for name, arr in scalars.items():
            safe_name = name.replace(" ", "_")
            f.write(f"SCALARS {safe_name} float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for val in arr.flatten(order="C"):
                f.write(f"{val:.8e}\n")
            f.write("\n")

    return path


def _write_xml_vtu(
    path: Path,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
    vectors: Optional[dict[str, np.ndarray]] = None,
    title: str = "Wing surface",
) -> Path:
    """Optional XML VTU writer using the VTK Python package."""
    if not HAS_VTK or vtk is None:
        raise ImportError("VTK Python package is required for .vtu output")

    nx, ny, _ = points.shape
    npoints = nx * ny

    structured_points = vtk.vtkStructuredGrid()
    structured_points.SetDimensions(nx, ny, 1)

    vtk_points = vtk.vtkPoints()
    pts = points.reshape(-1, 3)
    for p in pts:
        vtk_points.InsertNextPoint(p[0], p[1], p[2])
    structured_points.SetPoints(vtk_points)

    def _add_array(data, name: str, values: np.ndarray, n_components: int = 1):
        arr = vtk.vtkFloatArray()
        arr.SetName(name)
        arr.SetNumberOfComponents(n_components)
        arr.SetNumberOfTuples(len(values) // n_components)
        flat = values.flatten(order="C")
        for i, v in enumerate(flat):
            arr.SetValue(i, float(v))
        data.AddArray(arr)

    point_data = structured_points.GetPointData()
    if vectors:
        for name, arr in vectors.items():
            _add_array(point_data, name, arr, 3)
    for name, arr in scalars.items():
        _add_array(point_data, name, arr, 1)

    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(structured_points)
    writer.Write()
    return path


def _write_surface(
    path: Path,
    points: np.ndarray,
    scalars: dict[str, np.ndarray],
    vectors: Optional[dict[str, np.ndarray]] = None,
    title: str = "Wing surface",
    prefer_xml: bool = False,
) -> Path:
    """Write a wing surface to VTK (legacy .vtk or XML .vtu)."""
    if prefer_xml and HAS_VTK:
        return _write_xml_vtu(path, points, scalars, vectors, title)
    return _write_unstructured_grid_legacy(path, points, scalars, vectors, title)


def export_wing_surface(
    coupling_result: dict[str, Any],
    wing: Any,
    filename: str,
    deformed: bool = True,
) -> Path:
    """Export a single wing surface (undeformed or deformed) to VTK.

    Parameters
    ----------
    coupling_result : dict
        Result dictionary from :class:`AerostructuralCoupler.solve`.  Must contain
        ``aero``, ``structure`` and (for deformed surfaces) ``final_twist``.
    wing : WingStructure
        Wing geometry / structural definition.
    filename : str
        Destination file path.  The extension is normalised to ``.vtk`` unless
        XML output is requested.
    deformed : bool, optional
        If True, apply structural twist and vertical deflection.  Otherwise use
        the initial geometric twist and zero deflection.

    Returns
    -------
    Path
        Path to the written file.
    """
    path = Path(filename)
    _safe_mkdir(path)

    # Normalise extension: default to legacy .vtk
    if path.suffix.lower() not in {".vtk", ".vtu"}:
        path = path.with_suffix(".vtk")
    prefer_xml = path.suffix.lower() == ".vtu"

    struct = coupling_result.get("structure", None)
    if struct is not None:
        stations = np.asarray(struct.y)
    else:
        stations = wing.spanwise_stations()
    ny = len(stations)

    airfoil_coords = coupling_result.get(
        "airfoil_coords",
        (np.linspace(0.0, 1.0, 100), np.zeros(100)),
    )
    x_norm, y_norm, _ = _normalize_airfoil_coords(airfoil_coords)
    nx = len(x_norm)

    if deformed:
        twist_deg = coupling_result.get("final_twist", np.zeros(ny))
        deflection = struct.deflection if struct is not None else np.zeros(ny)
        title = "Deformed wing surface"
    else:
        twist_deg = coupling_result.get("initial_twist", np.zeros(ny))
        deflection = np.zeros(ny)
        title = "Undeformed wing surface"

    points = _build_wing_surface_points(
        airfoil_coords, wing, stations, twist_deg, deflection
    )

    scalars, _ = _scalar_fields(coupling_result, wing, x_norm, y_norm, nx, ny)

    # Displacement vector for deformed surface
    vectors: Optional[dict[str, np.ndarray]] = None
    if deformed and struct is not None:
        disp = np.zeros((nx, ny, 3), dtype=float)
        disp[:, :, 2] = np.asarray(struct.deflection)
        vectors = {"displacement": disp.reshape(nx, ny, 3)}

    written = _write_surface(path, points, scalars, vectors, title, prefer_xml)
    logger.info("  Exported %s wing to %s", "deformed" if deformed else "undeformed", written)
    return written


def export_aerostructural_result(
    coupling_result: dict[str, Any],
    wing: Any,
    filename: str,
) -> list[Path]:
    """Export undeformed and deformed wing geometry from an aerostructural result.

    Parameters
    ----------
    coupling_result : dict
        Coupled aero/struct result dictionary.
    wing : WingStructure
        Wing geometry / structural definition.
    filename : str
        Base file path (without ``_undeformed`` / ``_deformed`` suffix and without
        extension).  For example ``results/run_001/wing`` will create
        ``wing_undeformed.vtk`` and ``wing_deformed.vtk``.

    Returns
    -------
    list[Path]
        Paths to the written VTK files.
    """
    base = Path(filename)
    if base.suffix.lower() in {".vtk", ".vtu"}:
        base = base.with_suffix("")

    undeformed_path = base.parent / f"{base.name}_undeformed.vtk"
    deformed_path = base.parent / f"{base.name}_deformed.vtk"

    files = []
    files.append(export_wing_surface(coupling_result, wing, str(undeformed_path), deformed=False))
    files.append(export_wing_surface(coupling_result, wing, str(deformed_path), deformed=True))
    return files
