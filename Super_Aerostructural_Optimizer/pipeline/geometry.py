"""Geometry and mesh utilities for the aerostructural pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import WingGeometryConfig


@dataclass
class WingMesh:
    """Structured half-wing mesh used by both aero and structure solvers."""

    nodes: np.ndarray
    node_index_grid: np.ndarray
    node_id_grid: np.ndarray
    y_rows: np.ndarray
    chord_rows: np.ndarray
    x_le_rows: np.ndarray
    base_row_twist_deg: np.ndarray
    area_half_m2: float
    half_span_m: float
    n_span: int
    n_chord: int

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def tip_node_indices(self) -> np.ndarray:
        return self.node_index_grid[-1, :]


@dataclass
class SpanwiseAeroGrid:
    """Spanwise strips used for aerodynamic force prediction."""

    y_panel_mid: np.ndarray
    dy_panel: np.ndarray
    chord_panel: np.ndarray
    x_le_panel: np.ndarray
    twist_panel_deg: np.ndarray
    area_half_m2: float
    half_span_m: float


def _linear(a: float, b: float, eta: np.ndarray | float) -> np.ndarray | float:
    return a + (b - a) * eta


def _row_twist_from_nodes(nodes: np.ndarray, node_index_grid: np.ndarray) -> np.ndarray:
    n_rows = node_index_grid.shape[0]
    twist = np.zeros(n_rows, dtype=float)
    for row in range(n_rows):
        i_le = int(node_index_grid[row, 0])
        i_te = int(node_index_grid[row, -1])
        dx = float(nodes[i_te, 0] - nodes[i_le, 0])
        dz = float(nodes[i_te, 2] - nodes[i_le, 2])
        twist[row] = np.degrees(np.arctan2(dz, dx))
    return twist


def build_half_wing_mesh(config: WingGeometryConfig) -> WingMesh:
    """Generate a structured half-wing mesh with sweep/dihedral/twist."""
    n_span = int(config.n_span)
    n_chord = int(config.n_chord)
    half_span = float(config.half_span_m)
    root_chord = float(config.root_chord_m)
    tip_chord = float(config.root_chord_m * config.taper_ratio)
    sweep = np.radians(float(config.sweep_deg))
    dihedral = np.radians(float(config.dihedral_deg))

    eta_rows = np.linspace(0.0, 1.0, n_span + 1)
    y_rows = eta_rows * half_span
    chord_rows = _linear(root_chord, tip_chord, eta_rows)
    x_le_rows = np.tan(sweep) * y_rows
    twist_rows_deg = _linear(config.twist_root_deg, config.twist_tip_deg, eta_rows)
    twist_rows = np.radians(twist_rows_deg)

    nodes = np.zeros(((n_span + 1) * (n_chord + 1), 3), dtype=float)
    node_index_grid = np.zeros((n_span + 1, n_chord + 1), dtype=int)

    idx = 0
    for j in range(n_span + 1):
        y = y_rows[j]
        c = chord_rows[j]
        x_le = x_le_rows[j]
        z_dihedral = y * np.tan(dihedral)
        twist = twist_rows[j]
        x_quarter = x_le + 0.25 * c

        for i in range(n_chord + 1):
            xsi = i / float(n_chord)
            x = x_le + c * xsi
            z = z_dihedral

            # Twist section around quarter chord.
            x_rel = x - x_quarter
            x_rot = x_quarter + x_rel * np.cos(twist)
            z_rot = z + x_rel * np.sin(twist)

            nodes[idx, :] = np.array([x_rot, y, z_rot], dtype=float)
            node_index_grid[j, i] = idx
            idx += 1

    # Planform area of half-wing by trapezoidal integration.
    area_half = float(np.trapezoid(chord_rows, y_rows))
    node_id_grid = node_index_grid + 1
    base_row_twist_deg = _row_twist_from_nodes(nodes, node_index_grid)

    return WingMesh(
        nodes=nodes,
        node_index_grid=node_index_grid,
        node_id_grid=node_id_grid,
        y_rows=y_rows,
        chord_rows=chord_rows,
        x_le_rows=x_le_rows,
        base_row_twist_deg=base_row_twist_deg,
        area_half_m2=area_half,
        half_span_m=half_span,
        n_span=n_span,
        n_chord=n_chord,
    )


def build_spanwise_aero_grid(
    mesh: WingMesh,
    twist_root_delta_deg: float = 0.0,
    twist_tip_delta_deg: float = 0.0,
    aeroelastic_twist_row_deg: np.ndarray | None = None,
) -> SpanwiseAeroGrid:
    """Build spanwise aerodynamic strips from the structural mesh and twist state."""
    eta_rows = mesh.y_rows / mesh.half_span_m
    twist_delta_rows = _linear(twist_root_delta_deg, twist_tip_delta_deg, eta_rows)
    total_row_twist = mesh.base_row_twist_deg + twist_delta_rows
    if aeroelastic_twist_row_deg is not None:
        total_row_twist = total_row_twist + np.asarray(aeroelastic_twist_row_deg, dtype=float)

    y_panel_mid = 0.5 * (mesh.y_rows[:-1] + mesh.y_rows[1:])
    dy_panel = mesh.y_rows[1:] - mesh.y_rows[:-1]
    chord_panel = 0.5 * (mesh.chord_rows[:-1] + mesh.chord_rows[1:])
    x_le_panel = 0.5 * (mesh.x_le_rows[:-1] + mesh.x_le_rows[1:])
    twist_panel = 0.5 * (total_row_twist[:-1] + total_row_twist[1:])

    return SpanwiseAeroGrid(
        y_panel_mid=y_panel_mid,
        dy_panel=dy_panel,
        chord_panel=chord_panel,
        x_le_panel=x_le_panel,
        twist_panel_deg=twist_panel,
        area_half_m2=mesh.area_half_m2,
        half_span_m=mesh.half_span_m,
    )


def distribute_panel_forces_to_nodal_loads(
    mesh: WingMesh,
    panel_forces_xyz: np.ndarray,
) -> np.ndarray:
    """Distribute spanwise panel forces to nodal loads on the aero mesh."""
    panel_forces = np.asarray(panel_forces_xyz, dtype=float)
    expected_shape = (mesh.n_span, 3)
    if panel_forces.shape != expected_shape:
        raise ValueError(
            f"panel_forces_xyz shape must be {expected_shape}, got {panel_forces.shape}"
        )

    nodal_forces = np.zeros((mesh.num_nodes, 3), dtype=float)
    n_per_row = float(mesh.n_chord + 1)
    for j in range(mesh.n_span):
        row_a = mesh.node_index_grid[j, :]
        row_b = mesh.node_index_grid[j + 1, :]
        row_share = 0.5 * panel_forces[j, :] / n_per_row
        nodal_forces[row_a, :] += row_share
        nodal_forces[row_b, :] += row_share
    return nodal_forces


def aero_nodes_from_structural_mesh(mesh: WingMesh, z_offset_m: float) -> np.ndarray:
    """Offset structural nodes to define aerodynamic transfer nodes."""
    aero_nodes = mesh.nodes.copy()
    aero_nodes[:, 2] += float(z_offset_m)
    return aero_nodes


def compute_aeroelastic_row_twist_delta_deg(
    mesh: WingMesh,
    structural_displacements_xyz: np.ndarray,
) -> np.ndarray:
    """Compute incremental twist (deg) at each span row from structural displacement."""
    disp = np.asarray(structural_displacements_xyz, dtype=float)
    if disp.shape != (mesh.num_nodes, 3):
        raise ValueError(
            f"structural_displacements_xyz must have shape {(mesh.num_nodes, 3)}"
        )
    deformed_nodes = mesh.nodes + disp
    deformed_twist = _row_twist_from_nodes(deformed_nodes, mesh.node_index_grid)
    return deformed_twist - mesh.base_row_twist_deg


def write_structural_bdf(mesh: WingMesh, output_path: str | Path) -> Path:
    """Write a pyTACS-compatible BDF for the half-wing shell model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["SOL 103", "CEND", "BEGIN BULK"]

    # Nodes
    for node_id, xyz in enumerate(mesh.nodes, start=1):
        x, y, z = xyz
        lines.append(f"GRID,{node_id},,{x:.8f},{y:.8f},{z:.8f}")

    # Elements: each span strip gets its own component (PID) to expose spanwise thickness DVs.
    eid = 1
    for j in range(mesh.n_span):
        pid = j + 1
        for i in range(mesh.n_chord):
            n1 = int(mesh.node_id_grid[j, i])
            n2 = int(mesh.node_id_grid[j, i + 1])
            n3 = int(mesh.node_id_grid[j + 1, i + 1])
            n4 = int(mesh.node_id_grid[j + 1, i])
            lines.append(f"CQUAD4,{eid},{pid},{n1},{n2},{n3},{n4}")
            eid += 1

    # Cantilever root boundary condition.
    for nid in mesh.node_id_grid[0, :]:
        lines.append(f"SPC,1,{int(nid)},123456,0.0")

    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def spanwise_thickness_distribution(
    mesh: WingMesh,
    thickness_root_m: float,
    thickness_tip_m: float,
) -> np.ndarray:
    """Create spanwise shell thickness values aligned with spanwise component IDs."""
    eta = np.linspace(0.0, 1.0, mesh.n_span)
    return _linear(float(thickness_root_m), float(thickness_tip_m), eta)


def validate_tacs_node_order(struct_nodes: np.ndarray, mesh_nodes: np.ndarray) -> Tuple[bool, float]:
    """Check whether pyTACS node order matches generated BDF node order."""
    struct = np.asarray(struct_nodes, dtype=float)
    mesh = np.asarray(mesh_nodes, dtype=float)
    if struct.shape != mesh.shape:
        return False, float("inf")
    max_abs_err = float(np.max(np.abs(struct - mesh)))
    return bool(max_abs_err < 1.0e-7), max_abs_err
