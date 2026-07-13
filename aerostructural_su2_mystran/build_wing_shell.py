"""
Real aircraft wing skin structural shell mesh -- NACA2412 airfoil cross-section
(same geometry as the SU2 CFD mesh), lofted along the real 6 m semi-span with
real taper (4.5m root -> 1.5m tip chord) and 35-degree sweep. Multiple
spanwise zones, each with an independent PCOMP composite layup.

This replaces two earlier stand-ins, honestly:
  1. The rectangular-box wingbox approximation (build_multizone_wingbox.py)
     -- real airfoil shape now, not an artificial box cross-section.
  2. The user's Composite_Wing.txt reference file -- real CAD-derived mesh,
     but at the wrong physical scale (~0.8m x 0.6m panel, not the 12m-span
     aircraft wing) and isotropic (MAT1), not a true composite layup. Not used
     as the structural model here for that reason.

The mesh is a thin-walled closed-perimeter shell (upper + lower skin as one
continuous CQUAD4 loop per span station) -- a standard semi-monocoque
idealization of a wing skin structure, carrying bending/torsion via the
closed section, same physical idea as a real wingbox/wing-skin structure.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from pyNastran.bdf.bdf import BDF
from pyNastran.bdf.case_control_deck import CaseControlDeck
from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
from pyNastran.bdf.cards.nodes import GRID
from pyNastran.bdf.cards.elements.shell import CQUAD4
from pyNastran.bdf.cards.properties.shell import PCOMP
from pyNastran.bdf.cards.materials import MAT8

SEMI_SPAN = 6.0
CHORD_ROOT = 4.5
CHORD_TIP = 1.5
SWEEP_DEG = 35.0
AIRFOIL = "2412"
N_AIRFOIL_PTS = 20  # polygonal (not spline) -- matches the lesson learned
                     # building the SU2 volume mesh: fewer points, more robust.


def naca4_coordinates(code: str, n_points: int, chord: float) -> np.ndarray:
    """Closed-loop NACA 4-digit coordinates (TE -> upper -> LE -> lower -> TE),
    scaled by chord. Identical formulation to generate_wing_mesh.py, so
    the structural shell and the SU2 aero mesh share the same real airfoil
    shape."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0

    beta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5 * t * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    yt[-1] = 0.0

    if m > 0:
        yc = np.where(x < p, (m / p**2) * (2 * p * x - x**2),
                      (m / (1 - p) ** 2) * ((1 - 2 * p) + 2 * p * x - x**2))
        dyc = np.where(x < p, (2 * m / p**2) * (p - x),
                       (2 * m / (1 - p) ** 2) * (p - x))
    else:
        yc = np.zeros_like(x)
        dyc = np.zeros_like(x)
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    upper = np.column_stack([xu, yu])
    lower = np.column_stack([xl[::-1][1:-1], yl[::-1][1:-1]])
    loop = np.vstack([upper, lower]) * chord
    return loop


def build_wing_shell_bdf(
    bdf_path: str,
    n_zones: int = 8,
    n_span_per_zone: int = 4,
    skin_mat_id: int = 1,
    load_set_id: int = 1,
    spc_set_id: int = 1,
    ply_angles=(0.0, 45.0, -45.0, 90.0),
    zone_ply_counts=None,
    ply_thickness: float = 0.000131,
    mat_props=None,
) -> dict:
    mat_props = mat_props or dict(E1=171e9, E2=9.08e9, nu12=0.32, G12=5.29e9, G1z=5.29e9, G2z=5.29e9, rho=1580.0)
    if zone_ply_counts is None:
        zone_ply_counts = [{a: 2 for a in ply_angles} for _ in range(n_zones)]

    bdf_path = Path(bdf_path)
    bdf_path.parent.mkdir(parents=True, exist_ok=True)

    model = BDF()
    model.sol = 101
    model.case_control_deck = CaseControlDeck(
        [
            "TITLE = Real aircraft Wing Skin (NACA2412, real taper/sweep)",
            "SUBTITLE = Build 2 aerostructural production run",
            f"SPC = {spc_set_id}",
            f"LOAD = {load_set_id}",
            "DISPLACEMENT(PRINT,PUNCH) = ALL",
            "STRESS(PRINT,PUNCH,CORNER) = ALL",
            "STRAIN(PRINT,PUNCH,CORNER) = ALL",
            "BEGIN BULK",
        ],
        log=model.log,
    )

    n_span = n_zones * n_span_per_zone
    y_stations = np.linspace(0.0, SEMI_SPAN, n_span + 1)

    node_id = {}
    nid = 1
    n_perim_nodes = None
    for i, y in enumerate(y_stations):
        eta = y / SEMI_SPAN
        chord = CHORD_ROOT + (CHORD_TIP - CHORD_ROOT) * eta
        x_le = y * np.tan(np.radians(SWEEP_DEG))
        coords = naca4_coordinates(AIRFOIL, N_AIRFOIL_PTS, chord)
        if n_perim_nodes is None:
            n_perim_nodes = len(coords)
        for j, (x_local, z_local) in enumerate(coords):
            node_id[(i, j)] = nid
            model.nodes[nid] = GRID.add_card(
                BDFCard(["GRID", nid, 0, float(x_le + x_local), float(y), float(z_local)])
            )
            nid += 1

    mat_fields = ["MAT8", skin_mat_id, mat_props["E1"], mat_props["E2"], mat_props["nu12"],
                  mat_props["G12"], mat_props["G1z"], mat_props["G2z"], mat_props["rho"]]
    model.materials[skin_mat_id] = MAT8.add_card(BDFCard(mat_fields))

    zone_pcomp_id = {}
    for z in range(n_zones):
        pid = 100 + z
        angles = []
        for angle in ply_angles:
            angles.extend([angle] * int(zone_ply_counts[z].get(angle, 0)))
        if not angles:
            angles = [0.0]
        fields = ["PCOMP", pid, None, None, None, None, None, None, "SYM"]
        for theta in angles:
            fields += [skin_mat_id, ply_thickness, theta, "YES"]
        model.properties[pid] = PCOMP.add_card(BDFCard(fields))
        zone_pcomp_id[z] = pid

    eid = 1
    zone_elements = {z: [] for z in range(n_zones)}
    for i in range(n_span):
        zone = min(i // n_span_per_zone, n_zones - 1)
        pid = zone_pcomp_id[zone]
        for j in range(n_perim_nodes):
            j2 = (j + 1) % n_perim_nodes
            n1 = node_id[(i, j)]
            n2 = node_id[(i, j2)]
            n3 = node_id[(i + 1, j2)]
            n4 = node_id[(i + 1, j)]
            fields = ["CQUAD4", eid, pid, n1, n2, n3, n4]
            model.elements[eid] = CQUAD4.add_card(BDFCard(fields))
            zone_elements[zone].append(eid)
            eid += 1

    root_nodes = [node_id[(0, j)] for j in range(n_perim_nodes)]
    model.add_card(["SPC1", spc_set_id, 123456] + root_nodes, "SPC1")

    model.write_bdf(str(bdf_path), size=8, is_double=False)
    text = bdf_path.read_text()
    if "ENDDATA" not in text:
        with open(bdf_path, "a") as f:
            f.write("ENDDATA\n")

    return {
        "model": model,
        "zone_elements": zone_elements,
        "zone_pcomp_id": zone_pcomp_id,
        "y_stations": y_stations,
        "n_zones": n_zones,
        "n_span_per_zone": n_span_per_zone,
        "n_perim_nodes": n_perim_nodes,
        "node_id": node_id,
        "n_elements": eid - 1,
        "n_nodes": nid - 1,
    }


if __name__ == "__main__":
    info = build_wing_shell_bdf("wingshell.bdf", n_zones=8)
    print(f"Real aircraft wing shell: {info['n_nodes']} nodes, {info['n_elements']} elements, "
          f"{info['n_zones']} zones, {info['n_perim_nodes']} perimeter nodes/station")
