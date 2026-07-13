"""
Multi-zone aircraft wingbox BDF generator.

Extends the proven single-PCOMP `build_wingbox_bdf` geometry (rectangular
box, GRID/CQUAD4 mesh) with N independent PCOMP properties along the span --
one per zone -- so ply count and thickness can genuinely vary zone to zone,
which the Build 2 smoke test's single-PCOMP fixture could not demonstrate.
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
from pyNastran.bdf.cards.constraints import SPC1


def build_multizone_wingbox_bdf(
    bdf_path: str,
    span: float,
    width: float,
    height: float,
    n_zones: int = 5,
    n_span_per_zone: int = 4,
    n_perim: int = 8,
    skin_mat_id: int = 1,
    load_set_id: int = 1,
    spc_set_id: int = 1,
    ply_angles=(0.0, 45.0, -45.0, 90.0),
    zone_ply_counts=None,  # list of {angle: count} per zone; default = uniform
    ply_thickness: float = 0.000131,
    mat_props=None,  # dict of E1,E2,nu12,G12,G1z,G2z,rho (IM7/8552 defaults)
) -> dict:
    """Build a multi-zone composite wingbox. Returns a dict with the BDF model,
    zone->element-id mapping, and zone->PCOMP-id mapping for later per-zone edits.
    """
    mat_props = mat_props or dict(E1=171e9, E2=9.08e9, nu12=0.32, G12=5.29e9, G1z=5.29e9, G2z=5.29e9, rho=1580.0)
    if zone_ply_counts is None:
        zone_ply_counts = [{a: 2 for a in ply_angles} for _ in range(n_zones)]

    bdf_path = Path(bdf_path)
    bdf_path.parent.mkdir(parents=True, exist_ok=True)

    model = BDF()
    model.sol = 101
    model.case_control_deck = CaseControlDeck(
        [
            "TITLE = Multi-Zone aircraft Wingbox",
            "SUBTITLE = Build 2 VAM-FSD logged production run",
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
    y = np.linspace(0.0, span, n_span + 1)
    n_seg = n_perim // 4
    w2, h2 = 0.5 * width, 0.5 * height
    corners = [(w2, -h2), (-w2, -h2), (-w2, h2), (w2, h2)]
    perim_points = []
    for k in range(4):
        x0, z0 = corners[k]
        x1, z1 = corners[(k + 1) % 4]
        for i in range(n_seg):
            s = i / n_seg
            perim_points.append((x0 + s * (x1 - x0), z0 + s * (z1 - z0)))
    n_perim_nodes = len(perim_points)

    node_id = {}
    nid = 1
    for j, (px, pz) in enumerate(perim_points):
        for i, yy in enumerate(y):
            node_id[(i, j)] = nid
            model.nodes[nid] = GRID.add_card(BDFCard(["GRID", nid, 0, float(px), float(yy), float(pz)]))
            nid += 1

    # Material (shared across zones -- ply *counts/thickness* vary, not the base ply material).
    # Fully SI throughout (Pa, kg/m^3, meters) -- matches the meter-scale GRID
    # coordinates used everywhere in this generator. An earlier version mixed
    # an implicit MPa/tonne/mm convention here with meter-scale geometry,
    # which produced physically nonsensical (~1e8 m) deflections downstream.
    mat_fields = ["MAT8", skin_mat_id, mat_props["E1"], mat_props["E2"], mat_props["nu12"],
                  mat_props["G12"], mat_props["G1z"], mat_props["G2z"], mat_props["rho"]]
    model.materials[skin_mat_id] = MAT8.add_card(BDFCard(mat_fields))

    # One PCOMP per zone.
    zone_pcomp_id = {}
    for z in range(n_zones):
        pid = 100 + z
        angles = []
        for angle in ply_angles:
            angles.extend([angle] * int(zone_ply_counts[z].get(angle, 0)))
        n_plies = len(angles)
        if n_plies == 0:
            angles = [0.0]
            n_plies = 1
        fields = ["PCOMP", pid, None, None, None, None, None, None, "SYM"]
        for theta in angles:
            fields += [skin_mat_id, ply_thickness, theta, "YES"]
        model.properties[pid] = PCOMP.add_card(BDFCard(fields))
        zone_pcomp_id[z] = pid

    # Elements: assign to zone based on spanwise station -> PCOMP.
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

    # Clamp root (i=0) fully.
    root_nodes = [node_id[(0, j)] for j in range(n_perim_nodes)]
    model.add_card(["SPC1", spc_set_id, 123456] + root_nodes, "SPC1")

    model.write_bdf(str(bdf_path), size=8, is_double=False)
    # pyNastran's write_bdf does not always emit ENDDATA depending on model
    # state; MYSTRAN requires it (ERROR 1011 otherwise). Append if missing.
    text = bdf_path.read_text()
    if "ENDDATA" not in text:
        with open(bdf_path, "a") as f:
            f.write("ENDDATA\n")

    return {
        "model": model,
        "zone_elements": zone_elements,
        "zone_pcomp_id": zone_pcomp_id,
        "y_stations": y,
        "n_zones": n_zones,
        "n_span_per_zone": n_span_per_zone,
        "n_perim_nodes": n_perim_nodes,
        "node_id": node_id,
    }


if __name__ == "__main__":
    info = build_multizone_wingbox_bdf(
        "wingbox_multizone.bdf", span=6.0, width=1.5, height=0.36, n_zones=5,
    )
    print(f"Zones: {info['n_zones']}, elements/zone: {len(info['zone_elements'][0])}")
    print(f"Total elements: {sum(len(v) for v in info['zone_elements'].values())}")
