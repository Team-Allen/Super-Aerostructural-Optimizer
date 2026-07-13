"""
Generate a 2-D O-grid mesh around a NACA0012 airfoil and export in SU2 format.

This is the Build 2 validation case: confirms the gmsh -> SU2 mesh pipeline
works end-to-end before it is trusted for the real 3-D aircraft wing mesh.

Run with the gmsh Python API (installed in the `mdo_lab` WSL2 conda env):
    python generate_naca0012_mesh.py
"""

import numpy as np
import gmsh

CHORD = 1.0
FARFIELD_RADIUS = 20.0 * CHORD  # chords
N_AIRFOIL_POINTS = 120
MESH_SIZE_AIRFOIL = 0.01
MESH_SIZE_FARFIELD = 2.0


def naca0012_coordinates(n_points: int) -> np.ndarray:
    """Return closed-loop NACA0012 coordinates (TE -> upper -> LE -> lower -> TE)."""
    beta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(beta))  # cosine spacing, clusters at LE/TE

    t = 0.12
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )
    # Ensure exact closure at the trailing edge.
    yt[-1] = 0.0

    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1][1:-1], -yt[::-1][1:-1]])
    loop = np.vstack([upper, lower])
    return loop


def build_mesh(output_path: str):
    coords = naca0012_coordinates(N_AIRFOIL_POINTS)

    gmsh.initialize()
    gmsh.model.add("naca0012_ogrid")

    # Airfoil boundary points + spline.
    pt_tags = []
    for i, (x, y) in enumerate(coords):
        tag = gmsh.model.geo.addPoint(x, y, 0.0, MESH_SIZE_AIRFOIL)
        pt_tags.append(tag)
    pt_tags.append(pt_tags[0])  # close the loop
    airfoil_spline = gmsh.model.geo.addSpline(pt_tags)
    airfoil_loop = gmsh.model.geo.addCurveLoop([airfoil_spline])

    # Farfield circle.
    center = gmsh.model.geo.addPoint(0.5, 0.0, 0.0, MESH_SIZE_FARFIELD)
    far_pts = []
    for ang in np.linspace(0.0, 2 * np.pi, 5)[:-1]:
        x = 0.5 + FARFIELD_RADIUS * np.cos(ang)
        y = FARFIELD_RADIUS * np.sin(ang)
        far_pts.append(gmsh.model.geo.addPoint(x, y, 0.0, MESH_SIZE_FARFIELD))
    far_arcs = []
    for i in range(len(far_pts)):
        a = far_pts[i]
        b = far_pts[(i + 1) % len(far_pts)]
        far_arcs.append(gmsh.model.geo.addCircleArc(a, center, b))
    farfield_loop = gmsh.model.geo.addCurveLoop(far_arcs)

    surface = gmsh.model.geo.addPlaneSurface([farfield_loop, airfoil_loop])

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(1, [airfoil_spline], name="airfoil")
    gmsh.model.addPhysicalGroup(1, far_arcs, name="farfield")
    gmsh.model.addPhysicalGroup(2, [surface], name="fluid")

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.model.mesh.generate(2)

    gmsh.write(output_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    n_elem = sum(len(e) for e in gmsh.model.mesh.getElements()[1])
    gmsh.finalize()
    return n_nodes, n_elem


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "mesh_naca0012.su2"
    n_nodes, n_elem = build_mesh(out)
    print(f"Wrote {out}: {n_nodes} nodes, {n_elem} elements")
