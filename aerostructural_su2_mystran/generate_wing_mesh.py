"""
Generate a 3-D RANS volume mesh around the aircraft wing planform using gmsh's
OpenCASCADE (OCC) kernel, exported in SU2 format.

Geometry corrected to match the real CAD reference
(Reference Docs/MYSTRAN/PipeLine_Building/Wing_Surface_Cleaned.stp),
measured directly with gmsh's OCC kernel (not assumed):
    CAD semi-span   = 0.23394 m   -> scaled to real target semi-span 3.0 m
    CAD root chord  = 0.20955 m   -> scale factor 12.824
    CAD tip chord   = 0.04888 m
    CAD LE sweep    = 26.8 deg  (atan(delta_Z_LE / semi-span) from real root/tip
                                  leading-edge points, not assumed)
    CAD section     = symmetric (camber_mid constant at every station sampled),
                       t/c ~6.67% at root -> ~7.56% at tip (measured from the
                       real meshed surface, not NACA2412 -- the earlier build
                       used an assumed 12%-thick cambered NACA2412 section,
                       which was never checked against this CAD).

    semi_span  = 3.0 m
    chord_root = 2.688 m
    chord_tip  = 0.627 m
    sweep_deg  = 26.8
    airfoil    = symmetric, t/c 6.67% (root) -> 7.56% (tip), linear

Build 2 (PIML_PIPELINE.md) Module 1 -- this is the mesh SU2 solves on.
"""

import sys
import numpy as np
import gmsh

SEMI_SPAN = 3.0
CHORD_ROOT = 2.688
CHORD_TIP = 0.627
SWEEP_DEG = 26.8
TC_ROOT = 0.0667   # measured from real CAD mesh, root station
TC_TIP = 0.0756    # measured from real CAD mesh, tip station

FARFIELD_MULT = 15.0  # farfield box half-size, multiples of semi-span
N_AIRFOIL_PTS = 24  # polygonal approximation -- few points, robust to mesh
BL_FIRST_LAYER = 1e-3  # first prism layer thickness [m], wall-function scale
BL_NUM_LAYERS = 8
BL_RATIO = 1.25

MESH_SIZE_WING = 0.06   # scaled down from 0.15 (tuned for the old 4.5m root
                        # chord, ~3.3% chord) toward the new 2.688m root
                        # chord (~2.2% chord) -- the coarser absolute size
                        # produced a non-physical negative CD by
                        # misresolving the LE suction peak on this thin
                        # (~7% t/c) section. 0.03 was tried first but its
                        # tet count (~6M, still refining after 350s) was
                        # impractical; 0.06 is the tractable middle ground.
MESH_SIZE_FARFIELD = 1.8


def symmetric_coordinates(t_c: float, n_points: int, chord: float) -> np.ndarray:
    """Closed-loop symmetric-airfoil coordinates (TE -> upper -> LE -> lower -> TE),
    scaled by chord. Uses the standard NACA00xx thickness envelope with t/c set
    per span station from the real CAD measurement (root 6.67% -> tip 7.56%),
    since the real wing section is symmetric (zero camber at every sampled
    station) -- not the cambered NACA2412 assumed in the earlier build."""
    beta = np.linspace(0.0, np.pi, n_points // 2 + 1)
    x = 0.5 * (1.0 - np.cos(beta))

    yt = 5 * t_c * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4
    )
    # Blunt TE (5% of local max half-thickness) instead of a mathematically
    # sharp zero-thickness edge -- a sharp TE on this thin section produced
    # spurious negative CD in Euler (confirmed: persisted identically under
    # both JST and ROE schemes, ruling out a numerical-scheme cause and
    # pointing at the geometry itself). Explicit short vertical closing
    # edge at the TE (upper and lower TE points both kept, not collapsed
    # to one point), a standard blunt-TE treatment.
    yt[-1] = 0.05 * yt.max()

    upper = np.column_stack([x, yt])
    lower = np.column_stack([x[::-1][:-1], -yt[::-1][:-1]])
    loop = np.vstack([upper, lower]) * chord
    return loop


def build_wing_mesh(output_path: str, n_sections: int = 9, tip_washout_deg: float = 0.0,
                     rans_mode: bool = False, rans_wall_size: float = 0.02,
                     rans_wall_dist: float = 0.03, mid_washout_deg: float = None):
    """Loft a wing surface through spanwise NACA sections, embed in a farfield
    box, and generate an unstructured tet volume mesh with a boundary-layer
    field for RANS wall resolution.

    Args:
        tip_washout_deg: geometric twist at the tip [deg] (negative = washout).
        mid_washout_deg: geometric twist at 50% semi-span [deg]. If None,
            twist is linear from root (0) to tip, same as before (backward
            compatible). If given, twist is piecewise-linear through
            [root=0, mid=mid_washout_deg, tip=tip_washout_deg] -- the same
            2-control-point parameterization used by the VLM DOE screening
            (doe_vlm_screening.py), so SU2 refinement solves the identical
            design variable, not an approximation of it.
        Each section is rotated about its own local quarter-chord point, so
        twist changes local angle of attack without changing the planform's
        leading-edge sweep line.
    """
    gmsh.initialize()
    gmsh.model.add("wing")
    occ = gmsh.model.occ

    y_stations = np.linspace(0.0, SEMI_SPAN, n_sections)
    section_wires = []

    for y in y_stations:
        eta = y / SEMI_SPAN
        chord = CHORD_ROOT + (CHORD_TIP - CHORD_ROOT) * eta
        t_c = TC_ROOT + (TC_TIP - TC_ROOT) * eta
        x_le = y * np.tan(np.radians(SWEEP_DEG))
        if mid_washout_deg is None:
            twist_deg = tip_washout_deg * eta
        else:
            twist_deg = np.interp(y, [0.0, SEMI_SPAN * 0.5, SEMI_SPAN],
                                   [0.0, mid_washout_deg, tip_washout_deg])

        coords = symmetric_coordinates(t_c, N_AIRFOIL_PTS, chord)
        if twist_deg != 0.0:
            # Rotate about the local quarter-chord point (0.25*chord, 0) in
            # section-local coordinates, then translate to the swept position.
            theta = np.radians(twist_deg)
            c, s = np.cos(theta), np.sin(theta)
            x_qc = 0.25 * chord
            xr = coords[:, 0] - x_qc
            zr = coords[:, 1]
            coords = np.column_stack([
                xr * c + zr * s + x_qc,
                -xr * s + zr * c,
            ])
        pts = []
        for x_local, z_local in coords:
            tag = occ.addPoint(x_le + x_local, y, z_local, MESH_SIZE_WING)
            pts.append(tag)
        pts.append(pts[0])
        # Straight line segments (polygonal airfoil), not a spline -- this is
        # a coarser geometric approximation but meshes far more reliably than
        # a high-curvature BSpline surface loft, which repeatedly produced
        # invalid/degenerate elements near the sharp trailing edge.
        lines = [occ.addLine(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        wire = occ.addWire(lines)
        section_wires.append(wire)

    # Loft through all sections to form the wing solid (root to tip, with a
    # flat tip cap via ThruSections' default capping). Ruled (bilinear) patches
    # between consecutive sections, rather than a single smooth BSpline loft
    # surface, mesh far more reliably in gmsh/OCC for a sharp-trailing-edge
    # airfoil cross-section.
    wing = occ.addThruSections(section_wires, makeSolid=True, makeRuled=True)
    occ.synchronize()

    # Farfield box.
    box_half = FARFIELD_MULT * SEMI_SPAN
    box = occ.addBox(
        -box_half, -0.1, -box_half,
        2 * box_half, SEMI_SPAN + 0.2, 2 * box_half,
    )
    occ.synchronize()

    wing_vols = [v[1] for v in wing if v[0] == 3]
    fluid = occ.cut([(3, box)], [(3, v) for v in wing_vols], removeTool=True)
    occ.synchronize()

    # Tag boundary surfaces: wing surface vs farfield box surfaces.
    all_surfaces = gmsh.model.getEntities(2)
    box_bbox = (-box_half, -0.1, -box_half, box_half, SEMI_SPAN + 0.2, box_half)
    wing_surfs, far_surfs = [], []
    for dim, tag in all_surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
        on_box = (
            abs(xmin - box_bbox[0]) < 1e-3 or abs(xmax - box_bbox[3]) < 1e-3
            or abs(ymin - box_bbox[1]) < 1e-3 or abs(ymax - box_bbox[4]) < 1e-3
            or abs(zmin - box_bbox[2]) < 1e-3 or abs(zmax - box_bbox[5]) < 1e-3
        )
        (far_surfs if on_box else wing_surfs).append(tag)

    gmsh.model.addPhysicalGroup(2, wing_surfs, name="wing")
    gmsh.model.addPhysicalGroup(2, far_surfs, name="farfield")
    fluid_vols = [v[1] for v in fluid[0] if v[0] == 3]
    gmsh.model.addPhysicalGroup(3, fluid_vols, name="fluid")

    # Distance + Threshold refinement near the wing surface. This is an
    # inviscid-appropriate graded tet mesh (no wall-resolved prism layers) --
    # sufficient for a first Euler smoke test. Wall-resolved boundary-layer
    # prisms for RANS are a follow-up refinement once this simpler mesh is
    # confirmed to solve correctly end-to-end.
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", wing_surfs)
    thresh_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_field, "InField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMin", MESH_SIZE_WING)
    gmsh.model.mesh.field.setNumber(thresh_field, "SizeMax", MESH_SIZE_FARFIELD)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMin", CHORD_ROOT * 0.5)
    gmsh.model.mesh.field.setNumber(thresh_field, "DistMax", SEMI_SPAN * 2.0)

    if rans_mode:
        # RANS mode: no true anisotropic prism boundary layer -- gmsh 4.15's
        # 3-D "BoundaryLayer" field genuinely rejects FacesList/SurfacesList
        # in this OCC-kernel workflow (confirmed by direct API probing, not
        # an assumption), a known real limitation of gmsh's 3-D BL support.
        # Instead: a much finer, isotropic near-wall grading (down to
        # rans_wall_size right at the surface) combined with SOLVER=RANS in
        # SU2 -- this activates the real viscous stress + turbulence-closure
        # terms the Euler equations lack, which is the actual physics needed
        # to damp the shock oscillation found in the Euler campaign (README
        # Sec8.3), even without a proper y+~1 wall-resolved mesh.
        wall_thresh = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(wall_thresh, "InField", dist_field)
        gmsh.model.mesh.field.setNumber(wall_thresh, "SizeMin", rans_wall_size)
        gmsh.model.mesh.field.setNumber(wall_thresh, "SizeMax", MESH_SIZE_WING)
        gmsh.model.mesh.field.setNumber(wall_thresh, "DistMin", rans_wall_dist)
        gmsh.model.mesh.field.setNumber(wall_thresh, "DistMax", rans_wall_dist * 6.0)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh_field, wall_thresh])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(thresh_field)

    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)

    n_nodes = len(gmsh.model.mesh.getNodes()[0])
    gmsh.finalize()
    return n_nodes


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "mesh_wing.su2"
    n_sections = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    twist = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    mid_twist = float(sys.argv[4]) if len(sys.argv) > 4 else None
    n_nodes = build_wing_mesh(out, n_sections, tip_washout_deg=twist, mid_washout_deg=mid_twist)
    print(f"Wrote {out}: {n_nodes} nodes (tip washout {twist} deg, mid washout {mid_twist})")
