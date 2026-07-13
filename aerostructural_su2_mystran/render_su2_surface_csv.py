"""
Render the SU2 wing-surface pressure field directly from the surface CSV
(guaranteed wing-only points, no farfield-box contamination -- the .vtu
volume clip approach kept picking up the artificial clip-box faces).

Computes real static pressure from conservative variables (same ideal-gas
relation as fsi_transfer.py) and renders as a dense colored point cloud.
"""

import os
import numpy as np
from paraview.simple import *

HERE = os.path.dirname(__file__)
OUT_DIR = os.path.join(HERE, "renders")
os.makedirs(OUT_DIR, exist_ok=True)

GAMMA = 1.4
data = np.genfromtxt(os.path.join(HERE, "surface_flow_wing.csv"), delimiter=",", names=True)
x, y, z = data["x"], data["y"], data["z"]
rho, mx, my, mz, e = data["Density"], data["Momentum_x"], data["Momentum_y"], data["Momentum_z"], data["Energy"]
kinetic = 0.5 * (mx**2 + my**2 + mz**2) / rho
pressure = (GAMMA - 1.0) * (e - kinetic)
freestream_p = 26436.3
cp = (pressure - freestream_p) / (0.5 * 0.4127 * 254.546**2)

out_csv = os.path.join(HERE, "wing_surface_cp.csv")
with open(out_csv, "w") as f:
    f.write("x,y,z,Cp\n")
    for xi, yi, zi, cpi in zip(x, y, z, cp):
        f.write(f"{xi},{yi},{zi},{cpi}\n")

reader = CSVReader(FileName=[out_csv])
points = TableToPoints(Input=reader)
points.XColumn, points.YColumn, points.ZColumn = "x", "y", "z"
points.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1600, 1000]
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

disp = Show(points, view)
disp.SetRepresentationType("Point Gaussian")
disp.GaussianRadius = 0.06
disp.ShaderPreset = "Plain circle"
ColorBy(disp, ("POINTS", "Cp"))
disp.RescaleTransferFunctionToDataRange(True)
cp_lut = GetColorTransferFunction("Cp")
cp_lut.ApplyPreset("Cool to Warm", True)
cp_lut.InvertTransferFunction()  # convention: negative Cp (suction/upper surface) = warm
disp.SetScalarBarVisibility(view, True)
sb = GetScalarBar(cp_lut, view)
sb.Title = "Pressure Coefficient (Cp)"

cam = GetActiveCamera()
cam.SetPosition(3.0, -20.0, 12.0)
cam.SetFocalPoint(3.0, 3.0, 0.0)
cam.SetViewUp(0, 0, 1)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "04_su2_wing_surface_cp.png"), view)
print("Saved 04_su2_wing_surface_cp.png")

cam.SetPosition(3.0, 3.0, 25.0)
cam.SetFocalPoint(3.0, 3.0, 0.0)
cam.SetViewUp(0, 1, 0)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "05_su2_wing_surface_cp_topdown.png"), view)
print("Saved 05_su2_wing_surface_cp_topdown.png")

print(f"n points: {len(x)}, Cp range: [{cp.min():.3f}, {cp.max():.3f}]")
