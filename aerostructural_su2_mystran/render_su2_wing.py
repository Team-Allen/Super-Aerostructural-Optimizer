"""
Render the real SU2 aircraft wing solve (flow_wing.vtu) in ParaView: pressure
coefficient contours on the wing surface, plus a Mach-number cut plane.

Run with pvpython:
    "C:/Program Files/ParaView 5.13.1/bin/pvpython.exe" render_su2_wing.py
"""

from paraview.simple import *
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "renders")
os.makedirs(OUT_DIR, exist_ok=True)

reader = OpenDataFile(os.path.join(os.path.dirname(__file__), "flow_wing.vtu"))
reader.UpdatePipeline()

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1600, 1000]
view.Background = [1.0, 1.0, 1.0]
view.OrientationAxesVisibility = 0

# The farfield box is 15x the wing semi-span (90 m vs 6 m) -- extracting the
# raw external surface shows mostly farfield wall, with the wing a tiny
# sliver in the middle. Clip to a tight box around just the wing first.
clip = Clip(Input=reader)
clip.ClipType = "Box"
clip.ClipType.Position = [-0.5, -0.2, -0.35]
clip.ClipType.Length = [6.5, 6.4, 0.7]
clip.Invert = 0
clip.UpdatePipeline()

# --- 1. Extract the wing surface (external boundary) and color by Cp ---
surface = ExtractSurface(Input=clip)
surface.UpdatePipeline()

disp = Show(surface, view)
ColorBy(disp, ("POINTS", "Pressure_Coefficient"))
disp.RescaleTransferFunctionToDataRange(True)
cp_lut = GetColorTransferFunction("Pressure_Coefficient")
cp_lut.ApplyPreset("Cool to Warm", True)
disp.SetScalarBarVisibility(view, True)

view.ResetCamera()
cam = GetActiveCamera()
cam.Azimuth(35)
cam.Elevation(20)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "01_su2_wing_pressure_coefficient.png"), view)
print("Saved 01_su2_wing_pressure_coefficient.png")

# --- 2. Top-down view of the wing pressure field ---
cam.SetPosition(3.0, 3.0, 40.0)
cam.SetFocalPoint(3.0, 3.0, 0.0)
cam.SetViewUp(0, 1, 0)
view.ResetCamera()
Render()
SaveScreenshot(os.path.join(OUT_DIR, "02_su2_wing_pressure_topdown.png"), view)
print("Saved 02_su2_wing_pressure_topdown.png")

# --- 3. Volume slice colored by Mach number, through mid-span ---
Hide(surface, view)
slc = Slice(Input=reader)
slc.SliceType = "Plane"
slc.SliceType.Origin = [3.0, 3.0, 0.0]
slc.SliceType.Normal = [0.0, 1.0, 0.0]
slc.UpdatePipeline()

slice_disp = Show(slc, view)
ColorBy(slice_disp, ("POINTS", "Mach"))
slice_disp.RescaleTransferFunctionToDataRange(True)
mach_lut = GetColorTransferFunction("Mach")
mach_lut.ApplyPreset("Cool to Warm", True)
slice_disp.SetScalarBarVisibility(view, True)

cam.SetPosition(3.0, -12.0, 0.0)
cam.SetFocalPoint(3.0, 3.0, 0.0)
cam.SetViewUp(0, 0, 1)
cam.SetParallelProjection(1)
cam.SetParallelScale(6.0)
Render()
SaveScreenshot(os.path.join(OUT_DIR, "03_su2_wing_mach_slice.png"), view)
print("Saved 03_su2_wing_mach_slice.png")

print("Done.")
