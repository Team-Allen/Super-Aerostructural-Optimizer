"""
ParaView screenshot automation for PIML-MDO results.

The module can be used standalone:

    python -m piml_mdo.utils.paraview_screenshots results/<run_name>

or called from the pipeline orchestrator.  It:

1. Searches the run directory for ``*_undeformed.vtk`` and ``*_deformed.vtk``.
2. Writes an ``optimization_convergence.csv`` (and a matplotlib fallback PNG)
   from ``optimization_history.json``.
3. Generates a temporary ParaView Python script and runs it with ``pvpython``
   (or ``pvbatch``) to render 4-6 views:

   * undeformed wing
   * deformed wing
   * pressure coefficient distribution
   * failure index
   * laminate thickness
   * optimization convergence

All screenshots are saved to ``results/<run_name>/paraview_screenshots/``.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Common ParaView installation locations on Windows
_WINDOWS_PVPYTHON_PATHS = [
    r"C:\Program Files\ParaView 5.13.1\bin\pvpython.exe",
    r"C:\Program Files\ParaView 5.13\bin\pvpython.exe",
    r"C:\Program Files\ParaView\bin\pvpython.exe",
]

SCREENSHOT_NAMES = {
    "undeformed": "01_undeformed_wing.png",
    "deformed": "02_deformed_wing.png",
    "pressure": "03_pressure_distribution.png",
    "failure": "04_failure_index.png",
    "thickness": "05_laminate_thickness.png",
    "convergence": "06_optimization_convergence.png",
}


def _to_posix(path: Path) -> str:
    """Return a forward-slash path string that works in both Python and ParaView."""
    return path.as_posix()


def find_pvpython(pvpython_path: Optional[str] = None) -> Optional[str]:
    """Locate a usable ParaView Python executable.

    The search order is:
    1. User-supplied path from the config
    2. ``pvpython`` / ``pvbatch`` on PATH
    3. Common Windows ParaView installation directories
    """
    if pvpython_path:
        p = Path(pvpython_path)
        if p.exists():
            return str(p.resolve())

    for candidate in ("pvpython", "pvbatch"):
        exe = shutil.which(candidate)
        if exe:
            return exe

    for candidate in _WINDOWS_PVPYTHON_PATHS:
        p = Path(candidate)
        if p.exists():
            return str(p.resolve())

    return None


def _write_convergence_csv(run_dir: Path) -> Optional[Path]:
    """Write a CSV of the optimization history for ParaView's chart view."""
    history_path = run_dir / "optimization_history.json"
    if not history_path.exists():
        return None

    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as exc:
        logger.warning("Could not read optimization history: %s", exc)
        return None

    if not history:
        return None

    csv_path = run_dir / "optimization_convergence.csv"
    # Normalise keys
    keys = ["eval", "objective", "cl", "cd", "ld", "mass", "failure_index"]
    with open(csv_path, "w", newline="") as f:
        f.write(",".join(keys) + "\n")
        for row in history:
            values = []
            for k in keys:
                v = row.get(k, 0.0)
                try:
                    v = float(v)
                except Exception:
                    v = 0.0
                values.append(f"{v:.8e}")
            f.write(",".join(values) + "\n")

    return csv_path


def _matplotlib_convergence_plot(run_dir: Path) -> Optional[Path]:
    """Generate a fallback 2-D convergence plot with matplotlib."""
    history_path = run_dir / "optimization_history.json"
    if not history_path.exists():
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning("matplotlib not available for convergence plot: %s", exc)
        return None

    with open(history_path) as f:
        history = json.load(f)

    if not history:
        return None

    evals = [h.get("eval", i) for i, h in enumerate(history)]
    objs = [h.get("objective", np.nan) for h in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(evals, objs, "b-", linewidth=1.5)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Objective")
    ax.set_title("Optimization convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = run_dir / "paraview_screenshots" / SCREENSHOT_NAMES["convergence"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _generate_pvpython_script(
    run_dir: Path,
    undeformed_vtk: Optional[Path],
    deformed_vtk: Optional[Path],
    convergence_csv: Optional[Path],
    screenshot_dir: Path,
    width: int,
    height: int,
) -> str:
    """Build the ParaView Python script as a string."""
    undeformed_str = _to_posix(undeformed_vtk) if undeformed_vtk else ""
    deformed_str = _to_posix(deformed_vtk) if deformed_vtk else ""
    csv_str = _to_posix(convergence_csv) if convergence_csv else ""

    script = f'''# Auto-generated ParaView screenshot script
from paraview.simple import *
import sys, os

run_dir = r"{_to_posix(run_dir)}"
undeformed_path = r"{undeformed_str}"
deformed_path = r"{deformed_str}"
conv_csv = r"{csv_str}"
screenshot_dir = r"{_to_posix(screenshot_dir)}"
width = {width}
height = {height}

os.makedirs(screenshot_dir, exist_ok=True)

def _bounds_center(reader):
    info = reader.GetDataInformation()
    b = info.GetBounds()
    return [(b[0]+b[1])*0.5, (b[2]+b[3])*0.5, (b[4]+b[5])*0.5]

def _view_size(reader):
    info = reader.GetDataInformation()
    b = info.GetBounds()
    dx = b[1]-b[0]
    dy = b[3]-b[2]
    dz = b[5]-b[4]
    return max(dx, dy, dz)

def screenshot_3d(reader, scalar, out_file, view=None):
    if reader is None or not os.path.isfile(reader.FileNames[0]):
        print("SKIP:", out_file)
        return None
    if view is None:
        view = CreateView('RenderView')
    else:
        SetActiveView(view)

    display = Show(reader, view)
    display.Representation = 'Surface'
    display.Ambient = 0.4
    display.Diffuse = 0.6

    pdi = reader.GetPointDataInformation()
    available_scalars = [a.GetName() for a in pdi]
    if scalar and scalar in available_scalars:
        ColorBy(display, ('POINTS', scalar))
        display.RescaleTransferFunctionToDataRange(True, False)
        ctf = GetColorTransferFunction(scalar)
        sb = GetScalarBar(ctf, view)
        sb.Title = scalar.replace('_', ' ').title()
        sb.Visibility = 1

    view.ResetCamera()
    center = _bounds_center(reader)
    size = _view_size(reader)
    dist = max(size * 2.5, 1.0)
    view.CameraFocalPoint = center
    view.CameraPosition = [center[0] - dist*0.7, center[1] - dist*0.6, center[2] + dist*0.5]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.Update()

    SaveScreenshot(out_file, view,
                   ImageResolution=[width, height],
                   OverrideColorPalette='WhiteBackground',
                   TransparentBackground=0)
    print("SAVED:", out_file)
    Hide(reader, view)
    return view

def screenshot_convergence(csv_path, out_file):
    if not os.path.isfile(csv_path):
        print("SKIP convergence:", out_file)
        return
    try:
        csv = CSVReader(FileName=[csv_path])
        chart = CreateView('XYChartView')
        plot = Show(csv, chart)
        plot.AttributeType = 'Row Data'
        plot.XArrayName = 'eval'
        plot.SeriesVisibility = ['objective', '1']
        chart.LeftAxisTitle = 'Objective'
        chart.BottomAxisTitle = 'Evaluations'
        chart.ChartTitle = 'Optimization Convergence'
        chart.Update()
        SaveScreenshot(out_file, chart,
                       ImageResolution=[width, height],
                       OverrideColorPalette='WhiteBackground')
        print("SAVED:", out_file)
    except Exception as exc:
        print("SKIP convergence (chart error):", exc)

readers = {{}}
if undeformed_path:
    readers['undeformed'] = LegacyVTKReader(FileNames=[undeformed_path])
    readers['undeformed'].UpdatePipeline()
if deformed_path:
    readers['deformed'] = LegacyVTKReader(FileNames=[deformed_path])
    readers['deformed'].UpdatePipeline()

# 1. Undeformed wing
if 'undeformed' in readers:
    screenshot_3d(readers['undeformed'], 'pressure_coefficient',
                  os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["undeformed"]}'))

# 2. Deformed wing shape (no scalar coloring)
if 'deformed' in readers:
    screenshot_3d(readers['deformed'], None,
                  os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["deformed"]}'))

    # 3. Pressure coefficient
    screenshot_3d(readers['deformed'], 'pressure_coefficient',
                  os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["pressure"]}'))

    # 4. Failure index
    screenshot_3d(readers['deformed'], 'failure_index',
                  os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["failure"]}'))

    # 5. Laminate thickness
    screenshot_3d(readers['deformed'], 'laminate_thickness',
                  os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["thickness"]}'))

# 6. Optimization convergence chart (if CSV available)
if conv_csv:
    screenshot_convergence(conv_csv,
                           os.path.join(screenshot_dir, '{SCREENSHOT_NAMES["convergence"]}'))

print("ParaView screenshot generation complete")
'''
    return script


def _find_vtk_files(run_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Locate undeformed and deformed VTK files in the run directory."""
    undeformed = None
    deformed = None

    candidates = sorted(run_dir.glob("*.vtk"))
    for c in candidates:
        name = c.name.lower()
        if "undeformed" in name and undeformed is None:
            undeformed = c
        elif "deformed" in name and deformed is None:
            deformed = c

    # Fallback: take first two .vtk files if naming convention differs
    if undeformed is None and candidates:
        undeformed = candidates[0]
    if deformed is None and len(candidates) > 1:
        deformed = candidates[1]

    return undeformed, deformed


def generate_screenshots(
    run_dir: str | Path,
    config: Optional[Any] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    pvpython_path: Optional[str] = None,
) -> list[Path]:
    """Generate ParaView screenshots for a completed run.

    Parameters
    ----------
    run_dir : str or Path
        Directory containing the VTK files and ``optimization_history.json``.
    config : object, optional
        Pipeline configuration object.  If provided, ``vtk_export``,
        ``paraview_screenshots``, ``screenshot_width``, ``screenshot_height`` and
        ``pvpython_path`` are read from it.
    width, height : int, optional
        Screenshot resolution in pixels.
    pvpython_path : str, optional
        Path to the ParaView Python executable.

    Returns
    -------
    list[Path]
        Paths to all generated PNG files.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Run directory does not exist: {run_dir}")

    # Read configuration values
    if config is not None:
        width = width or getattr(config, "screenshot_width", 1920)
        height = height or getattr(config, "screenshot_height", 1080)
        pvpython_path = pvpython_path or getattr(config, "pvpython_path", None)

    width = width or 1920
    height = height or 1080

    screenshot_dir = run_dir / "paraview_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Convergence data for the chart view + fallback matplotlib plot
    convergence_csv = _write_convergence_csv(run_dir)
    convergence_png = _matplotlib_convergence_plot(run_dir)

    # Locate VTK datasets
    undeformed_vtk, deformed_vtk = _find_vtk_files(run_dir)
    if undeformed_vtk is None and deformed_vtk is None:
        logger.warning("No VTK files found in %s; skipping ParaView screenshots", run_dir)
        return [convergence_png] if convergence_png else []

    pvpython = find_pvpython(pvpython_path)
    if pvpython is None:
        logger.warning(
            "pvpython/pvbatch not found. Screenshots skipped. "
            "Set pvpython_path in the config or add ParaView to PATH."
        )
        return [convergence_png] if convergence_png else []

    # Write temporary ParaView script
    script = _generate_pvpython_script(
        run_dir, undeformed_vtk, deformed_vtk, convergence_csv, screenshot_dir, width, height
    )
    script_path = run_dir / ".paraview_screenshot_tmp.py"
    with open(script_path, "w") as f:
        f.write(script)

    logger.info("Running ParaView screenshot script: %s", pvpython)
    cmd = [pvpython, "--force-offscreen-rendering", str(script_path)]
    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.error("ParaView screenshot generation failed:\n%s", exc.stderr)
        # Keep the matplotlib fallback if it exists
        return [convergence_png] if convergence_png else []
    except FileNotFoundError:
        logger.error("ParaView executable not found: %s", pvpython)
        return [convergence_png] if convergence_png else []

    generated: list[Path] = []
    for name in SCREENSHOT_NAMES.values():
        p = screenshot_dir / name
        if p.exists():
            generated.append(p)

    # Remove temporary script unless we are debugging
    try:
        script_path.unlink()
    except OSError:
        pass

    logger.info("Generated %d screenshot(s) in %s", len(generated), screenshot_dir)
    return generated


def main() -> int:
    """CLI entry point for standalone usage."""
    if len(sys.argv) < 2:
        print("Usage: python -m piml_mdo.utils.paraview_screenshots <run_dir>")
        return 1

    run_dir = Path(sys.argv[1])
    try:
        paths = generate_screenshots(run_dir)
        print("Screenshots:")
        for p in paths:
            print("  ", p)
        return 0
    except Exception as exc:
        logger.error("Screenshot generation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
