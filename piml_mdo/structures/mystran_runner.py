"""
MYSTRAN high-fidelity structural runner using pyNastran.

Provides:
- MystranRunner: locate the local MYSTRAN executable, read/write BDF decks with
  pyNastran, execute the solver, and parse displacements/stresses from the F06.
- build_wingbox_bdf: generate a simple parametric composite wing-box BDF for
  offline structural DOEs.
- helpers to edit PCOMP laminates and PLOAD2 / FORCE loads in an existing BDF.

The runner is intentionally decoupled from the MDO loop so that MYSTRAN can be
used offline to generate truth data for the structural PINN/surrogate.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def find_mystran_exe(root_search: Optional[Path | str] = None) -> Path:
    """Locate a MYSTRAN executable under the project tree or PATH.

    Searches, in order:
    1. ``MYSTRAN_EXE`` environment variable.
    2. ``root_search`` directory tree for ``MYSTRAN.exe`` or ``mystran.exe``.
    3. Directories on ``PATH`` for ``MYSTRAN.exe`` / ``mystran.exe``.

    Returns:
        Absolute path to the executable.
    """
    env = os.environ.get("MYSTRAN_EXE")
    if env and Path(env).is_file():
        return Path(env).resolve()

    candidates = ["MYSTRAN.exe", "mystran.exe", "MYSTRAN", "mystran"]

    search_roots = []
    if root_search is not None:
        search_roots.append(Path(root_search))
    # Default fallback: project root and the common "Reference Docs" location
    search_roots.append(Path.cwd())
    search_roots.append(Path.cwd() / "Reference Docs")

    for root in search_roots:
        if root.is_dir():
            for cand in candidates:
                found = list(root.rglob(cand))
                if found:
                    return found[0].resolve()

    for cand in candidates:
        exe = shutil.which(cand)
        if exe:
            return Path(exe).resolve()

    raise FileNotFoundError(
        "Could not find MYSTRAN executable. Set MYSTRAN_EXE or place it on PATH."
    )


def _grid_coord(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Return a GRID coordinate tuple (Nastran basic coordinates)."""
    return float(x), float(y), float(z)


def _fmt(value, width: int = 8) -> str:
    """Format a value as a Nastran small-field entry.

    Integers are written as-is, floats use a compact scientific notation that
    fits in ``width`` characters (8 by default), and strings are passed through.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return str(value)
    # Try fixed or scientific notation and pick the shortest representation
    # that still fits in the requested width.
    for fmt in ("{:.6G}", "{:.5G}", "{:.4G}", "{:.3G}", "{:.2G}", "{:.1G}"):
        s = fmt.format(float(value))
        if len(s) <= width:
            return s
    # Fallback: exponent only
    s = f"{float(value):.{max(0, width - 6)}E}"
    return s[:width]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class MystranResult:
    """Container for parsed MYSTRAN static analysis results."""

    max_displacement: float = 0.0
    max_vertical_displacement: float = 0.0
    max_twist_angle: float = 0.0
    max_rotation_angle: float = 0.0
    max_von_mises: float = 0.0
    max_failure_index: float = 0.0
    total_mass: float = 0.0
    exit_code: int = -1
    log_tail: str = ""

    def to_dict(self) -> dict:
        return {
            "max_displacement": float(self.max_displacement),
            "max_vertical_displacement": float(self.max_vertical_displacement),
            "max_twist_angle": float(self.max_twist_angle),
            "max_rotation_angle": float(self.max_rotation_angle),
            "max_von_mises": float(self.max_von_mises),
            "max_failure_index": float(self.max_failure_index),
            "total_mass": float(self.total_mass),
            "exit_code": int(self.exit_code),
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MystranRunner:
    """Wrap MYSTRAN execution and result parsing for a single BDF model."""

    def __init__(
        self,
        bdf_path: Path | str,
        workdir: Path | str,
        mystran_exe: Optional[Path | str] = None,
        nastran_template: Optional[Path | str] = None,
    ):
        """
        Args:
            bdf_path: Path to the input BDF file (will be written if it does
                not exist and ``nastran_template`` is omitted).
            workdir: Directory where MYSTRAN writes F06/OP2/etc.
            mystran_exe: Path to MYSTRAN executable. If ``None``, searched
                automatically.
            nastran_template: Optional existing BDF to use as a template.
        """
        self.bdf_path = Path(bdf_path).resolve()
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        if mystran_exe is None:
            self.mystran_exe = find_mystran_exe(self.bdf_path.parent)
        else:
            self.mystran_exe = Path(mystran_exe).resolve()

        self.template_path = None
        if nastran_template is not None:
            self.template_path = Path(nastran_template).resolve()

        self._bdf_model: Optional["BDF"] = None  # lazy import / cache

    # ------------------------------------------------------------------
    # BDF access
    # ------------------------------------------------------------------

    def _get_bdf(self) -> "BDF":
        """Lazy-load pyNastran BDF model."""
        if self._bdf_model is None:
            from pyNastran.bdf.bdf import BDF

            model = BDF()
            source = self.template_path or self.bdf_path
            if source.exists():
                model.read_bdf(str(source))
            self._bdf_model = model
        return self._bdf_model

    def write_bdf(self, path: Optional[Path | str] = None) -> Path:
        """Write the current BDF model to disk."""
        model = self._get_bdf()
        out = Path(path) if path is not None else self.bdf_path
        out.parent.mkdir(parents=True, exist_ok=True)
        model.write_bdf(str(out), size=8, is_double=False)
        # pyNastran only re-emits ENDDATA if the in-memory model was
        # originally read from a file that had one; a purely programmatically
        # built model (no read_bdf call) has no such marker, and MYSTRAN
        # fails with "ERROR 1011: NO ENDDATA ENTRY FOUND" without it.
        text = out.read_text(errors="ignore")
        if "ENDDATA" not in text:
            with open(out, "a") as f:
                f.write("ENDDATA\n")
        return out

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------

    def set_pcomp_layup(
        self,
        pid: int,
        angles: Sequence[float],
        thicknesses: Sequence[float],
        mids: Optional[Sequence[int]] = None,
        lam: str = "SYM",
    ):
        """Replace a PCOMP property with a new stacking sequence.

        Args:
            pid: Property ID of the PCOMP to replace.
            angles: Ply angles in degrees (full stack if ``lam`` is blank).
            thicknesses: Ply thicknesses (one per ply).
            mids: Material IDs per ply. Defaults to all plies using the first
                MAT8 found in the model.
            lam: Nastran laminate option, e.g. ``"SYM"`` or ``""``.
        """
        from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
        from pyNastran.bdf.cards.properties.shell import PCOMP

        model = self._get_bdf()
        if pid not in model.properties:
            raise KeyError(f"PCOMP property {pid} not found in BDF")

        if mids is None:
            mat8_ids = [mid for mid, m in model.materials.items() if m.type == "MAT8"]
            if not mat8_ids:
                raise ValueError("No MAT8 material found; supply mids explicitly")
            mid0 = mat8_ids[0]
            mids = [mid0] * len(angles)

        # Build pyNastran PCOMP fields.  Fields 3-7 are NSM, SB, FT, TREF,
        # GE, then LAM (field 8, index 8), followed by alternating MID/T/THETA/SOUT.
        fields = ["PCOMP", pid, 0.0, "", "", "", "", "", lam]
        for mid, t, theta in zip(mids, thicknesses, angles):
            fields.extend([mid, float(t), float(theta), ""])

        model.properties.pop(pid, None)
        pcomp = PCOMP.add_card(BDFCard(fields))
        model.properties[pid] = pcomp
        logger.debug(f"Set PCOMP {pid}: {len(angles)} plies")

    def set_laminate_from_counts(
        self,
        pid: int,
        ply_counts: dict[float, int],
        ply_thickness: float,
        material_id: Optional[int] = None,
    ):
        """Build a symmetric PCOMP from angle-count pairs.

        Args:
            pid: Property ID.
            ply_counts: Mapping angle(deg) -> number of plies.
            ply_thickness: Single-ply thickness.
            material_id: MAT8 ID. If ``None``, first MAT8 in model is used.
        """
        angles = []
        for angle in sorted(ply_counts):
            angles.extend([angle] * int(ply_counts[angle]))
        thicknesses = [float(ply_thickness)] * len(angles)
        self.set_pcomp_layup(pid, angles, thicknesses, mids=None, lam="SYM")

    def set_pressure_load(
        self,
        load_id: int,
        pressure: float,
        element_ids: Optional[Sequence[int]] = None,
    ):
        """Set the magnitude of an existing PLOAD2 card or create one.

        Args:
            load_id: LOAD set ID.
            pressure: New pressure value (negative = suction/upward).
            element_ids: Optional element IDs for a new PLOAD2. If omitted and
                no PLOAD2 exists, the load is applied to all CQUAD4 elements.
        """
        from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
        from pyNastran.bdf.cards.loads.static_loads import PLOAD2

        model = self._get_bdf()
        if load_id not in model.loads:
            model.loads[load_id] = []

        pload_found = False
        for load in model.loads[load_id]:
            if load.type == "PLOAD2":
                load.pressure = float(pressure)
                pload_found = True
            elif load.type == "PLOAD4":
                load.pressures = [float(pressure)] * len(load.pressures)
                pload_found = True

        if not pload_found:
            if element_ids is None:
                element_ids = [eid for eid, e in model.elements.items() if e.type in ("CQUAD4", "CTRIA3")]
            fields = ["PLOAD2", load_id, float(pressure)] + list(element_ids)
            model.loads[load_id].append(PLOAD2.add_card(BDFCard(fields)))

    def set_pressure_field(
        self,
        load_id: int,
        element_pressures: dict[int, float],
    ):
        """Write a real, spatially-varying per-element pressure field as one
        PLOAD4 card per element -- replacing the flat, uniform placeholder
        used by ``set_pressure_load``.

        This is the Build 2 FSI-transfer entry point: ``element_pressures``
        is the output of the SU2 -> structural-mesh spatial transfer
        (``mdo-tools/su2_validation/fsi_transfer.py``), keyed by CQUAD4
        element ID with the gauge pressure at that element's location.

        Args:
            load_id: LOAD set ID.
            element_pressures: {element_id: pressure [Pa]}. Positive pressure
                is a compressive (inward) load on the element face, matching
                PLOAD4's sign convention.
        """
        from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
        from pyNastran.bdf.cards.loads.static_loads import PLOAD4

        model = self._get_bdf()
        if load_id not in model.loads:
            model.loads[load_id] = []

        # Drop any existing PLOAD2/PLOAD4 cards in this set -- the field
        # replaces them rather than layering on top.
        model.loads[load_id] = [
            ld for ld in model.loads[load_id] if ld.type not in ("PLOAD2", "PLOAD4")
        ]

        for eid, pressure in element_pressures.items():
            fields = ["PLOAD4", load_id, int(eid), float(pressure)]
            model.loads[load_id].append(PLOAD4.add_card(BDFCard(fields)))

    def set_force(
        self,
        load_id: int,
        node: int,
        vector: Sequence[float],
    ):
        """Add or replace a concentrated FORCE card in a LOAD set.

        Args:
            load_id: LOAD set ID.
            node: Node ID where force is applied.
            vector: (Fx, Fy, Fz) force components.
        """
        from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
        from pyNastran.bdf.cards.loads.static_loads import FORCE

        model = self._get_bdf()
        # Remove any existing FORCE on this node in this load set.
        existing = model.loads.get(load_id, [])
        cleaned = [ld for ld in existing if not (ld.type == "FORCE" and ld.node == node)]
        model.loads[load_id] = cleaned

        fields = ["FORCE", load_id, node, 0, float(vector[0]), float(vector[1]), float(vector[2])]
        f = FORCE.add_card(BDFCard(fields))
        model.loads[load_id].append(f)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        write_model: bool = True,
        cleanup: bool = False,
        timeout: Optional[int] = 600,
    ) -> MystranResult:
        """Run MYSTRAN and parse the F06.

        Args:
            write_model: Write the current in-memory BDF before running.
            cleanup: Delete previous MYSTRAN output files before running.
            timeout: Maximum wall time in seconds for the solver.

        Returns:
            ``MystranResult`` with parsed displacements, stresses, and mass.
        """
        if write_model:
            self.write_bdf(self.bdf_path)

        if cleanup:
            for ext in [".F04", ".F06", ".OP2", ".PCH", ".ANS", ".BUG", ".ERR", ".L1B", ".L1O", ".L1Q"]:
                p = self.workdir / (self.bdf_path.stem + ext)
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        log_path = self.workdir / f"{self.bdf_path.stem}_mystran.log"
        cmd = [str(self.mystran_exe), str(self.bdf_path.name)]

        logger.info(f"Running MYSTRAN: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error(f"MYSTRAN timed out after {timeout}s")
            # Try to capture partial log if any
            log_text = ""
            log_file = self.workdir / (self.bdf_path.stem + ".F06")
            if log_file.exists():
                try:
                    log_text = log_file.read_text(errors="ignore")[-4000:]
                except Exception:
                    pass
            return MystranResult(exit_code=-1, log_tail=log_text)

        log_path.write_text(proc.stdout, errors="ignore")
        f06_path = self.workdir / (self.bdf_path.stem + ".F06")
        result = self.parse_f06(f06_path) if f06_path.exists() else MystranResult()
        result.exit_code = proc.returncode
        result.log_tail = proc.stdout[-2000:] if proc.stdout else ""

        if proc.returncode != 0:
            logger.warning(f"MYSTRAN exit code {proc.returncode}; check {log_path}")
        else:
            logger.info(
                f"MYSTRAN finished: max_disp={result.max_vertical_displacement:.4e}, "
                f"max_vm={result.max_von_mises:.4e}, FI={result.max_failure_index:.4f}"
            )
        return result

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_f06(self, f06_path: Path | str) -> MystranResult:
        """Parse displacement and composite-stress sections from an F06 file."""
        f06_path = Path(f06_path)
        text = f06_path.read_text(errors="ignore")

        result = MystranResult()
        result.max_displacement = self._parse_max_displacement(text)
        result.max_vertical_displacement = self._parse_max_vertical_displacement(text)
        result.max_twist_angle = self._parse_max_rotation(text, col=5)
        result.max_rotation_angle = self._parse_max_rotation(text, col=3)
        result.max_von_mises = self._parse_max_von_mises(text)
        result.max_failure_index = self._parse_max_failure_index(text)
        result.total_mass = self._compute_mass_from_bdf()
        return result

    @staticmethod
    def _parse_floats(line: str) -> list[float]:
        """Extract all floating-point numbers from a line."""
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]

    def _parse_displacement_block(self, text: str) -> np.ndarray:
        """Parse the displacement block into an Nx6 array [T1,T2,T3,R1,R2,R3]."""
        pattern = re.compile(
            r"D\s+I\s+S\s+P\s+L\s+A\s+C\s+E\s+M\s+E\s+N\s+T\s+S.*?\n.*?GRID.*?\n.*?\n(.*?)\n\s*\n",
            re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            return np.empty((0, 6))

        rows = []
        for line in match.group(1).splitlines():
            nums = self._parse_floats(line)
            if len(nums) >= 7:
                # format: GRID COORD T1 T2 T3 R1 R2 R3
                rows.append(nums[1:7])
        return np.asarray(rows, dtype=float)

    def _parse_max_displacement(self, text: str) -> float:
        disp = self._parse_displacement_block(text)
        if disp.size == 0:
            return 0.0
        mag = np.linalg.norm(disp[:, :3], axis=1)
        return float(np.max(mag))

    def _parse_max_vertical_displacement(self, text: str) -> float:
        disp = self._parse_displacement_block(text)
        if disp.size == 0:
            return 0.0
        return float(np.max(np.abs(disp[:, 2])))

    def _parse_max_rotation(self, text: str, col: int) -> float:
        disp = self._parse_displacement_block(text)
        if disp.size == 0:
            return 0.0
        return float(np.max(np.abs(disp[:, col])))

    def _parse_composite_stress_block(self, text: str) -> np.ndarray:
        """Parse layered composite stress block. Returns array of
        [element_id, ply, sigma_1, sigma_2, tau_12, tau_13, tau_23, angle, major, minor, von_mises].
        """
        pattern = re.compile(
            r"S\s+T\s+R\s+E\s+S\s+S\s+E\s+S\s+I\s+N\s+L\s+A\s+Y\s+E\s+R\s+E\s+D.*?"
            r"Element\s+Ply.*?von\s*\n.*?\n(.*?)\n\s*(?:MAXIMUM|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            return np.empty((0, 11))

        rows = []
        current_eid = None
        for line in match.group(1).splitlines():
            line = line.rstrip()
            if not line.strip():
                continue
            nums = self._parse_floats(line)
            if len(nums) == 11:
                current_eid = int(nums[0])
                rows.append(nums)
            elif len(nums) == 10 and current_eid is not None:
                rows.append([float(current_eid)] + nums)
        return np.asarray(rows, dtype=float)

    def _parse_max_von_mises(self, text: str) -> float:
        data = self._parse_composite_stress_block(text)
        if data.size == 0:
            return 0.0
        return float(np.max(data[:, -1]))

    def _parse_max_failure_index(self, text: str) -> float:
        """Compute Tsai-Wu failure index from parsed ply stresses and MAT8."""
        data = self._parse_composite_stress_block(text)
        if data.size == 0:
            return 0.0

        model = self._get_bdf()
        mat8 = None
        for m in model.materials.values():
            if m.type == "MAT8":
                mat8 = m
                break
        if mat8 is None:
            return 0.0

        Xt = float(getattr(mat8, "Xt", 1e12) or 1e12)
        Xc = float(getattr(mat8, "Xc", 1e12) or 1e12)
        Yt = float(getattr(mat8, "Yt", 1e12) or 1e12)
        Yc = float(getattr(mat8, "Yc", 1e12) or 1e12)
        S = float(getattr(mat8, "S", 1e12) or 1e12)

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F66 = 1.0 / (S * S)
        F12 = -0.5 * np.sqrt(F11 * F22)

        s1 = data[:, 2]
        s2 = data[:, 3]
        t12 = data[:, 4]
        fi = F1 * s1 + F2 * s2 + F11 * s1**2 + F22 * s2**2 + F66 * t12**2 + 2.0 * F12 * s1 * s2
        return float(np.max(fi))

    def parse_element_failure_indices(self, f06_path: Path | str) -> dict[int, float]:
        """Per-element Tsai-Wu failure index (max over plies at that element).

        This is the Build 2 VAM-FSD entry point: unlike ``max_failure_index``
        (a single global scalar), this gives the per-element resolution the
        stress-ratio resizing formula needs to size each zone independently.
        Reuses the exact Tsai-Wu formula already validated in
        ``_parse_max_failure_index``, just grouped by element ID instead of
        reduced to a single max.
        """
        text = Path(f06_path).read_text(errors="ignore")
        data = self._parse_composite_stress_block(text)
        if data.size == 0:
            return {}

        model = self._get_bdf()
        mat8 = None
        for m in model.materials.values():
            if m.type == "MAT8":
                mat8 = m
                break
        if mat8 is None:
            return {}

        Xt = float(getattr(mat8, "Xt", 1e12) or 1e12)
        Xc = float(getattr(mat8, "Xc", 1e12) or 1e12)
        Yt = float(getattr(mat8, "Yt", 1e12) or 1e12)
        Yc = float(getattr(mat8, "Yc", 1e12) or 1e12)
        S = float(getattr(mat8, "S", 1e12) or 1e12)

        F1 = 1.0 / Xt - 1.0 / Xc
        F2 = 1.0 / Yt - 1.0 / Yc
        F11 = 1.0 / (Xt * Xc)
        F22 = 1.0 / (Yt * Yc)
        F66 = 1.0 / (S * S)
        F12 = -0.5 * np.sqrt(F11 * F22)

        eids = data[:, 0].astype(int)
        s1 = data[:, 2]
        s2 = data[:, 3]
        t12 = data[:, 4]
        fi = F1 * s1 + F2 * s2 + F11 * s1**2 + F22 * s2**2 + F66 * t12**2 + 2.0 * F12 * s1 * s2

        per_element: dict[int, float] = {}
        for eid, val in zip(eids, fi):
            per_element[eid] = max(per_element.get(eid, -np.inf), float(val))
        return per_element

    def _compute_mass_from_bdf(self) -> float:
        """Estimate total structural mass from BDF geometry and laminates."""
        try:
            model = self._get_bdf()
        except Exception:
            return 0.0

        total_mass = 0.0
        for elem in model.elements.values():
            if elem.type not in ("CQUAD4", "CTRIA3"):
                continue
            pid = elem.Pid()
            prop = model.properties.get(pid)
            if prop is None:
                continue

            nodes = [model.nodes[nid] for nid in elem.node_ids]
            coords = np.array([n.xyz for n in nodes])
            area = self._polygon_area(coords)

            if prop.type == "PCOMP":
                # Sum ply thicknesses * densities. prop.plies is a list of
                # [mid, thickness, theta, sout] entries (already expanded if SYM).
                rho_avg = 0.0
                t_total = 0.0
                for (mid, t, _theta, _sout) in prop.plies:
                    mat = model.materials.get(mid)
                    rho = float(getattr(mat, "rho", 0.0) or 0.0)
                    rho_avg += rho * t
                    t_total += t
                if t_total > 0:
                    total_mass += area * rho_avg
            elif prop.type == "PSHELL":
                mid = prop.mid1
                mat = model.materials.get(mid)
                rho = float(getattr(mat, "rho", 0.0) or 0.0)
                t = float(prop.t) if prop.t is not None else 0.0
                total_mass += area * rho * t
        return float(total_mass)

    @staticmethod
    def _polygon_area(coords: np.ndarray) -> float:
        """Area of a 3-D polygon (triangle or quad) via cross product."""
        if len(coords) == 3:
            return 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        # Quad: split into two triangles
        a1 = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        a2 = 0.5 * np.linalg.norm(np.cross(coords[2] - coords[0], coords[3] - coords[0]))
        return a1 + a2


# ---------------------------------------------------------------------------
# Parametric wing-box BDF generator
# ---------------------------------------------------------------------------


def build_wingbox_bdf(
    bdf_path: Path | str,
    span: float,
    width: float,
    height: float,
    n_span: int = 12,
    n_perim: int = 4,
    skin_mat_id: int = 1,
    skin_pcomp_id: int = 1,
    load_set_id: int = 1,
    spc_set_id: int = 1,
    pressure: Optional[float] = None,
    tip_force: Optional[Sequence[float]] = None,
    tip_moment: Optional[Sequence[float]] = None,
    tip_node: Optional[int] = None,
    units_scale: float = 1.0,
) -> Path:
    """Generate a simple straight composite wing-box BDF for MYSTRAN.

    The mesh is a single-cell rectangular box of length ``span`` with width
    ``width`` (chordwise) and height ``height`` (vertical). The box is clamped
    at the root (y=0) and may carry a pressure load on the upper skin, a tip
    force, and/or a tip moment.

    Args:
        bdf_path: Output BDF path.
        span: Beam length (m).
        width: Box width (m).
        height: Box height (m).
        n_span: Number of elements along the span.
        n_perim: Number of elements around the perimeter (must be multiple of 4
            for a rectangular box; set to 4 minimum).
        skin_mat_id: MAT8 ID for the composite material.
        skin_pcomp_id: PCOMP property ID.
        load_set_id: LOAD set ID.
        spc_set_id: SPC set ID.
        pressure: Optional uniform pressure on the upper skin (negative = up).
        tip_force: Optional (Fx, Fy, Fz) concentrated force at the tip.
        tip_moment: Optional (Mx, My, Mz) concentrated moment at the tip.
        tip_node: Node ID where concentrated loads are applied. Defaults to
            the tip mid-node of the upper skin.
        units_scale: Factor applied to all coordinates to match MYSTRAN input
            units. MYSTRAN input decks often use mm; use 1000.0 to convert m to
            mm. Loads/stresses should be consistent with the chosen unit system.

    Returns:
        Path to the written BDF.
    """
    from pyNastran.bdf.bdf import BDF
    from pyNastran.bdf.case_control_deck import CaseControlDeck
    from pyNastran.bdf.bdf_interface.bdf_card import BDFCard
    from pyNastran.bdf.cards.nodes import GRID
    from pyNastran.bdf.cards.elements.shell import CQUAD4
    from pyNastran.bdf.cards.properties.shell import PCOMP
    from pyNastran.bdf.cards.materials import MAT8
    from pyNastran.bdf.cards.loads.static_loads import LOAD, PLOAD2, FORCE, MOMENT
    from pyNastran.bdf.cards.constraints import SPC1

    bdf_path = Path(bdf_path)
    bdf_path.parent.mkdir(parents=True, exist_ok=True)

    if n_perim < 4:
        raise ValueError("n_perim must be at least 4")
    if n_perim % 4 != 0:
        raise ValueError("n_perim must be a multiple of 4 for a rectangular box")

    model = BDF()
    model.sol = 101
    model.case_control_deck = CaseControlDeck(
        [
            "TITLE = Parametric Composite Wing Box",
            "SUBTITLE = VAM/MYSTRAN Structural DOE",
            f"SPC = {spc_set_id}",
            f"LOAD = {load_set_id}",
            "DISPLACEMENT(PRINT,PUNCH) = ALL",
            "STRESS(PRINT,PUNCH,CORNER) = ALL",
            "STRAIN(PRINT,PUNCH,CORNER) = ALL",
            "BEGIN BULK",
        ],
        log=model.log,
    )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    y = np.linspace(0.0, span, n_span + 1) * units_scale
    # Perimeter divisions around the rectangular box.
    # Order: bottom-right, bottom-left, top-left, top-right.
    n_seg = n_perim // 4
    w2 = 0.5 * width * units_scale
    h2 = 0.5 * height * units_scale
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
            model.nodes[nid] = GRID.add_card(
                BDFCard(["GRID", nid, 0, float(px), float(yy), float(pz)])
            )
            nid += 1

    # Elements: CQUAD4 around the perimeter, spanning from station i to i+1
    eid = 1
    upper_elements = []
    for i in range(n_span):
        for j in range(n_perim_nodes):
            jnext = (j + 1) % n_perim_nodes
            n1 = node_id[(i, j)]
            n2 = node_id[(i + 1, j)]
            n3 = node_id[(i + 1, jnext)]
            n4 = node_id[(i, jnext)]
            model.elements[eid] = CQUAD4.add_card(
                BDFCard(["CQUAD4", eid, skin_pcomp_id, n1, n2, n3, n4, 0.0])
            )
            # Identify top-skin elements (middle of top edge division)
            top_j_start = 2 * n_seg
            top_j_end = 3 * n_seg
            if top_j_start <= j < top_j_end:
                upper_elements.append(eid)
            eid += 1

    # ------------------------------------------------------------------
    # Material placeholder (will be overwritten by set_laminate_from_counts)
    # ------------------------------------------------------------------
    model.materials[skin_mat_id] = MAT8.add_card(
        BDFCard(["MAT8", skin_mat_id, 140e3, 9e3, 0.3, 5e3, 5e3, 5e3, 1.6e-9])
    )

    # Placeholder PCOMP: 8 plies, 0.5 mm each, 0 deg (valid geometry)
    pcomp_card = BDFCard(["PCOMP", skin_pcomp_id, 0.0, "", "", "", "", "", "SYM", skin_mat_id, 0.5, 0.0])
    model.properties[skin_pcomp_id] = PCOMP.add_card(pcomp_card)

    # ------------------------------------------------------------------
    # Constraints: clamp root nodes (y=0)
    # ------------------------------------------------------------------
    root_nodes = [node_id[(0, j)] for j in range(n_perim_nodes)]
    spc_fields = ["SPC1", spc_set_id, 123456] + root_nodes
    model.spcs[spc_set_id] = [SPC1.add_card(BDFCard(spc_fields))]

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------
    # Use a LOAD combination card so that FORCE/MOMENT and PLOAD2 loads can
    # be mixed safely under a single case-control LOAD set.
    pload_sid = load_set_id * 10 + 1
    force_sid = load_set_id * 10 + 2
    moment_sid = load_set_id * 10 + 3

    load_entries = []
    load_scales = []

    if pressure is not None and upper_elements:
        pload_fields = ["PLOAD2", pload_sid, float(pressure)] + upper_elements
        model.loads.setdefault(pload_sid, []).append(PLOAD2.add_card(BDFCard(pload_fields)))
        load_entries.append(pload_sid)
        load_scales.append(1.0)

    if tip_force is not None or tip_moment is not None:
        if tip_node is None:
            # tip mid-node of top skin
            top_mid_j = top_j_start + n_seg // 2
            tip_node = node_id[(n_span, top_mid_j % n_perim_nodes)]
        if tip_force is not None:
            fx, fy, fz = tip_force
            force_fields = ["FORCE", force_sid, tip_node, 0, float(fx), float(fy), float(fz)]
            model.loads.setdefault(force_sid, []).append(FORCE.add_card(BDFCard(force_fields)))
            load_entries.append(force_sid)
            load_scales.append(1.0)
        if tip_moment is not None:
            mx, my, mz = tip_moment
            moment_fields = ["MOMENT", moment_sid, tip_node, 0, float(mx), float(my), float(mz)]
            model.loads.setdefault(moment_sid, []).append(MOMENT.add_card(BDFCard(moment_fields)))
            load_entries.append(moment_sid)
            load_scales.append(1.0)

    if load_entries:
        # LOAD card: SID, overall scale, then (scale, load set) pairs
        load_fields = ["LOAD", load_set_id, 1.0]
        for scale, sid in zip(load_scales, load_entries):
            load_fields.extend([scale, sid])
        model.loads.setdefault(load_set_id, []).append(LOAD.add_card(BDFCard(load_fields)))

    model.write_bdf(str(bdf_path), size=8, is_double=False, enddata=True)
    return bdf_path


# ---------------------------------------------------------------------------
# Convenience: run from a template with a few edits
# ---------------------------------------------------------------------------


def run_mystran_on_template(
    template_bdf: Path | str,
    workdir: Path | str,
    mystran_exe: Optional[Path | str] = None,
    pressure: Optional[float] = None,
    pcomp_pid: Optional[int] = None,
    angles: Optional[Sequence[float]] = None,
    thicknesses: Optional[Sequence[float]] = None,
) -> MystranResult:
    """One-shot helper: copy a template BDF, edit load/laminate, run MYSTRAN."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    bdf_out = workdir / Path(template_bdf).name

    runner = MystranRunner(
        bdf_path=bdf_out,
        workdir=workdir,
        mystran_exe=mystran_exe,
        nastran_template=template_bdf,
    )
    if pcomp_pid is not None and angles is not None and thicknesses is not None:
        runner.set_pcomp_layup(pcomp_pid, angles, thicknesses)
    if pressure is not None:
        runner.set_pressure_load(1, pressure)
    return runner.run(cleanup=True)
