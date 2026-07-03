"""TACS structural model wrapper for shell-based wing analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import StructuralConfig

try:
    from mpi4py import MPI
    from tacs import constitutive, elements, functions, pyTACS
except ImportError as exc:  # pragma: no cover - environment-specific import
    raise ImportError(
        "TACS and mpi4py are required. Run this pipeline from the Linux/WSL mdo-best environment."
    ) from exc


@dataclass
class StructuralResult:
    """Outputs of one structural solve."""

    mass_kg: float
    ks_failure: float
    nodal_displacements_xyz: np.ndarray
    tip_deflection_m: float


class TacsShellModel:
    """pyTACS static model with spanwise shell-thickness design variables."""

    def __init__(self, bdf_path: str | Path, config: StructuralConfig) -> None:
        self.config = config
        self.comm = MPI.COMM_WORLD
        self.bdf_path = str(Path(bdf_path))

        options = {"printtiming": bool(config.tacs_print_timing)}
        self.fea = pyTACS(self.bdf_path, comm=self.comm, options=options)
        self.fea.initialize(self._element_callback)

        self.problem = self.fea.createStaticProblem("aerostruct")
        self.problem.addFunction("mass", functions.StructuralMass)
        self.problem.addFunction(
            "ks_failure",
            functions.KSFailure,
            ksWeight=float(self.config.ks_weight),
        )

        self.num_nodes = int(self.fea.getNumOwnedNodes())
        self.vars_per_node = int(self.fea.getVarsPerNode())
        self.local_node_ids = np.arange(self.num_nodes, dtype=int)

        # Build explicit mapping between BDF/Nastran order and TACS local order.
        bdf_info = self.fea.getBDFInfo()
        self.nastran_node_ids = np.array(sorted(int(nid) for nid in bdf_info.nodes.keys()), dtype=int)
        local_from_nastran = np.asarray(
            self.fea.getLocalNodeIDsFromGlobal(self.nastran_node_ids, nastranOrdering=True),
            dtype=int,
        )
        if np.any(local_from_nastran < 0):
            raise RuntimeError("TACS did not return local IDs for all BDF nodes.")
        self.local_from_nastran = local_from_nastran
        self.nastran_from_local = np.empty(self.num_nodes, dtype=int)
        self.nastran_from_local[self.local_from_nastran] = np.arange(self.num_nodes, dtype=int)

        struct_nodes_local = np.asarray(self.fea.getOrigNodes(), dtype=float).reshape((-1, 3))
        self.struct_nodes = struct_nodes_local[self.local_from_nastran, :]

        self.num_design_vars = int(self.problem.getNumDesignVars())
        self.design_vars = np.asarray(self.problem.getDesignVars(), dtype=float)

    def _element_callback(
        self,
        dv_num: int,
        comp_id: int,
        comp_desc: str,
        elem_descs,
        global_dvs,
        **kwargs,
    ):
        del comp_id, comp_desc, global_dvs, kwargs
        mat = constitutive.MaterialProperties(
            rho=float(self.config.density_kg_m3),
            E=float(self.config.youngs_modulus_pa),
            nu=float(self.config.poisson_ratio),
            ys=float(self.config.yield_stress_pa),
        )
        shell = constitutive.IsoShellConstitutive(
            mat,
            t=float(self.config.thickness_root_m),
            tNum=int(dv_num),
            tlb=float(self.config.thickness_min_m),
            tub=float(self.config.thickness_max_m),
        )

        transform = elements.ShellNaturalTransform()
        elem_kind = elem_descs[0] if isinstance(elem_descs, (list, tuple)) else elem_descs
        if elem_kind in ("CQUAD4", "CQUADR"):
            elem = elements.Quad4Shell(transform, shell)
        elif elem_kind in ("CTRIA3", "CTRIAR"):
            elem = elements.Tri3Shell(transform, shell)
        else:
            raise ValueError(f"Unsupported element type from BDF: {elem_kind}")
        return elem, [float(self.config.tacs_dv_scale)]

    def set_thickness_distribution(self, thickness_by_strip_m: np.ndarray) -> None:
        """Set spanwise shell thickness values into active TACS design variables."""
        raw = np.asarray(thickness_by_strip_m, dtype=float).reshape(-1)
        if raw.size < 1:
            raise ValueError("thickness_by_strip_m cannot be empty")

        if self.num_design_vars == raw.size:
            mapped = raw
        else:
            eta_src = np.linspace(0.0, 1.0, raw.size)
            eta_tgt = np.linspace(0.0, 1.0, self.num_design_vars)
            mapped = np.interp(eta_tgt, eta_src, raw)

        mapped = np.clip(
            mapped,
            float(self.config.thickness_min_m),
            float(self.config.thickness_max_m),
        )
        self.design_vars[:] = mapped
        self.problem.setDesignVars(self.design_vars)

    def solve(self, nodal_forces_xyz: np.ndarray, tip_node_indices: np.ndarray) -> StructuralResult:
        """Solve static structure under nodal loads from FUNtoFEM transfer."""
        loads_xyz = np.asarray(nodal_forces_xyz, dtype=float)
        if loads_xyz.shape != (self.num_nodes, 3):
            raise ValueError(
                f"nodal_forces_xyz must have shape {(self.num_nodes, 3)}, got {loads_xyz.shape}"
            )

        loads_tacs = np.zeros((self.num_nodes, self.vars_per_node), dtype=float)
        loads_tacs[:, :3] = loads_xyz

        self.problem.zeroLoads()
        self.problem.zeroVariables()
        self.problem.addLoadToNodes(
            self.nastran_node_ids,
            loads_tacs,
            nastranOrdering=True,
        )
        self.problem.solve()

        funcs = {}
        self.problem.evalFunctions(funcs)
        mass = self._extract_function(funcs, "mass")
        ks_failure = self._extract_function(funcs, "ks_failure")

        states_local = np.asarray(self.problem.getVariables(), dtype=float).reshape(
            (self.num_nodes, self.vars_per_node)
        )
        states_nastran = states_local[self.local_from_nastran, :]
        nodal_disps = states_nastran[:, :3]
        tip_idx = np.asarray(tip_node_indices, dtype=int).reshape(-1)
        tip_deflection = float(np.min(nodal_disps[tip_idx, 2]))

        return StructuralResult(
            mass_kg=float(mass),
            ks_failure=float(ks_failure),
            nodal_displacements_xyz=nodal_disps,
            tip_deflection_m=tip_deflection,
        )

    def write_solution(self, output_dir: str | Path, base_name: str = "aerostruct") -> None:
        """Write TACS solution files for post-processing."""
        self.problem.writeSolution(outputDir=str(Path(output_dir)), baseName=base_name)

    @staticmethod
    def _extract_function(funcs: dict, suffix: str) -> float:
        for key, value in funcs.items():
            if str(key).lower().endswith(suffix.lower()):
                return float(value)
        raise KeyError(f"Function with suffix '{suffix}' not found in TACS outputs: {list(funcs)}")
