"""FUNtoFEM transfer wrapper for aero-structural handoff."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from mpi4py import MPI
    from funtofem import Body, Scenario, TransferSettings
except ImportError as exc:  # pragma: no cover - environment-specific import
    raise ImportError(
        "FUNtoFEM and mpi4py are required. Run this pipeline from the Linux/WSL mdo-best environment."
    ) from exc


@dataclass
class TransferState:
    """Container for current transfer arrays."""

    aero_loads_xyz: np.ndarray
    struct_loads_xyz: np.ndarray
    struct_disps_xyz: np.ndarray
    aero_disps_xyz: np.ndarray


class FuntofemMeldTransfer:
    """Minimal MELD transfer manager for serial coupled analyses."""

    def __init__(
        self,
        struct_nodes_xyz: np.ndarray,
        aero_nodes_xyz: np.ndarray,
        elastic_scheme: str = "rbf",
        npts: int = 100,
        beta: float = 0.5,
    ) -> None:
        self.comm = MPI.COMM_WORLD
        self.body = Body.aeroelastic("wing")
        self.scenario = Scenario.steady("cruise", steps=1)
        self.scenario.id = 0

        self._set_nodes(struct_nodes_xyz, aero_nodes_xyz)
        max_neighbors = max(1, min(self._num_aero_nodes - 1, self._num_struct_nodes - 1))
        npts_safe = max(1, min(int(npts), max_neighbors))
        settings = TransferSettings(
            elastic_scheme=str(elastic_scheme),
            npts=npts_safe,
            beta=float(beta),
        )
        self.body.initialize_transfer(
            self.comm,
            self.comm,
            0,
            self.comm,
            0,
            settings,
        )
        self.body.initialize_variables(self.scenario)

        self.state = TransferState(
            aero_loads_xyz=np.zeros((self._num_aero_nodes, 3), dtype=float),
            struct_loads_xyz=np.zeros((self._num_struct_nodes, 3), dtype=float),
            struct_disps_xyz=np.zeros((self._num_struct_nodes, 3), dtype=float),
            aero_disps_xyz=np.zeros((self._num_aero_nodes, 3), dtype=float),
        )

    def _set_nodes(self, struct_nodes_xyz: np.ndarray, aero_nodes_xyz: np.ndarray) -> None:
        struct = np.asarray(struct_nodes_xyz, dtype=float)
        aero = np.asarray(aero_nodes_xyz, dtype=float)
        if struct.ndim != 2 or struct.shape[1] != 3:
            raise ValueError("struct_nodes_xyz must have shape (N, 3)")
        if aero.ndim != 2 or aero.shape[1] != 3:
            raise ValueError("aero_nodes_xyz must have shape (N, 3)")

        self._num_struct_nodes = int(struct.shape[0])
        self._num_aero_nodes = int(aero.shape[0])
        struct_ids = np.arange(1, self._num_struct_nodes + 1, dtype=int)
        aero_ids = np.arange(1, self._num_aero_nodes + 1, dtype=int)
        self.body.initialize_struct_nodes(struct.reshape(-1), struct_ids)
        self.body.initialize_aero_nodes(aero.reshape(-1), aero_ids)

    def update_nodes(self, struct_nodes_xyz: np.ndarray, aero_nodes_xyz: np.ndarray) -> None:
        """Update transfer coordinates after geometry changes."""
        struct = np.asarray(struct_nodes_xyz, dtype=float)
        aero = np.asarray(aero_nodes_xyz, dtype=float)
        if struct.shape != (self._num_struct_nodes, 3):
            raise ValueError("updated struct_nodes_xyz shape mismatch")
        if aero.shape != (self._num_aero_nodes, 3):
            raise ValueError("updated aero_nodes_xyz shape mismatch")
        self.body.struct_X[:] = struct.reshape(-1)
        self.body.aero_X[:] = aero.reshape(-1)
        self.body.update_transfer()

    def transfer_loads(self, aero_loads_xyz: np.ndarray) -> np.ndarray:
        """Transfer aerodynamic nodal loads to structural nodal loads."""
        loads = np.asarray(aero_loads_xyz, dtype=float)
        if loads.shape != (self._num_aero_nodes, 3):
            raise ValueError("aero_loads_xyz shape mismatch")

        self.state.aero_loads_xyz = loads.copy()
        self.body.aero_loads[self.scenario.id][:] = loads.reshape(-1)
        self.body.transfer_loads(self.scenario)

        struct_flat = self.body.struct_loads[self.scenario.id]
        self.state.struct_loads_xyz = struct_flat.reshape((-1, 3)).copy()
        return self.state.struct_loads_xyz

    def transfer_displacements(self, struct_displacements_xyz: np.ndarray) -> np.ndarray:
        """Transfer structural displacements to aerodynamic nodes."""
        disps = np.asarray(struct_displacements_xyz, dtype=float)
        if disps.shape != (self._num_struct_nodes, 3):
            raise ValueError("struct_displacements_xyz shape mismatch")

        self.state.struct_disps_xyz = disps.copy()
        self.body.struct_disps[self.scenario.id][:] = disps.reshape(-1)
        self.body.transfer_disps(self.scenario)

        aero_flat = self.body.aero_disps[self.scenario.id]
        self.state.aero_disps_xyz = aero_flat.reshape((-1, 3)).copy()
        return self.state.aero_disps_xyz
