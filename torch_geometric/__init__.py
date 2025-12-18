"""Minimal local stub of torch_geometric to allow dry-run testing.
This provides very small, non-optimized implementations of the
interfaces used by the project's GNN code so we can run inference
with random weights without installing the full PyG stack.
"""
from .data import Data
from . import nn

__all__ = ["Data", "nn"]
