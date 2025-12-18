"""Minimal implementations of a few torch_geometric.nn primitives used by the repo.

These implementations purposely ignore graph connectivity and simply apply
linear transforms so the model can execute for dry-run testing. They are
NOT substitutes for real message-passing layers and are only for testing.
"""
import torch
import torch.nn as nn


def global_mean_pool(x, batch=None):
    # simple global mean over nodes
    return x.mean(dim=0, keepdim=True)


class MessagePassing:
    def __init__(self):
        pass


class _SimpleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index=None):
        # ignore edge_index, apply linear layer
        return self.lin(x)


class GCNConv(_SimpleConv):
    pass


class GATConv(_SimpleConv):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, dropout=0.0, add_self_loops=True):
        # emulate output size when concat=True (heads concat)
        out = out_channels * (heads if concat else 1)
        super().__init__(in_channels, out)


class SAGEConv(_SimpleConv):
    pass
