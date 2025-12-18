"""Simple Data container compatible with expected attributes used in the repo.
This is a minimal stand-in for torch_geometric.data.Data.
"""
class Data(dict):
    def __init__(self, x=None, pos=None, edge_index=None, **kwargs):
        super().__init__()
        self.x = x
        self.pos = pos
        self.edge_index = edge_index
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"Data(x={None if self.x is None else tuple(self.x.shape)}, pos={None if self.pos is None else tuple(self.pos.shape)}, edge_index={None if self.edge_index is None else tuple(self.edge_index.shape)})"
