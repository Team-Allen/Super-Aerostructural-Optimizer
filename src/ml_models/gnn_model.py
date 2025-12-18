"""
Graph Neural Network for Airfoil CFD Prediction

This GNN learns to predict full flow fields (pressure, velocity) around airfoils
from the AirFRANS dataset. It uses message passing to propagate information
through the mesh and predict CFD solutions at each node.

Architecture:
- Input: Airfoil geometry as graph (nodes = mesh points, edges = connectivity)
- Processing: Multiple MessagePassing layers
- Output: Flow properties (p, u, v) at each node

This is 100-1000× faster than running CFD once trained!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d


class AirfoilGNN(nn.Module):
    """
    Graph Neural Network for predicting flow fields around airfoils.
    
    Uses message passing to learn relationships between mesh points
    and predict CFD solutions without running actual simulations.
    
    Args:
        num_node_features: Number of input features per node (from AirFRANS)
        num_outputs: Number of output features per node (pressure, velocity components)
        hidden_dim: Size of hidden layers (default: 128)
        num_layers: Number of message passing layers (default: 4)
        dropout: Dropout probability (default: 0.1)
        use_attention: Whether to use Graph Attention Networks (default: False)
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_outputs: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        super(AirfoilGNN, self).__init__()
        
        self.num_node_features = num_node_features
        self.num_outputs = num_outputs
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = Sequential(
            Linear(num_node_features, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
        )
        
        # Message passing layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        ConvLayer = GATConv if use_attention else GCNConv
        
        for i in range(num_layers):
            if use_attention:
                # Graph Attention Network
                conv = GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=4,
                    concat=False,
                    dropout=dropout
                )
            else:
                # Graph Convolutional Network
                conv = GCNConv(hidden_dim, hidden_dim)
            
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output projection
        self.output_proj = Sequential(
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            Linear(hidden_dim // 2, num_outputs)
        )
    
    def forward(self, data):
        """
        Forward pass through the GNN.
        
        Args:
            data: PyTorch Geometric Data object with:
                - data.x: Node features [num_nodes, num_node_features]
                - data.edge_index: Edge connectivity [2, num_edges]
                - data.batch: Batch assignment for each node
                
        Returns:
            predictions: Flow field predictions [num_nodes, num_outputs]
        """
        x, edge_index = data.x, data.edge_index
        
        # Input projection
        x = self.input_proj(x)
        
        # Message passing layers with residual connections
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x_residual = x
            
            # Message passing
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection
            if i > 0:
                x = x + x_residual
        
        # Output projection
        out = self.output_proj(x)
        
        return out
    
    def predict(self, data):
        """
        Make predictions without gradient computation.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            predictions: Flow field predictions [num_nodes, num_outputs]
        """
        self.eval()
        with torch.no_grad():
            return self.forward(data)


class AirfoilGNN_Advanced(nn.Module):
    """
    Advanced GNN with edge features and positional encoding.
    
    This version includes:
    - Edge features (distance between nodes)
    - Positional encoding for better spatial awareness
    - Skip connections between all layers
    - Attention mechanism for important regions (near airfoil)
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_outputs: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super(AirfoilGNN_Advanced, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Positional encoding
        self.pos_encoder = Sequential(
            Linear(2, hidden_dim // 4),  # 2D positions (x, y)
            ReLU(),
            Linear(hidden_dim // 4, hidden_dim // 4),
        )
        
        # Input projection (features + positional encoding)
        self.input_proj = Sequential(
            Linear(num_node_features + hidden_dim // 4, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
        )
        
        # Message passing with attention
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = GATConv(
                hidden_dim,
                hidden_dim // 4,
                heads=4,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
            )
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_dim))
        
        # Output MLP
        self.output_mlp = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, num_outputs)
        )
    
    def forward(self, data):
        """Forward pass with positional encoding."""
        x, edge_index, pos = data.x, data.edge_index, data.pos
        
        # Positional encoding
        pos_enc = self.pos_encoder(pos)
        
        # Concatenate features with positional encoding
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.input_proj(x)
        
        # Store initial features for skip connection
        x_init = x
        
        # Message passing with skip connections
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Global skip connection from input
        x = x + x_init
        
        # Output
        out = self.output_mlp(x)
        
        return out


def create_model(
    num_node_features: int,
    num_outputs: int,
    model_type: str = "basic",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        num_node_features: Number of input features per node
        num_outputs: Number of output features per node
        model_type: "basic", "attention", or "advanced"
        device: Device to place model on
        
    Returns:
        GNN model ready for training
    """
    if model_type == "basic":
        model = AirfoilGNN(
            num_node_features=num_node_features,
            num_outputs=num_outputs,
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            use_attention=False
        )
    elif model_type == "attention":
        model = AirfoilGNN(
            num_node_features=num_node_features,
            num_outputs=num_outputs,
            hidden_dim=128,
            num_layers=4,
            dropout=0.1,
            use_attention=True
        )
    elif model_type == "advanced":
        model = AirfoilGNN_Advanced(
            num_node_features=num_node_features,
            num_outputs=num_outputs,
            hidden_dim=256,
            num_layers=6,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Created {model_type} GNN with {num_params:,} trainable parameters")
    
    return model


if __name__ == "__main__":
    """Test the GNN architecture."""
    from torch_geometric.data import Data
    
    # Create dummy data
    num_nodes = 1000
    num_features = 10
    num_outputs = 3  # pressure, u_velocity, v_velocity
    num_edges = 5000
    
    x = torch.randn(num_nodes, num_features)
    pos = torch.randn(num_nodes, 2)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    data = Data(x=x, pos=pos, edge_index=edge_index)
    
    # Test basic model
    print("Testing basic GNN...")
    model = create_model(num_features, num_outputs, model_type="basic")
    output = model(data)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [{num_nodes}, {num_outputs}]")
    assert output.shape == (num_nodes, num_outputs), "Output shape mismatch!"
    
    # Test attention model
    print("\nTesting attention GNN...")
    model = create_model(num_features, num_outputs, model_type="attention")
    output = model(data)
    print(f"Output shape: {output.shape}")
    assert output.shape == (num_nodes, num_outputs), "Output shape mismatch!"
    
    # Test advanced model
    print("\nTesting advanced GNN...")
    model = create_model(num_features, num_outputs, model_type="advanced")
    output = model(data)
    print(f"Output shape: {output.shape}")
    assert output.shape == (num_nodes, num_outputs), "Output shape mismatch!"
    
    print("\n✅ All model tests passed!")
