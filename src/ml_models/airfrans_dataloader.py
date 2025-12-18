"""
AirFRANS Dataset Loader for GNN Training

This module loads the AirFRANS dataset containing ~1000 2D airfoil CFD simulations
with full flow field solutions. Data is stored as PyTorch Geometric graph objects.

Data Structure:
- Node features (x): Flow properties at mesh points
- Node positions (pos): (x,y) coordinates of mesh points
- Edge index: Connectivity between nodes
- Labels (y): Ground truth CFD solutions (pressure, velocity)
"""

import os
import torch
from torch_geometric.data import Data, Dataset
from pathlib import Path
import numpy as np
from typing import List, Tuple


class AirFRANSDataset(Dataset):
    """
    PyTorch Geometric dataset for AirFRANS airfoil CFD data.
    
    Dataset contains 1000 2D airfoil simulations across 10 batches.
    Each sample includes:
    - Airfoil geometry (as graph nodes)
    - Flow field mesh
    - CFD solution (pressure, velocity at each node)
    
    Args:
        root_dir: Path to AirFRANS data (e.g., 'F:/MDO LAB/RESEARCH/Airfrans Airfoil Data/archive')
        train: If True, load training split. If False, load validation split.
        transform: Optional transform to apply to each sample
    """
    
    def __init__(self, root_dir: str, train: bool = True, transform=None):
        self.root_dir = Path(root_dir)
        self.train = train
        self.transform = transform
        
        # Register Data class for safe loading
        torch.serialization.add_safe_globals([Data])
        
        # Collect all .pth files from 10 batches
        self.file_paths = self._collect_file_paths()
        
        # Split into train/validation (80/20 split)
        n_total = len(self.file_paths)
        n_train = int(0.8 * n_total)
        
        if self.train:
            self.file_paths = self.file_paths[:n_train]
        else:
            self.file_paths = self.file_paths[n_train:]
        
        print(f"Loaded {'train' if train else 'validation'} dataset: {len(self.file_paths)} samples")
    
    def _collect_file_paths(self) -> List[Path]:
        """Collect all .pth file paths from 10 batches."""
        file_paths = []
        
        for batch_num in range(1, 11):  # 10 batches
            batch_dir = self.root_dir / f"graph_airfrans_data_batch_{batch_num}"
            
            if not batch_dir.exists():
                print(f"Warning: Batch {batch_num} not found at {batch_dir}")
                continue
            
            # Get all .pth files in this batch
            pth_files = sorted(batch_dir.glob("*.pth"))
            file_paths.extend(pth_files)
        
        print(f"Found {len(file_paths)} total samples across all batches")
        return file_paths
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Load a single graph sample.
        
        Args:
            idx: Index of sample to load
            
        Returns:
            PyTorch Geometric Data object with:
            - data.x: Node features [num_nodes, num_features]
            - data.pos: Node positions [num_nodes, 2]
            - data.edge_index: Edge connectivity [2, num_edges]
            - data.y: Target labels (CFD solution) [num_nodes, num_outputs]
        """
        file_path = self.file_paths[idx]
        
        try:
            # Load the checkpoint
            checkpoint = torch.load(
                file_path,
                map_location=torch.device('cpu'),
                weights_only=False
            )
            
            # Create Data object
            data = Data(**checkpoint)
            
            # Apply optional transform
            if self.transform:
                data = self.transform(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy data object on error
            return Data(
                x=torch.zeros(1, 1),
                pos=torch.zeros(1, 2),
                edge_index=torch.zeros(2, 0, dtype=torch.long),
                y=torch.zeros(1, 1)
            )
    
    def get_statistics(self) -> dict:
        """
        Compute dataset statistics for normalization.
        
        Returns:
            Dictionary with mean and std for features and labels
        """
        print("Computing dataset statistics (this may take a few minutes)...")
        
        # Sample 100 random files for statistics
        sample_indices = np.random.choice(len(self), min(100, len(self)), replace=False)
        
        x_list = []
        y_list = []
        
        for idx in sample_indices:
            data = self[idx]
            x_list.append(data.x)
            y_list.append(data.y)
        
        # Concatenate all samples
        x_all = torch.cat(x_list, dim=0)
        y_all = torch.cat(y_list, dim=0)
        
        stats = {
            'x_mean': x_all.mean(dim=0),
            'x_std': x_all.std(dim=0) + 1e-8,  # Add epsilon to avoid division by zero
            'y_mean': y_all.mean(dim=0),
            'y_std': y_all.std(dim=0) + 1e-8,
            'num_features': x_all.shape[1],
            'num_outputs': y_all.shape[1],
        }
        
        print(f"Dataset statistics:")
        print(f"  Input features: {stats['num_features']}")
        print(f"  Output features: {stats['num_outputs']}")
        print(f"  X mean: {stats['x_mean']}")
        print(f"  X std: {stats['x_std']}")
        print(f"  Y mean: {stats['y_mean']}")
        print(f"  Y std: {stats['y_std']}")
        
        return stats


def create_dataloaders(
    root_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    shuffle_train: bool = True
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.
    
    Args:
        root_dir: Path to AirFRANS data directory
        batch_size: Number of samples per batch
        num_workers: Number of workers for data loading (0 for Windows)
        shuffle_train: Whether to shuffle training data
        
    Returns:
        (train_loader, val_loader) tuple
    """
    from torch_geometric.loader import DataLoader
    
    # Create datasets
    train_dataset = AirFRANSDataset(root_dir, train=True)
    val_dataset = AirFRANSDataset(root_dir, train=False)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test the dataloader."""
    
    # Path to AirFRANS data
    data_path = r"F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive"
    
    # Create dataset
    print("Creating dataset...")
    dataset = AirFRANSDataset(data_path, train=True)
    
    # Load first sample
    print("\nLoading first sample...")
    data = dataset[0]
    
    print(f"\nSample structure:")
    print(f"  Node features (x): {data.x.shape}")
    print(f"  Node positions (pos): {data.pos.shape}")
    print(f"  Edge index: {data.edge_index.shape}")
    print(f"  Labels (y): {data.y.shape}")
    
    # Compute statistics
    print("\n" + "="*50)
    stats = dataset.get_statistics()
    
    # Create dataloaders
    print("\n" + "="*50)
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(data_path, batch_size=4)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test iteration
    print("\nTesting batch iteration...")
    for batch in train_loader:
        print(f"Batch node features: {batch.x.shape}")
        print(f"Batch positions: {batch.pos.shape}")
        print(f"Batch labels: {batch.y.shape}")
        break  # Just test first batch
    
    print("\n✅ Dataloader test complete!")
