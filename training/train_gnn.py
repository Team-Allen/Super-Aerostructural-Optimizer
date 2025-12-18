"""
Training Script for Airfoil GNN

Trains a Graph Neural Network on the AirFRANS dataset to predict
flow fields around airfoils. Once trained, this model can predict
CFD results 1000× faster than running actual simulations.

Usage:
    python train_gnn.py --epochs 100 --batch_size 8 --model_type attention

Features:
- Automatic GPU detection and usage
- Training progress tracking with loss plots
- Model checkpointing (saves best model)
- Validation metrics
- Resume training from checkpoint
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ml_models.airfrans_dataloader import create_dataloaders, AirFRANSDataset
from ml_models.gnn_model import create_model


class GNNTrainer:
    """
    Trainer class for Airfoil GNN.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "checkpoints"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function (MSE for regression)
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epoch = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # Compute loss
            loss = self.criterion(predictions, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]")
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Compute loss
                loss = self.criterion(predictions, batch.y)
                
                epoch_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"✅ Saved best model (val_loss: {self.best_val_loss:.6f})")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int, resume: bool = False):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume: Whether to resume from latest checkpoint
        """
        if resume and (self.checkpoint_dir / 'latest.pth').exists():
            self.load_checkpoint(self.checkpoint_dir / 'latest.pth')
        
        start_epoch = self.epoch
        
        print(f"\n{'='*60}")
        print(f"Starting training from epoch {start_epoch} to {num_epochs}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
            
            # Plot progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_curves()
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        # Final plot
        self.plot_training_curves(save=True)
    
    def plot_training_curves(self, save=False):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        
        epochs = range(len(self.train_losses))
        
        plt.plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('GNN Training Progress', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        if save:
            plt.savefig(self.checkpoint_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {self.checkpoint_dir / 'training_curves.png'}")
        else:
            plt.savefig(self.checkpoint_dir / 'training_curves_temp.png', dpi=100)
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Airfoil GNN on AirFRANS dataset')
    
    # Data arguments
    parser.add_argument('--data_path', type=str,
                        default=r'F:\MDO LAB\RESEARCH\Airfrans Airfoil Data\archive',
                        help='Path to AirFRANS dataset')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='attention',
                        choices=['basic', 'attention', 'advanced'],
                        help='GNN architecture type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    
    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='training/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from latest checkpoint')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create dataloaders
    print("\n" + "="*60)
    print("Loading AirFRANS dataset...")
    print("="*60)
    
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=0,  # Windows compatibility
        shuffle_train=True
    )
    
    # Get dataset statistics for model initialization
    print("\nComputing dataset statistics...")
    train_dataset = AirFRANSDataset(args.data_path, train=True)
    stats = train_dataset.get_statistics()
    
    # Create model
    print("\n" + "="*60)
    print("Creating GNN model...")
    print("="*60)
    
    model = create_model(
        num_node_features=stats['num_features'],
        num_outputs=stats['num_outputs'],
        model_type=args.model_type,
        device=device
    )
    
    # Save configuration
    config = {
        'model_type': args.model_type,
        'num_features': stats['num_features'],
        'num_outputs': stats['num_outputs'],
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'device': device,
        'timestamp': datetime.now().isoformat(),
    }
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create trainer
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train(num_epochs=args.epochs, resume=args.resume)
    
    print("\n✅ Training complete! Best model saved to:", checkpoint_dir / 'best.pth')


if __name__ == "__main__":
    main()
