"""
Training script for Task Graph Classification
"""
import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from configs.config import Config
from utils.data_loader import TaskGraphDataset
from models.dagnn import DAGNN, GCNClassifier


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def main(args):
    # Load config
    config = Config.from_args(args)
    
    print("="*60)
    print("Task Graph Classification - Training")
    print("="*60)
    print(f"Config: {config}")
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(config.SEED)
    
    # Handle paths - ensure they're absolute
    observed_graphs_dir = Path(config.OBSERVED_GRAPHS_DIR).resolve()
    annotations_dir = Path(config.ANNOTATIONS_DIR).resolve()
    extension3_dir = Path(config.EXTENSION3_DIR).resolve() if config.EXTENSION3_DIR else None
    
    print(f"\nData paths:")
    print(f"  Annotations: {annotations_dir}")
    print(f"  Observed graphs: {observed_graphs_dir}")
    print(f"  Extension3: {extension3_dir if extension3_dir else 'Not provided'}")
    print(f"  Annotations exists: {annotations_dir.exists()}")
    print(f"  Observed graphs exists: {observed_graphs_dir.exists()}")
    if extension3_dir:
        print(f"  Extension3 exists: {extension3_dir.exists()}")
    
    # Create dataset
    print("\nLoading dataset...")
    dataset = TaskGraphDataset(
        task_graphs_dir=str(annotations_dir),
        observed_graphs_dir=str(observed_graphs_dir),
        extension3_dir=str(extension3_dir) if extension3_dir else None,
        use_hiero_embeddings=config.USE_EXTENSION3_EMBEDDINGS
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("ERROR: No data loaded. Please check your data paths.")
        return
    
    # Split dataset (simple train/val split for now)
    # TODO: Implement Leave-One-Out cross-validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    
    # Get input dimension from first sample
    sample = dataset[0]
    in_channels = sample.x.size(1)
    print(f"Input channels: {in_channels}")
    
    # Create model
    print(f"\nCreating {config.MODEL_TYPE.upper()} model...")
    if config.MODEL_TYPE == 'dagnn':
        model = DAGNN(
            in_channels=in_channels,
            hidden_channels=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_classes=2,
            dropout=config.DROPOUT
        )
    elif config.MODEL_TYPE == 'gcn':
        model = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            num_classes=2,
            dropout=config.DROPOUT
        )
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if config.SAVE_CHECKPOINT:
                checkpoint_path = config.RESULTS_DIR / "checkpoints" / "best_model.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': config
                }, checkpoint_path)
                print(f"  → Saved checkpoint (val_acc: {val_acc:.4f})")
    
    print("\n" + "="*60)
    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Task Graph Classifier")
    
    # Data arguments
    parser.add_argument('--observed_graphs_dir', type=str, default=None,
                       help='Path to observed graphs directory (overrides config)')
    parser.add_argument('--use_drive', action='store_true',
                       help='Use Google Drive path for observed graphs')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='dagnn',
                       choices=['dagnn', 'gcn', 'graphsage'],
                       help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)
