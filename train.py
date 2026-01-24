"""
Simplified Training Script for Task Graph Classification
Modified: Adds F1/AUC metrics and CSV logging without removing original structure.
"""

import os
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import numpy as np
import time
import csv

# Import sklearn metrics
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from models.dagnn import DAGNN, GCNClassifier


# ============================================================================
# 1. Dataset
# ============================================================================
class PreloadedGraphDataset(Dataset):
    """
    Load preprocessed .pt files directly without complex initialization
    """
    def __init__(self, pt_file_path, weights_only=False):
        super().__init__()
        self.data_list = torch.load(pt_file_path, weights_only=weights_only)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


# ============================================================================
# 2. evaluate and train functions
# ============================================================================
def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Handle both float and long labels
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        loss = F.cross_entropy(out, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += batch.num_graphs
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model with full metrics (Acc, F1, AUC, Precision, Recall)"""
    model.eval()
    total_loss = 0
    
    # Containers: store all predictions and labels for computing sklearn metrics
    all_preds = []
    all_labels = []
    all_probs = [] # for AUC
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * batch.num_graphs
        
        # Get probabilities (Class 1) and predicted classes
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        # Collect data (convert to numpy)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # 1. Basic metrics
    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 2. Advanced metrics (add zero_division=0 to prevent errors)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 # If validation set has only one class, AUC cannot be computed
        
    return avg_loss, accuracy, f1, auc, precision, recall


# ============================================================================
# 3. Main function - Supports two evaluation modes
# ============================================================================
def train_and_evaluate_loo(args, device):
    """
    Leave-One-Recipe-Out Cross-Validation
    Each time, use all samples from one recipe as the test set, train on the rest
    """
    print("\n" + "="*70)
    print("Leave-One-Recipe-Out Cross-Validation")
    print("="*70)
    
    # Load data
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False)
    print(f"\nTotal samples: {len(dataset)}")
    
    # Group by recipe (assumes data has task_name or recipe_id field)
    recipe_groups = {}
    for idx, data in enumerate(dataset):
        recipe_name = data.task_name if hasattr(data, 'task_name') else data.recipe_id
        if recipe_name not in recipe_groups:
            recipe_groups[recipe_name] = []
        recipe_groups[recipe_name].append(idx)
    
    recipes = sorted(recipe_groups.keys())
    print(f"Found {len(recipes)} recipes: {recipes}")
    
    # Store results for each recipe
    recipe_results = []
    
    # Perform leave-one-out test for each recipe
    for test_recipe in recipes:
        print(f"\n{'='*70}")
        print(f"Testing on recipe: {test_recipe}")
        print(f"{'='*70}")
        
        # Split train and test sets
        test_indices = recipe_groups[test_recipe]
        train_indices = []
        for recipe in recipes:
            if recipe != test_recipe:
                train_indices.extend(recipe_groups[recipe])
        
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        
        print(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        in_channels = dataset[0].x.shape[1]
        if args.model_type == 'dagnn':
            model = DAGNN(
                in_channels=in_channels,
                hidden_channels=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=2,
                dropout=args.dropout
            )
        else:
            model = GCNClassifier(
                in_channels=in_channels,
                hidden_channels=args.hidden_dim,
                num_layers=args.num_layers,
                num_classes=2,
                dropout=args.dropout
            )
        model = model.to(device)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Training loop
        best_train_acc = 0
        patience_counter = 0
        
        for epoch in range(1, args.num_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        
        # Evaluate on test set
        test_loss, test_acc, test_f1, test_auc, test_prec, test_rec = evaluate(model, test_loader, device)
        
        print(f"\n  Test Results for {test_recipe}:")
        print(f"    Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print(f"    Precision={test_prec:.4f}, Recall={test_rec:.4f}")
        
        recipe_results.append({
            'recipe': test_recipe,
            'accuracy': test_acc,
            'f1': test_f1,
            'auc': test_auc,
            'precision': test_prec,
            'recall': test_rec
        })
    
    # Summarize results
    print("\n" + "="*70)
    print("Per-Recipe Results Summary")
    print("="*70)
    for result in recipe_results:
        print(f"{result['recipe']:25s}: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc']:.4f}")
    
    # Calculate average performance
    avg_acc = np.mean([r['accuracy'] for r in recipe_results])
    avg_f1 = np.mean([r['f1'] for r in recipe_results])
    avg_auc = np.mean([r['auc'] for r in recipe_results])
    avg_prec = np.mean([r['precision'] for r in recipe_results])
    avg_rec = np.mean([r['recall'] for r in recipe_results])
    
    print("\n" + "="*70)
    print("Average Performance Across All Recipes")
    print("="*70)
    print(f"Accuracy:  {avg_acc:.4f} ± {np.std([r['accuracy'] for r in recipe_results]):.4f}")
    print(f"F1 Score:  {avg_f1:.4f} ± {np.std([r['f1'] for r in recipe_results]):.4f}")
    print(f"AUC:       {avg_auc:.4f} ± {np.std([r['auc'] for r in recipe_results]):.4f}")
    print(f"Precision: {avg_prec:.4f} ± {np.std([r['precision'] for r in recipe_results]):.4f}")
    print(f"Recall:    {avg_rec:.4f} ± {np.std([r['recall'] for r in recipe_results]):.4f}")
    
    # Save results to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "loo_results.csv"
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Recipe", "Accuracy", "F1", "AUC", "Precision", "Recall"])
        for result in recipe_results:
            writer.writerow([
                result['recipe'],
                f"{result['accuracy']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['auc']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}"
            ])
        writer.writerow(["Average", f"{avg_acc:.4f}", f"{avg_f1:.4f}", f"{avg_auc:.4f}", 
                        f"{avg_prec:.4f}", f"{avg_rec:.4f}"])
    
    print(f"\nPer-recipe results saved to {log_file}")
    return avg_acc, avg_f1, avg_auc


def main(args):
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*70)
    print("Task Graph Classification")
    print("="*70)
    
    # Select evaluation mode
    if args.eval_mode == 'loo':
        # Leave-one-out cross-validation
        avg_acc, avg_f1, avg_auc = train_and_evaluate_loo(args, device)
        return
    
    # Otherwise use standard training process
    print(f"\nLoading data from {args.data_path}...")
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False)
    
    # Simple validation check
    sample = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Input channels: {sample.x.shape[1]} (expected: 258 = 256 features + 1 time + 1 mask)")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    in_channels = sample.x.shape[1]
    print(f"\nCreating {args.model_type.upper()} model...")
    
    if args.model_type == 'dagnn':
        model = DAGNN(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout
        )
    elif args.model_type == 'gcn':
        model = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=args.hidden_dim,
            num_layers=args.num_layers,
            num_classes=2,
            dropout=args.dropout
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    
    best_val_acc = 0
    best_val_f1 = 0   #  F1
    best_val_auc = 0  #  AUC
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        val_loss, val_acc, val_f1, val_auc, val_prec, val_rec = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1   # Update best record
            best_val_auc = val_auc # Update best record
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = Path("results/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, checkpoint_dir / "best_model.pt")
            # print(f"  → Saved checkpoint (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
    
    print("\n" + "="*60)
    print(f"Training completed! Best Val Acc: {best_val_acc:.4f} (F1: {best_val_f1:.4f}, AUC: {best_val_auc:.4f})")
    print("="*60)

    # ==========================
    # [New Feature]: Auto-save experiment log to CSV
    # ==========================
    # Save results under results/ folder
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model", "Dim", "Layers", "LR", "Drop", "Batch", 
                "Best_Acc", "Best_F1", "Best_AUC", "Total_Epochs"
            ])
        
        # Write data
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M"),
            args.model_type,
            args.hidden_dim,
            args.num_layers,
            args.lr,
            args.dropout,
            args.batch_size,
            f"{best_val_acc:.4f}",
            f"{best_val_f1:.4f}",
            f"{best_val_auc:.4f}",
            epoch
        ])
    print(f"Experiment results saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Training Script")
    
    # Data arguments
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed_graphs.pt',
        help='Path to preprocessed graphs (.pt file)'
    )
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='dagnn',
                       choices=['dagnn', 'gcn'],
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
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    
    # Evaluation mode
    parser.add_argument('--eval_mode', type=str, default='standard',
                       choices=['standard', 'loo'],
                       help='Evaluation mode: standard (80/20 split) or loo (leave-one-recipe-out)')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)