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
import json
from datetime import datetime

# Import sklearn metrics
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

from models.dagnn import DAGNN, GCNClassifier

try:
    import wandb
except ImportError:
    wandb = None


# ============================================================================
# 1. Helper Functions
# ============================================================================
def create_model(
    model_type,
    in_channels,
    hidden_channels,
    num_layers,
    dropout,
    device,
    pooling='mean',
    input_dropout=0.3,
    classifier_dropout=0.5,
):
    """Create model based on configuration."""
    if model_type == 'dagnn':
        model = DAGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout,
            input_dropout=input_dropout,
            classifier_dropout=classifier_dropout,
            pooling=pooling,
        )
    elif model_type == 'gcn':
        model = GCNClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_classes=2,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def compute_class_weights(labels):
    """
    Compute class weights to handle imbalanced datasets.
    Classes with fewer samples get higher weights.
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = {}
    for cls, count in zip(unique, counts):
        # Weight = inverse of class frequency
        weights[cls] = total / (len(unique) * count)
    
    # Convert to tensor in proper format for PyTorch
    class_weights = torch.tensor([weights.get(i, 1.0) for i in range(len(unique))], 
                                 dtype=torch.float32)
    return class_weights


def find_best_threshold(y_true, y_probs, metric='f1'):
    """
    Find best classification threshold by sweeping on validation set.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities for class 1
        metric: 'f1' (recommended), 'accuracy', or 'precision'
    
    Returns:
        best_threshold, best_metric_value
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = 0.6
    best_value = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'accuracy':
            value = np.mean(y_pred == y_true)
        elif metric == 'precision':
            value = precision_score(y_true, y_pred, average='macro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if value > best_value:
            best_value = value
            best_threshold = threshold
    
    return best_threshold, best_value


def save_checkpoint(model, checkpoint_path, **metadata):
    """Save model checkpoint with metadata."""
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **metadata
    }
    torch.save(checkpoint, checkpoint_path)


def init_wandb(args):
    """Initialize Weights & Biases run if enabled."""
    if not args.use_wandb:
        return None

    if wandb is None:
        print("⚠️  wandb is not installed. Disable --use_wandb or install wandb.")
        return None

    run_name = args.wandb_run_name
    if not run_name:
        run_name = (
            f"{args.eval_mode}_{args.model_type}_h{args.hidden_dim}_"
            f"l{args.num_layers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        mode=args.wandb_mode,
        config=vars(args),
        tags=[args.eval_mode, args.model_type],
    )


def log_to_wandb(metrics, step=None):
    """Safe wrapper for logging metrics to wandb."""
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_curves_to_wandb(split_name, y_true, y_probs, step=None):
    """Log ROC/PR curve images and curve tables for binary classification."""
    if wandb is None or wandb.run is None:
        return

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    # ROC/AUC curve is undefined when only one class is present.
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    roc_table = wandb.Table(data=[[float(x), float(y)] for x, y in zip(fpr, tpr)], columns=['fpr', 'tpr'])
    pr_table = wandb.Table(data=[[float(x), float(y)] for x, y in zip(recall, precision)], columns=['recall', 'precision'])

    log_data = {
        f'{split_name}/roc_table': roc_table,
        f'{split_name}/pr_table': pr_table,
    }

    try:
        import matplotlib.pyplot as plt

        roc_auc = roc_auc_score(y_true, y_probs)

        fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
        ax_roc.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
        ax_roc.plot([0, 1], [0, 1], linestyle='--', alpha=0.6)
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve ({split_name})')
        ax_roc.legend(loc='lower right')
        fig_roc.tight_layout()
        log_data[f'{split_name}/roc_curve'] = wandb.Image(fig_roc)
        plt.close(fig_roc)

        fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
        ax_pr.plot(recall, precision)
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'PR Curve ({split_name})')
        fig_pr.tight_layout()
        log_data[f'{split_name}/pr_curve'] = wandb.Image(fig_pr)
        plt.close(fig_pr)
    except Exception:
        # Keep scalar/table logging even if plotting backend is unavailable.
        pass

    wandb.log(log_data, step=step)


# ============================================================================
# 2. Dataset
# ============================================================================
class PreloadedGraphDataset(Dataset):
    """
    Load preprocessed .pt files directly without complex initialization
    """
    def __init__(self, pt_file_path, weights_only=False, device='cpu'):
        super().__init__()
        self.data_list = torch.load(pt_file_path, weights_only=weights_only, map_location=device)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


# ============================================================================
# 2. evaluate and train functions
# ============================================================================
def train_epoch(model, loader, optimizer, device, class_weights=None):
    """Train for one epoch with optional class weights"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Move class weights to device if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Handle both float and long labels
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        
        # Cross entropy with class weights
        if class_weights is not None:
            loss = F.cross_entropy(out, y, weight=class_weights)
        else:
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
def evaluate(model, loader, device, threshold=None):
    """
    Evaluate model with full metrics (Acc, F1, AUC, Precision, Recall)
    
    Args:
        threshold: If provided, use this threshold instead of 0.6 for binary classification
    """
    model.eval()
    total_loss = 0
    
    # Containers: store all predictions and labels for computing sklearn metrics
    all_preds = []
    all_labels = []
    all_probs = [] # for AUC and threshold tuning
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * batch.num_graphs
        
        # Get probabilities (Class 1) and predicted classes
        probs = torch.softmax(out, dim=1)[:, 1]
        
        # Use custom threshold if provided, else default to 0.6
        current_threshold = threshold if threshold is not None else 0.6
        preds = (probs >= current_threshold).long()
        
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
        
    return avg_loss, accuracy, f1, auc, precision, recall, np.array(all_probs), np.array(all_labels)


# ============================================================================
# 4. Training Functions
# ============================================================================
def train_standard(args, device):
    """
    Standard train/val split (80/20) training with:
    - Class weights for imbalanced data
    - Best-by-F1 checkpoint saving
    - Threshold tuning on validation set
    """
    print("\n" + "="*70)
    print("Standard Training (80/20 split) with Improvements")
    print("  - Class Weights: YES (handles imbalanced data)")
    print("  - Best-by-F1: YES (saves best F1 model)")
    print("  - Threshold Tuning: YES (optimizes on validation set)")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from {args.data_path}...")
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False, device=device)
    
    sample = dataset[0]
    print(f"Dataset size: {len(dataset)}")
    print(f"Input channels: {sample.x.shape[1]}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {train_size} | Val size: {val_size}")

    log_to_wandb({
        'data/total_samples': len(dataset),
        'data/train_samples': train_size,
        'data/val_samples': val_size,
        'data/input_channels': int(sample.x.shape[1]),
    })
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compute class weights for training set
    train_indices = train_dataset.indices
    train_labels = np.array([dataset[i].y.item() for i in train_indices])
    class_weights = compute_class_weights(train_labels)
    class_weights = class_weights.to(device)
    print(f"\n✓ Class Weights: {class_weights.tolist()}")
    
    # Create model
    in_channels = sample.x.shape[1]
    print(f"\nCreating {args.model_type.upper()} model...")
    model = create_model(args.model_type, in_channels, args.hidden_dim, 
                        args.num_layers, args.dropout, device,
                        pooling=args.pooling,
                        input_dropout=args.input_dropout,
                        classifier_dropout=args.classifier_dropout)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0  # Now tracking best F1 instead of best Acc
    best_val_acc = 0
    best_val_auc = 0
    best_threshold = 0.6  # Store best threshold
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, class_weights=class_weights)
        
        # Evaluate on validation set with default threshold
        val_loss, val_acc, val_f1, val_auc, val_prec, val_rec, val_probs, val_labels = evaluate(
            model, val_loader, device, threshold=None
        )
        
        print(f"Epoch {epoch:03d}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

        log_to_wandb({
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'val/f1': val_f1,
            'val/auc': val_auc,
            'val/precision': val_prec,
            'val/recall': val_rec,
        }, step=epoch)
        
        # IMPROVEMENT 1 & 2: Save best model based on F1 (not Acc)
        # Also tune threshold on validation set
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_val_auc = val_auc
            patience_counter = 0
            
            # IMPROVEMENT 3: Find best threshold on validation set
            threshold, threshold_f1 = find_best_threshold(val_labels, val_probs, metric='f1')
            best_threshold = threshold
            
            print(f"  ✓ New best F1! Threshold: {best_threshold:.3f} (tuned F1: {threshold_f1:.4f})")

            log_to_wandb({
                'best/val_f1': best_val_f1,
                'best/val_acc': best_val_acc,
                'best/val_auc': best_val_auc,
                'best/threshold': best_threshold,
                'best/tuned_val_f1': threshold_f1,
            }, step=epoch)

            log_curves_to_wandb('best_val', val_labels, val_probs, step=epoch)
            
            save_checkpoint(
                model,
                Path("results/checkpoints/best_model.pt"),
                epoch=epoch,
                optimizer_state_dict=optimizer.state_dict(),
                val_acc=val_acc,
                val_f1=val_f1,
                val_auc=val_auc,
                best_threshold=best_threshold,  # Save optimal threshold
                model_config={
                    'model_type': args.model_type,
                    'in_channels': in_channels,
                    'hidden_channels': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'num_classes': 2,
                    'dropout': args.dropout,
                    'pooling': args.pooling,
                    'input_dropout': args.input_dropout,
                    'classifier_dropout': args.classifier_dropout,
                },
                args=vars(args)
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
    
    print("\n" + "="*70)
    print(f"Training completed!")
    print(f"  Best Val F1: {best_val_f1:.4f} (Acc: {best_val_acc:.4f}, AUC: {best_val_auc:.4f})")
    print(f"  Best Threshold: {best_threshold:.3f}")
    print("="*70)

    # Save experiment log to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "experiment_results.csv"
    file_exists = log_file.exists()
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model", "Dim", "Layers", "LR", "Drop", "Batch", 
                "Best_F1", "Best_Acc", "Best_AUC", "Best_Threshold", "Total_Epochs"
            ])
        
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M"),
            args.model_type,
            args.hidden_dim,
            args.num_layers,
            args.lr,
            args.dropout,
            args.batch_size,
            f"{best_val_f1:.4f}",
            f"{best_val_acc:.4f}",
            f"{best_val_auc:.4f}",
            f"{best_threshold:.3f}",
            epoch
        ])
    
    print(f"Experiment results saved to {log_file}")
    log_to_wandb({
        'summary/best_val_f1': best_val_f1,
        'summary/best_val_acc': best_val_acc,
        'summary/best_val_auc': best_val_auc,
        'summary/best_threshold': best_threshold,
    })
    return best_val_f1, best_val_acc, best_val_auc


def train_and_evaluate_loo(args, device):
    """
    Leave-One-Recipe-Out Cross-Validation with improvements:
    - Class weights for imbalanced data
    - Best-by-F1 checkpoint saving
    - Threshold tuning on training set
    """
    print("\n" + "="*70)
    print("Leave-One-Recipe-Out Cross-Validation (with improvements)")
    print("  - Class Weights: YES")
    print("  - Best-by-F1: YES")
    print("  - Threshold Tuning: YES")
    print("="*70)
    
    # Load data
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False, device=device)
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

    log_to_wandb({
        'data/total_samples': len(dataset),
        'data/num_recipes': len(recipes),
    })
    
    # Store results for each recipe
    recipe_results = []
    
    # Perform leave-one-out test for each recipe
    for recipe_idx, test_recipe in enumerate(recipes):
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
        
        # Split train_dataset into inner_train (80%) and inner_val (20%)
        inner_train_size = int(0.8 * len(train_dataset))
        inner_val_size = len(train_dataset) - inner_train_size
        inner_train_dataset, inner_val_dataset = torch.utils.data.random_split(
            train_dataset, [inner_train_size, inner_val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        print(f"Inner Train size: {len(inner_train_dataset)} | Inner Val size: {len(inner_val_dataset)} | Test size: {len(test_dataset)}")
        
        # Create data loaders
        inner_train_loader = DataLoader(inner_train_dataset, batch_size=args.batch_size, shuffle=True)
        inner_val_loader = DataLoader(inner_val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Compute class weights on inner training set
        inner_train_indices = [train_dataset.indices[i] for i in inner_train_dataset.indices]
        train_labels = np.array([dataset[i].y.item() for i in inner_train_indices])
        class_weights = compute_class_weights(train_labels)
        class_weights = class_weights.to(device)
        
        # Create model
        in_channels = dataset[0].x.shape[1]
        model = create_model(args.model_type, in_channels, args.hidden_dim,
                           args.num_layers, args.dropout, device,
                           pooling=args.pooling,
                           input_dropout=args.input_dropout,
                           classifier_dropout=args.classifier_dropout)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Training loop with Best-by-F1 on inner validation set
        best_val_f1 = 0
        best_val_threshold = 0.6
        best_val_probs = None
        best_val_labels = None
        best_epoch_num = 0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(1, args.num_epochs + 1):
            train_loss, train_acc = train_epoch(model, inner_train_loader, optimizer, device, class_weights=class_weights)
            
            # Evaluate on inner validation set
            val_loss, val_acc, val_f1, val_auc, val_prec, val_rec, val_probs, val_labels = evaluate(
                model, inner_val_loader, device, threshold=None
            )
            
            # Track best inner validation F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_probs = val_probs
                best_val_labels = val_labels
                patience_counter = 0
                best_epoch_num = epoch
                
                # Find best threshold on inner validation set
                threshold, threshold_f1 = find_best_threshold(val_labels, val_probs, metric='f1')
                best_val_threshold = threshold
                
                # Save best epoch weights
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:03d}: "
                          f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                          f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f} "
                          f"[✓ Best, Threshold={best_val_threshold:.3f}]")
            else:
                patience_counter += 1
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:03d}: "
                          f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                          f"Val Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}")
            
            if patience_counter >= args.patience:
                print(f"  Early stopping after {epoch} epochs (best epoch was {best_epoch_num})")
                break
            
            global_step = recipe_idx * args.num_epochs + epoch
            log_to_wandb({
                'loo/train_loss': train_loss,
                'loo/train_acc': train_acc,
                'loo/val_f1': val_f1,
                'loo/val_auc': val_auc,
                'loo/val_threshold': best_val_threshold,
            }, step=global_step)
        
        # Restore best epoch weights
        if best_model_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
        # Evaluate on test set (once, with best epoch and best threshold)
        test_loss, test_acc, test_f1, test_auc, test_prec, test_rec, test_probs, test_labels_np = evaluate(
            model, test_loader, device, threshold=best_val_threshold
        )
        
        # Save checkpoint for this recipe
        checkpoint_path = Path("results") / "checkpoints" / "loo" / f"{test_recipe}_best.pt"
        save_checkpoint(
            model,
            checkpoint_path,
            test_recipe=test_recipe,
            best_epoch=best_epoch_num,
            val_f1=best_val_f1,
            test_acc=test_acc,
            test_f1=test_f1,
            test_auc=test_auc,
            best_threshold=best_val_threshold,
            model_config={
                'model_type': args.model_type,
                'in_channels': in_channels,
                'hidden_channels': args.hidden_dim,
                'num_layers': args.num_layers,
                'num_classes': 2,
                'dropout': args.dropout,
                'pooling': args.pooling,
                'input_dropout': args.input_dropout,
                'classifier_dropout': args.classifier_dropout,
            }
        )
        
        print(f"\n  Test Results for {test_recipe}:")
        print(f"    Best inner-val epoch: {best_epoch_num} (Val F1={best_val_f1:.4f}, Threshold={best_val_threshold:.3f})")
        print(f"    Test Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print(f"    Precision={test_prec:.4f}, Recall={test_rec:.4f}")

        log_to_wandb({
            'loo_test/accuracy': test_acc,
            'loo_test/f1': test_f1,
            'loo_test/auc': test_auc,
            'loo_test/precision': test_prec,
            'loo_test/recall': test_rec,
            'loo_test/threshold': best_val_threshold,
        })

        safe_recipe = str(test_recipe).replace('/', '_').replace(' ', '_')
        log_curves_to_wandb(f'loo_test/{safe_recipe}', test_labels_np, test_probs)
        
        recipe_results.append({
            'recipe': test_recipe,
            'accuracy': test_acc,
            'f1': test_f1,
            'auc': test_auc,
            'precision': test_prec,
            'recall': test_rec,
            'threshold': best_val_threshold
        })
    
    # Summarize results
    print("\n" + "="*70)
    print("Per-Recipe Results Summary")
    print("="*70)
    for result in recipe_results:
        print(f"{result['recipe']:25s}: Acc={result['accuracy']:.4f}, F1={result['f1']:.4f}, AUC={result['auc']:.4f}, Threshold={result['threshold']:.3f}")
    
    # Calculate average performance
    avg_acc = np.mean([r['accuracy'] for r in recipe_results])
    avg_f1 = np.mean([r['f1'] for r in recipe_results])
    avg_auc = np.mean([r['auc'] for r in recipe_results])
    avg_prec = np.mean([r['precision'] for r in recipe_results])
    avg_rec = np.mean([r['recall'] for r in recipe_results])
    avg_threshold = np.mean([r['threshold'] for r in recipe_results])
    
    print("\n" + "="*70)
    print("Average Performance Across All Recipes")
    print("="*70)
    print(f"Accuracy:  {avg_acc:.4f} ± {np.std([r['accuracy'] for r in recipe_results]):.4f}")
    print(f"F1 Score:  {avg_f1:.4f} ± {np.std([r['f1'] for r in recipe_results]):.4f}")
    print(f"AUC:       {avg_auc:.4f} ± {np.std([r['auc'] for r in recipe_results]):.4f}")
    print(f"Precision: {avg_prec:.4f} ± {np.std([r['precision'] for r in recipe_results]):.4f}")
    print(f"Recall:    {avg_rec:.4f} ± {np.std([r['recall'] for r in recipe_results]):.4f}")
    print(f"Avg Threshold: {avg_threshold:.3f}")

    log_to_wandb({
        'summary/loo_avg_accuracy': avg_acc,
        'summary/loo_avg_f1': avg_f1,
        'summary/loo_avg_auc': avg_auc,
        'summary/loo_avg_precision': avg_prec,
        'summary/loo_avg_recall': avg_rec,
        'summary/loo_avg_threshold': avg_threshold,
    })

    if wandb is not None and wandb.run is not None:
        table = wandb.Table(columns=['recipe', 'accuracy', 'f1', 'auc', 'precision', 'recall', 'threshold'])
        for result in recipe_results:
            table.add_data(
                result['recipe'],
                result['accuracy'],
                result['f1'],
                result['auc'],
                result['precision'],
                result['recall'],
                result['threshold'],
            )
        wandb.log({'loo/per_recipe_table': table})
    
    # Save results to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "loo_results.csv"
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Recipe", "Accuracy", "F1", "AUC", "Precision", "Recall", "Threshold"])
        for result in recipe_results:
            writer.writerow([
                result['recipe'],
                f"{result['accuracy']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['auc']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['threshold']:.3f}"
            ])
        writer.writerow(["Average", f"{avg_acc:.4f}", f"{avg_f1:.4f}", f"{avg_auc:.4f}", 
                        f"{avg_prec:.4f}", f"{avg_rec:.4f}", f"{avg_threshold:.3f}"])
    
    print(f"\nPer-recipe results saved to {log_file}")
    
    # 🔑 Flush to disk to ensure complete write
    import os
    try:
        with open(log_file, 'r') as f:
            os.fsync(f.fileno())
    except Exception as e:
        print(f"⚠️  Warning: fsync failed: {e}")
    
    # Create completion marker with verification info
    done_marker = results_dir / "loo_results.done"
    num_recipes = len(recipe_results)
    verification_data = {
        "num_recipes": num_recipes,
        "avg_f1": f"{avg_f1:.6f}",
        "avg_auc": f"{avg_auc:.6f}",
        "timestamp": str(datetime.now()),
        "csv_file": str(log_file)
    }
    with open(done_marker, 'w') as f:
        json.dump(verification_data, f)
    print(f"✅ Created completion marker with verification data: {done_marker.name}")
    
    return avg_acc, avg_f1, avg_auc


# ============================================================================
# 5. Main Entry Point
# ============================================================================
def main(args):
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*70)
    print("Task Graph Classification")
    print("="*70)

    init_wandb(args)

    try:
        # Dispatch to appropriate training function
        if args.eval_mode == 'loo':
            train_and_evaluate_loo(args, device)
        else:
            train_standard(args, device)
    finally:
        if wandb is not None and wandb.run is not None:
            wandb.finish()


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
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--input_dropout', type=float, default=0.3,
                       help='Dropout on raw node inputs before projection (DAGNN only)')
    parser.add_argument('--classifier_dropout', type=float, default=0.5,
                       help='Dropout in classifier head (DAGNN only)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'mean_max'],
                       help='Global pooling mode (DAGNN only)')
    
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

    # W&B logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='task-graph-classification',
                       help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='W&B entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name (optional)')
    parser.add_argument('--wandb_mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled'],
                       help='W&B logging mode')
    
    args = parser.parse_args()
    main(args)