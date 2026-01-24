"""
Evaluation Script for Task Graph Classification

Loads a trained model checkpoint and evaluates it on test data.
Supports both LOO checkpoints (per-recipe) and standard checkpoints.
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from train import PreloadedGraphDataset
from models.dagnn import DAGNN, GCNClassifier


@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * batch.num_graphs
        
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Incorrect', 'Correct'],
                yticklabels=['Incorrect', 'Correct'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        model_type = config['model_type']
        
        # Create model
        if model_type == 'dagnn':
            model = DAGNN(
                in_channels=config['in_channels'],
                hidden_channels=config['hidden_channels'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes'],
                dropout=config['dropout']
            )
        else:  # gcn
            model = GCNClassifier(
                in_channels=config['in_channels'],
                hidden_channels=config['hidden_channels'],
                num_layers=config['num_layers'],
                num_classes=config['num_classes'],
                dropout=config['dropout']
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("Checkpoint missing 'model_config' field. Cannot reconstruct model.")
    
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def evaluate_loo_checkpoints(args, device):
    """Evaluate all LOO checkpoints and report per-recipe results."""
    checkpoint_dir = Path("results") / "checkpoints" / "loo"
    
    if not checkpoint_dir.exists():
        print(f"❌ LOO checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Load dataset
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False)
    
    # Group by recipe
    recipe_groups = {}
    for idx, data in enumerate(dataset):
        recipe_name = data.task_name if hasattr(data, 'task_name') else data.recipe_id
        if recipe_name not in recipe_groups:
            recipe_groups[recipe_name] = []
        recipe_groups[recipe_name].append(idx)
    
    print("\n" + "="*70)
    print("Evaluating LOO Checkpoints")
    print("="*70)
    
    results = []
    
    for recipe_name in sorted(recipe_groups.keys()):
        checkpoint_path = checkpoint_dir / f"{recipe_name}_best.pt"
        
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found for {recipe_name}")
            continue
        
        # Load model
        model, checkpoint = load_model(checkpoint_path, device)
        
        # Create test loader for this recipe
        test_indices = recipe_groups[recipe_name]
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, device)
        
        print(f"\n{recipe_name}:")
        print(f"  Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"AUC: {metrics['auc']:.4f}")
        
        results.append({
            'recipe': recipe_name,
            **metrics
        })
    
    # Aggregate results
    if results:
        print("\n" + "="*70)
        print("Average Performance")
        print("="*70)
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_auc = np.mean([r['auc'] for r in results])
        print(f"Accuracy:  {avg_acc:.4f}")
        print(f"F1 Score:  {avg_f1:.4f}")
        print(f"AUC:       {avg_auc:.4f}")


def evaluate_standard_checkpoint(args, device):
    """Evaluate a standard (single) checkpoint on test data."""
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading checkpoint from {checkpoint_path}")
    
    # Load model
    model, checkpoint = load_model(checkpoint_path, device)
    
    # Load dataset
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False)
    
    # Use same random split as training (if seed is the same)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Evaluating on {len(val_dataset)} samples...")
    
    # Evaluate
    metrics = evaluate_model(model, val_loader, device)
    
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"Loss:      {metrics['loss']:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    
    # Plot confusion matrix
    output_dir = Path("results") / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics['confusion_matrix'], 
                         save_path=output_dir / "confusion_matrix.png")
    
    print(f"\n✅ Evaluation complete!")


def main(args):
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.eval_loo:
        # Evaluate all LOO checkpoints
        evaluate_loo_checkpoints(args, device)
    else:
        # Evaluate single checkpoint
        evaluate_standard_checkpoint(args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/processed_graphs.pt',
        help='Path to preprocessed graphs'
    )
    parser.add_argument(
        '--eval_loo',
        action='store_true',
        help='Evaluate all LOO checkpoints instead of a single checkpoint'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (for standard eval)'
    )
    
    args = parser.parse_args()
    main(args)
