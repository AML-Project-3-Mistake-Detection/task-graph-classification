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

# 引入 sklearn 指标
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# 假设你的模型文件在这里
from models.dagnn import DAGNN, GCNClassifier


# ============================================================================
# 1. Dataset 类 (保持你的极简版)
# ============================================================================
class PreloadedGraphDataset(Dataset):
    """
    直接加载已经处理好的 .pt 文件，无需复杂初始化
    """
    def __init__(self, pt_file_path, weights_only=False):
        super().__init__()
        self.data_list = torch.load(pt_file_path, weights_only=weights_only)
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]


# ============================================================================
# 2. 训练与评估函数 (修改了 evaluate 以计算更多指标)
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
        
        # 兼容 float/long 标签
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
    
    # 容器：存储所有预测和标签，用于计算 sklearn 指标
    all_preds = []
    all_labels = []
    all_probs = [] # 用于 AUC
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        
        y = batch.y.long().view(-1) if batch.y.dtype == torch.float32 else batch.y
        loss = F.cross_entropy(out, y)
        total_loss += loss.item() * batch.num_graphs
        
        # 获取概率 (Class 1) 和 预测类别
        probs = torch.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        
        # 收集数据 (转为 numpy)
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    
    # 1. 基础指标
    avg_loss = total_loss / len(loader.dataset)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # 2. 高级指标 (加 zero_division=0 防止报错)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5 # 如果验证集只有一类数据，AUC 无法计算
        
    return avg_loss, accuracy, f1, auc, precision, recall


# ============================================================================
# 3. Main 函数
# ============================================================================
def main(args):
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("="*60)
    print("Task Graph Classification - Training")
    print("="*60)
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    dataset = PreloadedGraphDataset(args.data_path, weights_only=False)
    
    # 简单检查
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
    best_val_f1 = 0   # 记录最佳 F1
    best_val_auc = 0  # 记录最佳 AUC
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        
        # [核心修改]: 这里正确接收 6 个返回值
        val_loss, val_acc, val_f1, val_auc, val_prec, val_rec = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch:03d}: "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1   # 更新最佳记录
            best_val_auc = val_auc # 更新最佳记录
            patience_counter = 0
            
            # 保存 Checkpoint
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
    # [新增功能]: 自动保存实验记录到 CSV
    # ==========================
    # Save results under results/ folder
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # 写表头
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model", "Dim", "Layers", "LR", "Drop", "Batch", 
                "Best_Acc", "Best_F1", "Best_AUC", "Total_Epochs"
            ])
        
        # 写数据
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
    print(f"📝 Experiment results saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simplified Training Script")
    
    # Data arguments (保留你的原始参数定义)
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
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    main(args)