"""Standard-only K-fold training script.

This script keeps the main `train.py` lightweight by moving the k-fold
evaluation path into a dedicated entry point. It reuses the shared helpers
from `train.py`:
- `PreloadedGraphDataset`
- `compute_class_weights`
- `create_model`
- `train_epoch`
- `evaluate`
- `find_best_threshold`
- `save_checkpoint`
- `init_wandb`
- `log_to_wandb`

The protocol is:
1. Stratified K-fold split on the full dataset.
2. Inside each fold, split the training portion into train/validation.
3. Use validation F1 for early stopping and threshold tuning.
4. Evaluate the held-out fold with the tuned threshold.
5. Save per-fold rows and an average row to `results/`.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import train as base


def stratified_train_val_split(indices, labels, val_fraction: float, seed: int):
    """Split indices into train/validation subsets while preserving label balance."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    subset_labels = labels[indices]

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_rel, val_rel = next(splitter.split(np.zeros(len(indices)), subset_labels))
        return indices[train_rel].tolist(), indices[val_rel].tolist()
    except ValueError:
        rng = np.random.default_rng(seed)
        permuted = indices[rng.permutation(len(indices))]
        if len(permuted) <= 1:
            return permuted.tolist(), permuted.tolist()

        val_size = max(1, int(round(len(permuted) * val_fraction)))
        val_size = min(val_size, len(permuted) - 1)
        val_indices = permuted[:val_size].tolist()
        train_indices = permuted[val_size:].tolist()
        if not train_indices:
            train_indices = val_indices[:1]
            val_indices = val_indices[1:] if len(val_indices) > 1 else val_indices
        return train_indices, val_indices


def load_existing_rows(csv_path: Path):
    """Load already completed fold rows so the script can resume safely."""
    if not csv_path.exists():
        return [], set()

    rows = []
    completed = set()
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fold_value = str(row.get("Fold", "")).strip()
            if not fold_value or fold_value.lower() == "average":
                continue

            try:
                fold_idx = int(float(fold_value))
            except ValueError:
                continue

            parsed = {"Fold": fold_idx}
            for key, value in row.items():
                if key == "Fold":
                    continue
                if value in (None, ""):
                    parsed[key] = ""
                    continue
                try:
                    parsed[key] = int(float(value)) if key in {"Layers", "Train_Size", "Val_Size", "Test_Size", "Epochs"} else float(value)
                except ValueError:
                    parsed[key] = value

            rows.append(parsed)
            completed.add(fold_idx)

    return rows, completed


def write_results(csv_path: Path, rows):
    """Write fold rows and an average row to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Fold",
        "Model",
        "Hidden_Dim",
        "Layers",
        "LR",
        "Dropout",
        "Train_Size",
        "Val_Size",
        "Test_Size",
        "Best_Val_F1",
        "Best_Val_Acc",
        "Best_Val_AUC",
        "Best_Threshold",
        "Test_Accuracy",
        "Test_F1",
        "Test_AUC",
        "Test_Precision",
        "Test_Recall",
        "Epochs",
        "Duration_Sec",
    ]

    numeric_cols = [
        "Best_Val_F1",
        "Best_Val_Acc",
        "Best_Val_AUC",
        "Best_Threshold",
        "Test_Accuracy",
        "Test_F1",
        "Test_AUC",
        "Test_Precision",
        "Test_Recall",
        "Duration_Sec",
    ]

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

        if rows:
            avg_row = {"Fold": "Average"}
            for key in ["Model", "Hidden_Dim", "Layers", "LR", "Dropout"]:
                avg_row[key] = rows[0].get(key, "")
            for key in ["Train_Size", "Val_Size", "Test_Size", "Epochs"]:
                avg_row[key] = ""
            for key in numeric_cols:
                values = [r[key] for r in rows if r.get(key) not in (None, "")]
                avg_row[key] = float(np.mean(values)) if values else ""
            writer.writerow(avg_row)


def run_kfold_standard(args):
    print("\n" + "=" * 70)
    print(f"Standard K-Fold Evaluation ({args.k_folds}-fold)")
    print("  - Protocol: standard split only")
    print("  - Outer folds: stratified K-fold over the full dataset")
    print("  - Inner split: train/validation for early stopping and threshold tuning")
    print("=" * 70)

    dataset = base.PreloadedGraphDataset(args.data_path, weights_only=False)
    labels = np.array([dataset[i].y.item() for i in range(len(dataset))])

    if len(np.unique(labels)) < 2:
        raise ValueError("K-fold requires at least two classes in the dataset.")

    dataset_tag = Path(args.data_path).stem
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"kfold_standard_{dataset_tag}.csv"

    fold_rows, completed_folds = load_existing_rows(csv_path)
    if completed_folds:
        print(f"\n📂 Resuming from existing results: {csv_path}")
        print(f"✅ Completed folds found: {sorted(completed_folds)}")

    outer_splitter = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    all_indices = np.arange(len(dataset))
    in_channels = dataset[0].x.shape[1]

    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_splitter.split(all_indices, labels), start=1):
        if fold_idx in completed_folds:
            print(f"[Fold {fold_idx}/{args.k_folds}] ⏭️  Skipping (already completed)")
            continue

        print(f"\n{'=' * 70}")
        print(f"[Fold {fold_idx}/{args.k_folds}] Testing on held-out fold")
        print(f"Train+Val size: {len(train_val_idx)} | Test size: {len(test_idx)}")
        print(f"{'=' * 70}")

        inner_train_idx, inner_val_idx = stratified_train_val_split(
            train_val_idx,
            labels,
            val_fraction=args.val_fraction,
            seed=args.seed + fold_idx,
        )

        train_labels = labels[inner_train_idx]
        class_weights = base.compute_class_weights(train_labels).to(args.device)
        print(f"✓ Class Weights: {class_weights.tolist()}")

        train_loader = base.DataLoader(
            torch.utils.data.Subset(dataset, inner_train_idx),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = base.DataLoader(
            torch.utils.data.Subset(dataset, inner_val_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = base.DataLoader(
            torch.utils.data.Subset(dataset, test_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )

        model = base.create_model(
            args.model_type,
            in_channels,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
            args.device,
            pooling=args.pooling,
            input_dropout=args.input_dropout,
            classifier_dropout=args.classifier_dropout,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_f1 = 0.0
        best_val_acc = 0.0
        best_val_auc = 0.0
        best_threshold = 0.6
        patience_counter = 0
        epochs_run = 0
        fold_start = time.time()

        for epoch in range(1, args.num_epochs + 1):
            epochs_run = epoch
            train_loss, train_acc = base.train_epoch(
                model,
                train_loader,
                optimizer,
                args.device,
                class_weights=class_weights,
            )

            val_loss, val_acc, val_f1, val_auc, _, _, val_probs, val_labels = base.evaluate(
                model,
                val_loader,
                args.device,
                threshold=None,
            )

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}, AUC={val_auc:.4f}"
                )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_val_acc = val_acc
                best_val_auc = val_auc
                patience_counter = 0
                best_threshold, tuned_val_f1 = base.find_best_threshold(val_labels, val_probs, metric="f1")
                print(f"  ✓ New best validation F1! Threshold: {best_threshold:.3f} (tuned F1: {tuned_val_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        test_loss, test_acc, test_f1, test_auc, test_prec, test_rec, _, _ = base.evaluate(
            model,
            test_loader,
            args.device,
            threshold=best_threshold,
        )

        duration_sec = time.time() - fold_start

        checkpoint_path = results_dir / "checkpoints" / "kfold_standard" / dataset_tag / f"fold_{fold_idx}_best.pt"
        base.save_checkpoint(
            model,
            checkpoint_path,
            fold=fold_idx,
            test_acc=test_acc,
            test_f1=test_f1,
            test_auc=test_auc,
            best_threshold=best_threshold,
            model_config={
                "model_type": args.model_type,
                "in_channels": in_channels,
                "hidden_channels": args.hidden_dim,
                "num_layers": args.num_layers,
                "num_classes": 2,
                "dropout": args.dropout,
                "pooling": args.pooling,
                "input_dropout": args.input_dropout,
                "classifier_dropout": args.classifier_dropout,
            },
            args=vars(args),
        )

        print(f"\n  Fold {fold_idx} Test Results (Threshold: {best_threshold:.3f}):")
        print(f"    Acc={test_acc:.4f}, F1={test_f1:.4f}, AUC={test_auc:.4f}")
        print(f"    Precision={test_prec:.4f}, Recall={test_rec:.4f}")

        fold_rows.append(
            {
                "Fold": fold_idx,
                "Model": args.model_type,
                "Hidden_Dim": args.hidden_dim,
                "Layers": args.num_layers,
                "LR": args.lr,
                "Dropout": args.dropout,
                "Train_Size": len(inner_train_idx),
                "Val_Size": len(inner_val_idx),
                "Test_Size": len(test_idx),
                "Best_Val_F1": best_val_f1,
                "Best_Val_Acc": best_val_acc,
                "Best_Val_AUC": best_val_auc,
                "Best_Threshold": best_threshold,
                "Test_Accuracy": test_acc,
                "Test_F1": test_f1,
                "Test_AUC": test_auc,
                "Test_Precision": test_prec,
                "Test_Recall": test_rec,
                "Epochs": epochs_run,
                "Duration_Sec": duration_sec,
            }
        )
        write_results(csv_path, fold_rows)
        print(f"💾 Saved fold {fold_idx} results to {csv_path}")

        base.log_to_wandb(
            {
                f"kfold/fold_{fold_idx}_test_accuracy": test_acc,
                f"kfold/fold_{fold_idx}_test_f1": test_f1,
                f"kfold/fold_{fold_idx}_test_auc": test_auc,
                f"kfold/fold_{fold_idx}_best_threshold": best_threshold,
            },
            step=fold_idx,
        )

    if not fold_rows:
        print("No new folds were run.")
        return None

    avg_test_acc = float(np.mean([r["Test_Accuracy"] for r in fold_rows]))
    avg_test_f1 = float(np.mean([r["Test_F1"] for r in fold_rows]))
    avg_test_auc = float(np.mean([r["Test_AUC"] for r in fold_rows]))
    avg_test_prec = float(np.mean([r["Test_Precision"] for r in fold_rows]))
    avg_test_rec = float(np.mean([r["Test_Recall"] for r in fold_rows]))
    avg_threshold = float(np.mean([r["Best_Threshold"] for r in fold_rows]))

    print("\n" + "=" * 70)
    print(f"Standard K-Fold Completed ({len(fold_rows)}/{args.k_folds} folds)")
    print(f"  Accuracy:  {avg_test_acc:.4f} ± {np.std([r['Test_Accuracy'] for r in fold_rows]):.4f}")
    print(f"  F1 Score:  {avg_test_f1:.4f} ± {np.std([r['Test_F1'] for r in fold_rows]):.4f}")
    print(f"  AUC:       {avg_test_auc:.4f} ± {np.std([r['Test_AUC'] for r in fold_rows]):.4f}")
    print(f"  Precision: {avg_test_prec:.4f} ± {np.std([r['Test_Precision'] for r in fold_rows]):.4f}")
    print(f"  Recall:    {avg_test_rec:.4f} ± {np.std([r['Test_Recall'] for r in fold_rows]):.4f}")
    print(f"  Avg Threshold: {avg_threshold:.3f}")
    print(f"  CSV: {csv_path}")
    print("=" * 70)

    base.log_to_wandb(
        {
            "summary/kfold_avg_accuracy": avg_test_acc,
            "summary/kfold_avg_f1": avg_test_f1,
            "summary/kfold_avg_auc": avg_test_auc,
            "summary/kfold_avg_precision": avg_test_prec,
            "summary/kfold_avg_recall": avg_test_rec,
            "summary/kfold_avg_threshold": avg_threshold,
        }
    )

    return avg_test_acc, avg_test_f1, avg_test_auc


def build_parser():
    parser = argparse.ArgumentParser(description="Standard-only K-fold training script")

    parser.add_argument("--data_path", type=str, default="data/processed_graphs.pt")
    parser.add_argument("--model_type", type=str, default="dagnn", choices=["dagnn", "gcn"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--input_dropout", type=float, default=0.3)
    parser.add_argument("--classifier_dropout", type=float, default=0.5)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "mean_max"])

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--val_fraction", type=float, default=0.2)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="task-graph-classification")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    base.init_wandb(args)

    try:
        device = torch.device(args.device)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        print(f"Using device: {device}")
        print("=" * 70)
        print("Standard K-Fold Runner")
        print("=" * 70)

        run_kfold_standard(args)
    finally:
        if base.wandb is not None and base.wandb.run is not None:
            base.wandb.finish()


if __name__ == "__main__":
    main()