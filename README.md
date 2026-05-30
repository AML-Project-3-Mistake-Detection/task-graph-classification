# Task Graph Classification

Graph Neural Network based classifier for verifying recipe execution correctness.

**Part of:** CaptainCook4D - Mistake Detection Project (Substep 4)

## Overview

This project implements a GNN-based classifier to predict whether a recipe execution is correct by analyzing the observed task-graph structure from video analysis.

## Model Architecture

- **DAGNN** (Directed Acyclic Graph Neural Network) - Recommended
- **GCN** (Graph Convolutional Network)

## Project Structure

```
task-graph-classification/
├── annotations/                # CaptainCook4D annotations (submodule)
│   ├── annotation_csv/         # Annotation tables
│   ├── annotation_json/        # Annotation JSON files
│   └── task_graphs/            # 24 standard recipe task graphs
├── data/
│   ├── matched_features_gt/     # GT matched pairs
│   ├── matched_features_actionformer_person/
│   ├── matched_features_actionformer_recordings/
│   ├── matched_features_hiero/
│   ├── fusion_model_gt/         # GT fusion model
│   ├── fusion_model_actionformer_person/
│   ├── fusion_model_actionformer_recordings/
│   ├── fusion_model_hiero/
│   ├── task_graph_encodings_256/ # Shared task graph encodings
│   ├── processed_graphs_gt.pt    # Extracted PyG graphs
│   ├── processed_graphs_actionformer_person.pt
│   ├── processed_graphs_actionformer_recordings.pt
│   └── processed_graphs_hiero.pt
├── models/                      # GNN model implementations (DAGNN, GCN)
├── configs/                   # Configuration files
├── results/                    # Training results and checkpoints
│   ├── checkpoints/            # Saved models
│   ├── experiment_results.csv  # Standard split results
│   ├── loo_results.csv         # Per-recipe metrics (LOO)
├── check_fusion_quality.py     # Inspect fusion outputs
├── train.py                    # Training (standard + LOO)
├── train_kfold_standard.py     # Standalone K-fold training script
└── run_experiments.py          # Full grid search
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric

### Installation

#### Option 1: Local Installation (macOS/Linux/Windows - CPU)

**Step 1: Clone Repository**
```bash
git clone --recursive https://github.com/storylei/task-graph-classification.git
cd task-graph-classification

# Initialize annotations submodule (if not cloned with --recursive)
git submodule init
git submodule update

# Verify annotations (should see 24 recipe JSON files)
ls annotations/task_graphs/
```

**Step 2: Create and Activate a Conda Environment**
```bash
conda create -n taskgraph python=3.8 -y
conda activate taskgraph
```

**Step 3: Install Dependencies**
```bash
# PyTorch and core scientific packages
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y -c conda-forge numpy pandas scipy scikit-learn networkx matplotlib seaborn tensorboard tqdm pyyaml jupyter

# PyTorch Geometric
python -m pip install torch-geometric
```

**Step 4: Verify Installation**
```bash
python -c "import torch; import torch_geometric; print(torch.__version__); print(torch_geometric.__version__)"
```

#### Option 2: Google Colab Installation (CUDA)

**Step 1: Clone Repository**
```python
!git clone --recursive https://github.com/storylei/task-graph-classification.git
%cd task-graph-classification

# Initialize submodule (if needed)
!git submodule init && git submodule update
```

**Step 2: Install Dependencies**
```python
# Core packages
!pip install -q torch numpy pandas scipy scikit-learn networkx matplotlib seaborn tensorboard tqdm pyyaml jupyter

# PyTorch Geometric (includes CUDA support automatically)
!pip install -q torch_geometric
```

**Step 3: Verify Installation**
```python
import torch
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA: {torch.version.cuda}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

import torch_geometric
print(f"✓ PyTorch Geometric: {torch_geometric.__version__}")
```

**Step 4: Mount Google Drive (Optional)**
```python
from google.colab import drive
drive.mount('/content/drive')

import os

# Copy Substep 3 outputs from Google Drive

DRIVE_PATH = '/content/drive/MyDrive/AMLproject/extension3'
# Copy real data from Google Drive (if available)
if os.path.exists(DRIVE_PATH):
    print("Found real data in Google Drive, copying...")

    # Ensure destination directory for user's request exists
    os.makedirs('data/extension3', exist_ok=True)
    !cp -r {DRIVE_PATH}/* data/extension3/
    print("Files copied to data/extension3/")
else:
    print("No real data found")
```

## Data

### Input Data Sources (Extension 3 Outputs)

The `extract_graphs.py` script processes four Extension 3 variants to create PyG Data objects. In the current workspace layout, the type-specific files live directly under `data/` as sibling folders; only `task_graph_encodings_256` is shared across all variants.

#### 1. **Type-specific matched pairs**
Each variant has its own matched pair file:
- `data/matched_features_gt/matched_pairs.json`
- `data/matched_features_actionformer_person/matched_pairs.json`
- `data/matched_features_actionformer_recordings/matched_pairs.json`
- `data/matched_features_hiero/matched_pairs.json`

These files contain the matched visual-task pairs for each recording and determine which graph type is being extracted.

#### 2. **Type-specific fusion model**
Each variant also has its own fusion checkpoint:
- `data/fusion_model_gt/best_fusion_model.pth`
- `data/fusion_model_actionformer_person/best_fusion_model.pth`
- `data/fusion_model_actionformer_recordings/best_fusion_model.pth`
- `data/fusion_model_hiero/best_fusion_model.pth`

#### 3. **Type-independent task graph encodings**
Shared across all four variants:
- `data/task_graph_encodings_256/task_graph_embeddings.npz`
- `data/task_graph_encodings_256/task_graph_metadata.json`

`task_graph_embeddings.npz` stores the recipe-step text embeddings, and `task_graph_metadata.json` stores the recipe structures and edges. These files do not change across GT, ActionFormer-Person, ActionFormer-Recordings, or HiERO.

The fusion model is used to combine the matched visual and task embeddings for each variant.

### 4. **Extracted graph output**
The resulting PyG dataset can be saved separately for each variant, for example:
- `data/processed_graphs_gt.pt`
- `data/processed_graphs_actionformer_person.pt`
- `data/processed_graphs_actionformer_recordings.pt`
- `data/processed_graphs_hiero.pt`

## Usage

### Step 0: Evaluation Strategy

Use the following protocols depending on your goal:

- **`standard`**: Single 80/20 split for quick model iteration and ablation checks
- **`loo`**: Leave-One-Recipe-Out (24 folds), recommended for final reporting
- **`kfold` (optional)**: Stratified K-fold on standard protocol for robustness checks
- **`grid_search` (optional)**: Hyperparameter search before final runs (see Step 4)

#### Recommended Workflow

1. Run `standard` mode to verify training settings and model behavior quickly.
2. Run optional `grid_search` to select hyperparameters.
3. Run optional `kfold` to check stability across random splits.
4. Run `loo` with selected hyperparameters for final per-recipe and average metrics.

#### Notes

- Use the same `--data_path` variant consistently within one experiment set.
- Do hyperparameter selection (`grid_search`) on training/validation logic only; avoid tuning on final LOO test folds.
- `evaluate.py` uses dataset samples from `--data_path`; checkpoints only provide model weights/config.

### Step 1: Extract Graphs

#### Data Processing Pipeline

The `extract_graphs.py` script performs the following steps for each recording:

```
1. Load its matched pairs from the type-specific `matched_pairs.json`
2. Load shared task graph embeddings and recipe metadata
3. Fuse each matched task-step / visual-step pair using the type-specific fusion model
4. Build one PyG graph per recording with:
   - Node features: [256-dim fused embedding]
   - Edges: from recipe metadata
   - Label: correctness label (1=correct, 0=incorrect)
```

#### Command

Convert Extension 3 outputs to PyTorch Geometric Data objects:

```bash
# Basic usage (uses default paths)
python extract_graphs.py

# Custom paths
python extract_graphs.py \
  --matched_pairs data/matched_features_gt/matched_pairs.json \
  --task_embeddings data/task_graph_encodings_256/task_graph_embeddings.npz \
  --metadata data/task_graph_encodings_256/task_graph_metadata.json \
  --fusion_checkpoint data/fusion_model_gt/best_fusion_model.pth \
  --output data/processed_graphs_gt.pt \
  --device cuda  # or 'cpu'
```

**Output**: `data/processed_graphs_gt.pt` (384 graphs ready for training)

To build the other three variants, replace the type-specific `--matched_pairs`, `--fusion_checkpoint`, and `--output` paths with the corresponding `actionformer_person`, `actionformer_recordings`, or `hiero` directories.

### Step 2: Train Model

#### Key Parameters

- **`--data_path`**: Path to processed graph file (default: `data/processed_graphs.pt`)
- **`--eval_mode`**: Evaluation protocol - `standard` (80/20 split) or `loo` (Leave-One-Recipe-Out, default: `standard`)
- **`--model_type`**: Model architecture - `dagnn` (recommended) or `gcn` (default: `dagnn`)
- **`--hidden_dim`**: Hidden dimension size (default: 128)
- **`--num_layers`**: Number of GNN layers (default: 3)
- **`--dropout`**: Dropout rate (default: 0.3)
- **`--batch_size`**: Batch size (default: 8)
- **`--num_epochs`**: Maximum number of epochs (default: 100)
- **`--lr`**: Learning rate (default: 0.001)
- **`--patience`**: Early stopping patience (default: 20 epochs)
- **`--device`**: Compute device - `cuda` or `cpu` (default: `cuda` if available)
- **`--use_wandb`**: Enable Weights & Biases logging (optional)

#### Basic Examples

```bash
# Standard train/val split (80/20) with DAGNN
python train.py \
  --data_path data/processed_graphs_gt.pt \
  --eval_mode standard \
  --model_type dagnn \
  --device cuda

# Leave-One-Recipe-Out (LOO) cross-validation with GCN
python train.py \
  --data_path data/processed_graphs_gt.pt \
  --eval_mode loo \
  --model_type gcn \
  --device cuda
```

#### Advanced Examples

```bash
# Custom hyperparameters for standard mode
python train.py \
  --data_path data/processed_graphs_gt.pt \
  --eval_mode standard \
  --model_type dagnn \
  --hidden_dim 256 \
  --num_layers 4 \
  --dropout 0.5 \
  --batch_size 16 \
  --lr 0.0005 \
  --num_epochs 150 \
  --patience 30 \
  --device cuda

# LOO with fixed top hyperparameters from grid search
python train.py \
  --data_path data/processed_graphs_gt.pt \
  --eval_mode loo \
  --model_type dagnn \
  --hidden_dim 128 \
  --num_layers 3 \
  --batch_size 8 \
  --lr 0.001 \
  --device cuda

#### Output Files

- **Standard mode**: `results/checkpoints/best_model.pt`, `results/experiment_results.csv`
- **LOO mode**: Per-recipe checkpoints in `results/checkpoints/loo/`, results in `results/loo_results.csv`

### Step 3: K-Fold Cross-Validation (Optional - Standard Mode Only)

For additional robustness validation, use stratified K-fold cross-validation on the standard train/val protocol. This runs multiple independent training runs with different train/test splits.

#### Key Parameters

- **`--data_path`**: Path to processed graph file (default: `data/processed_graphs.pt`)
- **`--k_folds`**: Number of folds (default: 5)
- **`--val_fraction`**: Validation fraction within each fold's training set (default: 0.2)
- **`--model_type`**: Model architecture - `dagnn` (recommended) or `gcn` (default: `dagnn`)
- **`--hidden_dim`**: Hidden dimension size (default: 128)
- **`--num_layers`**: Number of GNN layers (default: 3)
- **`--dropout`**: Dropout rate (default: 0.3)
- **`--batch_size`**: Batch size (default: 8)
- **`--num_epochs`**: Maximum number of epochs (default: 100)
- **`--lr`**: Learning rate (default: 0.001)
- **`--patience`**: Early stopping patience (default: 20 epochs)
- **`--seed`**: Random seed for reproducibility (default: 42)
- **`--device`**: Compute device - `cuda` or `cpu` (default: `cuda` if available)

#### Basic Examples

```bash
# 5-fold cross-validation with DAGNN (default)
python train_kfold_standard.py \
  --data_path data/processed_graphs_gt.pt \
  --k_folds 5

# 10-fold cross-validation with GCN
python train_kfold_standard.py \
  --data_path data/processed_graphs_gt.pt \
  --model_type gcn \
  --k_folds 10 \
  --device cuda
```

#### Advanced Examples

```bash
# Custom hyperparameters with 5-fold validation
python train_kfold_standard.py \
  --data_path data/processed_graphs_gt.pt \
  --model_type dagnn \
  --hidden_dim 256 \
  --num_layers 4 \
  --dropout 0.5 \
  --batch_size 16 \
  --lr 0.0005 \
  --num_epochs 150 \
  --patience 30 \
  --k_folds 5 \
  --device cuda

# Top hyperparameters from grid search with leave-one-out validation
python train_kfold_standard.py \
  --data_path data/processed_graphs_gt.pt \
  --model_type dagnn \
  --hidden_dim 128 \
  --num_layers 3 \
  --batch_size 8 \
  --lr 0.001 \
  --k_folds 10 \
  --device cuda

#### Output Files

- **Per-fold checkpoints**: `results/checkpoints/kfold_standard/{dataset_tag}/fold_{N}_best.pt` (one per fold)
- **Aggregated results**: `results/kfold_standard_{dataset_tag}.csv`
  - Columns: Fold, Model, Hidden_Dim, Layers, LR, Dropout, Train/Val/Test sizes, Best_Val_F1, Best_Val_Acc, Best_Val_AUC, Test_Accuracy, Test_F1, Test_AUC, Test_Precision, Test_Recall, Epochs, Duration_Sec
  - Last row: Average metrics across all folds

#### Resume from Checkpoint

K-fold training supports resuming from incomplete runs:

```bash
# If interrupted, re-run the same command to resume
python train_kfold_standard.py \
  --data_path data/processed_graphs_gt.pt \
  --k_folds 5
```

The script checks `results/kfold_standard_{dataset_tag}.csv` for completed folds and skips them automatically.

### Step 4: Grid Search

Run comprehensive hyperparameter grid search over multiple model configurations. This systematically explores the hyperparameter space and outputs top-performing configurations ranked by their evaluation metrics.

#### Key Parameters

- **`--data_path`**: Path to processed graph file (default: `data/processed_graphs.pt`)
- **`--grid_config`**: Grid search configuration file (JSON with param ranges, optional)
- **`--device`**: Compute device - `cuda` or `cpu` (default: `cuda` if available)
- **`--seed`**: Random seed for reproducibility (default: 42)
- **`--use_wandb`**: Enable Weights & Biases logging (optional)

#### Basic Examples

```bash
# Default grid search (scans predefined hyperparameter ranges)
python run_experiments.py \
  --data_path data/processed_graphs_gt.pt

# Grid search with custom GPU device
python run_experiments.py \
  --data_path data/processed_graphs_gt.pt \
  --device cuda
```

#### Advanced Examples

```bash
# Grid search with Weights & Biases logging
python run_experiments.py \
  --data_path data/processed_graphs_gt.pt \
  --use_wandb \
  --wandb_project task-graph-classification \
  --wandb_run_name grid_search_gt

# Grid search on multiple datasets sequentially
for dataset in gt actionformer_person actionformer_recordings hiero; do
  python run_experiments.py \
    --data_path data/processed_graphs_${dataset}.pt \
    --seed 42 \
    --device cuda
done

# Run grid search, then use top config for final validation
python run_experiments.py \
  --data_path data/processed_graphs_gt.pt \
  --device cuda
# Check results/loo_results_grid.csv for top 5 configurations
# Then run: python train.py --data_path data/processed_graphs_gt.pt --eval_mode loo <top_hyperparams>
```

#### Output Files

- **Grid search results**: `results/loo_results_grid.csv`
  - Columns: ModelType, HiddenDim, NumLayers, LR, Dropout, BatchSize, Accuracy, F1, AUC, Precision, Recall, Epochs, Duration_Sec
  - Sorted by F1 score (best first)
  - Last section: Top 5 best configurations printed to console
  
- **Experiment log**: `results/experiment_results.csv` (appended with each run)

### Step 5: Evaluate Model

Evaluate saved model checkpoints on test data. The evaluation loads the trained model weights from a checkpoint and runs inference on the test/validation split from the dataset file (e.g., `processed_graphs_gt.pt`).

#### Key Points

- **Data Source**: Test data is loaded from the dataset file (`--data_path`), not from the checkpoint
  - **LOO mode**: Uses all samples from each held-out recipe
  - **Standard mode**: Uses the 20% validation split (with same random seed as training)
- **Checkpoint**: Contains only model weights (`model_state_dict`) and configuration (`model_config`)
- **Output**: Metrics (Accuracy, F1, AUC, Precision, Recall) and confusion matrix visualization

#### Basic Examples

```bash
# Evaluate all LOO checkpoints (24 models, one per recipe)
python evaluate.py \
  --data_path data/processed_graphs_gt.pt \
  --eval_loo \
  --device cuda

# Evaluate a single standard checkpoint on validation split
python evaluate.py \
  --data_path data/processed_graphs_gt.pt \
  --checkpoint results/checkpoints/best_model.pt \
  --device cuda
```

#### Advanced Examples

```bash
# Evaluate a specific recipe's LOO checkpoint
python evaluate.py \
  --data_path data/processed_graphs_gt.pt \
  --checkpoint results/checkpoints/loo/coffee_best.pt \
  --device cuda

# Evaluate on a different dataset variant
python evaluate.py \
  --data_path data/processed_graphs_actionformer_person.pt \
  --eval_loo \
  --device cuda

# Batch evaluate all variants (for comparison)
for variant in gt actionformer_person actionformer_recordings hiero; do
  python evaluate.py \
    --data_path data/processed_graphs_${variant}.pt \
    --eval_loo \
    --device cuda
done
```

#### Output Files

- **Confusion matrix**: `results/evaluation/confusion_matrix.png`
- **Console output**: Per-recipe or overall metrics with F1, AUC, Precision, Recall

## Citation

```bibtex
@inproceedings{captaincook4d,
  title={CaptainCook4D: Understanding Errors in Procedural Activities},
  author={...},
  booktitle={...},
  year={2023}
}
```
## License

MIT License

## Contact

For questions, please open an issue on GitHub.


