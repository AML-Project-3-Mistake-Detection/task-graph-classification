# Task Graph Classification

Graph Neural Network based classifier for verifying recipe execution correctness.

**Part of:** CaptainCook4D - Mistake Detection Project (Substep 4)

## Overview

This project implements a GNN-based classifier to predict whether a recipe execution is correct by analyzing the observed task-graph structure from video analysis.

## Project Structure

```
task-graph-classification/
├── annotations/                # CaptainCook4D annotations (submodule)
│   └── task_graphs/           # 24 standard recipe task graphs
├── data/
│   ├── extension3_outputs/    # Visual features, task graph encodings, fusion model
│   └── processed_graphs.pt    # Extracted PyG graphs (384 videos)
├── models/                    # GNN model implementations (DAGNN, GCN)
├── configs/                   # Configuration files
├── results/                   # Training results and checkpoints
│   ├── checkpoints/           # Saved models (standard + LOO)
│   ├── loo_results.csv        # Per-recipe metrics (LOO)
│   └── loo_results_grid.csv   # Grid search summary (optional)
├── extract_graphs.py          # Build graphs from Extension 3 outputs
├── train.py                   # Training (standard + LOO)
├── evaluate.py                # Evaluation of saved checkpoints
└── run_experiments.py         # Full grid search
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

**Step 2: Install Dependencies**
```bash
# Core packages
pip install torch numpy pandas scipy scikit-learn networkx matplotlib seaborn tensorboard tqdm pyyaml jupyter

# PyTorch Geometric
pip install torch_geometric
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
```


## Data

### Input Data Sources (Extension 3 Outputs)

The `extract_graphs.py` script processes outputs from Extension 3 (Fusion & Matching) to create 384 PyG Data objects:

#### 1. **Visual Features** (`data/extension3_outputs/visual_features/`)
   - `hiero_step_embeddings_256.npz`: Hierarchical visual embeddings for all 384 videos
     - Shape: (384, 61, 256) = [num_videos, max_steps, embedding_dim]
     - Contains step-level visual features extracted from video frames
     - Includes step_mask indicating valid/invalid steps
   - `visual_features_mapping.json`: Video-to-recipe mapping
     - Maps video indices to recipe names (e.g., `{"0": "pinwheels", "1": "coffee", ...}`)

#### 2. **Task Graph Encodings** (`data/extension3_outputs/task_graph_encodings/`)
   - `task_graph_embeddings.npz`: Text embeddings for 24 standard recipes
     - Contains one array per recipe (e.g., `npz['coffee']` has shape [N, 256])
     - N = number of standard steps for that recipe
   - `task_graph_metadata.json`: Recipe structure metadata
     - Contains step descriptions and graph edges for each recipe
     - Example: `{"coffee": {"steps": {...}, "edges": [[0,1], [1,2], ...]}}`

#### 3. **Fusion Model** (`data/extension3_outputs/fusion_model/`)
   - `best_fusion_model.pth`: Trained fusion model checkpoint
     - Fuses matched visual + task embeddings
     - Supports multiple fusion types: concat, cross_attention, gated

### Data Processing Pipeline

```
For each video (384 total):
  1. Load visual embeddings (variable number of valid steps)
  2. Get task embeddings for matched recipe (fixed number of standard steps)
  3. Hungarian matching: align visual steps to task steps
  4. For each standard task step:
     - If matched: fuse visual + task embedding using fusion model
     - If unmatched: use zero-filled embedding (missing step indicator)
  5. Construct graph with:
     - Node features: [256-dim fused embeddings + time encoding + presence mask]
     - Edges: from recipe metadata
     - Label: correctness label (1=correct, 0=incorrect)
  
Output: 384 PyG Data objects saved to data/processed_graphs.pt
```

### Feature Dimensions

Each node has **258 dimensions**:
- 256: Fused embedding (visual + task)
- 1: Normalized time encoding (step_position / total_steps)
- 1: Presence indicator (1 if matched, 0 if missing)

## Usage

### Step 1: Extract Graphs

Convert Extension 3 outputs to PyTorch Geometric Data objects:

```bash
# Basic usage (uses default paths)
python extract_graphs.py

# Custom paths
python extract_graphs.py \
    --matched_pairs data/extension3_outputs/matched_features/matched_pairs.json \
    --task_embeddings data/extension3_outputs/task_graph_encodings/task_graph_embeddings.npz \
    --metadata data/extension3_outputs/task_graph_encodings/task_graph_metadata.json \
    --fusion_checkpoint data/extension3_outputs/fusion_model/best_fusion_model.pth \
    --output data/processed_graphs.pt \
    --device cuda  # or 'cpu'
```

**Output**: `data/processed_graphs.pt` (384 graphs ready for training)

### Step 2: Train Model

```bash
# Leave-One-Recipe-Out (LOO) cross-validation
python train.py --eval_mode loo --model_type dagnn --num_epochs 50 --device cuda

# Standard train/val split (80/20)
python train.py --eval_mode standard --model_type dagnn --num_epochs 100 --device cuda

# With custom hyperparameters
python train.py \
    --eval_mode loo \
    --model_type dagnn \
    --hidden_dim 256 \
    --num_layers 4 \
    --batch_size 16 \
    --lr 0.0005 \
    --num_epochs 100
```

**Output**: 
- Per-recipe results: `results/loo_results.csv`
- Per-recipe checkpoints: `results/checkpoints/loo/{recipe_name}_best.pt` (24 files)
- Standard mode checkpoint: `results/checkpoints/best_model.pt` (if using `--eval_mode standard`)

### Step 3: Evaluate Model

Evaluate saved model checkpoints:

```bash
# Evaluate all LOO checkpoints (24 models)
python evaluate.py --eval_loo --device cuda

# Evaluate a single checkpoint (standard mode)
python evaluate.py --checkpoint results/checkpoints/best_model.pt --device cuda

# Evaluate a specific recipe's LOO checkpoint
python evaluate.py --checkpoint results/checkpoints/loo/coffee_best.pt --device cuda
```

**Output**: Confusion matrix saved to `results/evaluation/confusion_matrix.png`

### Step 4: Grid Search (Optional)

Run hyperparameter grid search over multiple configurations:

```bash
# This will train multiple models with different hyperparameters using LOO cross-validation
python run_experiments.py
```

**Output**: 
- Aggregated results: `results/loo_results_grid.csv`
- Top 5 best configurations printed at the end

## Evaluation Strategy

**Leave-One-Recipe-Out (LOO) Cross-Validation:**
- Use all samples from one recipe as the test set
- Train on the remaining 23 recipes
- Repeat for all 24 recipes
- Report per-recipe and average metrics (Accuracy, F1, AUC, Precision, Recall)

**Standard Train/Val Split:**
- 80% training, 20% validation
- Single train/val run with early stopping

## Model Architecture

- **DAGNN** (Directed Acyclic Graph Neural Network) - Recommended
- **GCN** (Graph Convolutional Network)

## Results

Results and checkpoints are saved in `results/` directory:

### LOO Mode (Leave-One-Recipe-Out)

- **`loo_results.csv`**: Per-recipe metrics from LOO cross-validation
  - Columns: Recipe, Accuracy, F1, AUC, Precision, Recall
  - Last row: Average across all 24 recipes
  
- **`checkpoints/loo/{recipe_name}_best.pt`**: Model checkpoint for each recipe (24 files)
  - Each checkpoint contains: model_state_dict, test metrics, model config
  
- **`loo_results_grid.csv`**: Grid search results (if using `run_experiments.py`)
  - Compares different hyperparameter configurations

### Standard Mode (80/20 Split)
  
- **`experiment_results.csv`**: Training log for standard train/val splits
  - Columns: Timestamp, ModelType, HiddenDim, NumLayers, LR, Dropout, BatchSize, BestAcc, BestF1, BestAUC, Epoch

- **`checkpoints/best_model.pt`**: Best model state dict from training

### Evaluation Outputs

- **`evaluation/confusion_matrix.png`**: Confusion matrix visualization

### Example LOO Results

```
Recipe,Accuracy,F1,AUC,Precision,Recall
blenderbananapancakes,0.6316,0.3871,0.3095,0.3158,0.5000
coffee,0.4667,0.3182,0.5625,0.2333,0.5000
zoodles,0.8667,0.8295,0.8409,0.8295,0.8295
...
Average,0.5140,0.4333,0.5024,0.4488,0.5048
```

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


