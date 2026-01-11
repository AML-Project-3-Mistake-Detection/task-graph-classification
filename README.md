# Task Graph Classification

Graph Neural Network based classifier for verifying recipe execution correctness.

**Part of:** CaptainCook4D - Mistake Detection Project (Substep 4)

## Overview

This project implements a GNN-based classifier to predict whether a recipe execution is correct by analyzing the observed task-graph structure from video analysis.

## Project Structure

```
task-graph-classification/
├── annotations/          # Git submodule (CaptainCook4D annotations)
│   └── task_graphs/     # 24 standard recipe task graphs
├── data/
│   ├── observed_graphs/ # Substep 3 outputs (from Google Drive)
│   └── processed/       # Processed PyG data
├── models/              # GNN model implementations (DAGNN, GCN)
├── utils/               # Data loaders and graph utilities
├── configs/             # Configuration management
├── results/             # Training results and checkpoints
├── train.py            # Training script
├── evaluate.py         # Evaluation script (TODO)
└── notebooks/          # Jupyter notebooks for experiments
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric

### Installation

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/storylei/task-graph-classification.git
cd task-graph-classification

# Initialize annotations submodule (if not cloned with --recursive)
git submodule init
git submodule update

# Verify annotations (should see 24 recipe JSON files)
ls annotations/task_graphs/

# Install dependencies - Step 1: Core packages
pip install torch numpy pandas scipy scikit-learn networkx matplotlib seaborn tensorboard tqdm pyyaml jupyter

# Install dependencies - Step 2: PyTorch Geometric
pip install torch-geometric

# Create data directories
mkdir -p data/observed_graphs data/processed results/checkpoints
```

**Why two-step installation?** PyTorch Geometric extensions (`torch-scatter`, `torch-sparse`) require compilation and need PyTorch to be installed first. The core `torch-geometric` package is sufficient for this project.

### For Google Colab

```python
# Clone with submodules
!git clone --recursive https://github.com/storylei/task-graph-classification.git
%cd task-graph-classification

# Initialize submodule (if needed)
!git submodule init && git submodule update

# Install dependencies - Step 1: Core packages
!pip install torch numpy pandas scipy scikit-learn networkx matplotlib seaborn tensorboard tqdm pyyaml jupyter

# Install dependencies - Step 2: PyTorch Geometric
!pip install torch-geometric

# Mount Google Drive for Substep 3 outputs
from google.colab import drive
drive.mount('/content/drive')
```

## Data

### Input Data Sources

1. **Standard Task Graphs** (from `annotations/task_graphs/`)
2. **Observed Graphs** (from Google Drive → `data/observed_graphs/`)
   - Output from Substep 3: Observed task graphs from video analysis
   - Contains: Matched step sequences, graph structures, and labels
   - Binary labels: 1=correct execution, 0=incorrect execution
   - Observed task graphs from video analysis
   - Matched step sequences and graph structures
   - Binary labels (correct/incorrect execution)

### Data Format

Expected input format from Substep 3:
```json
{
  "recipe_id": "coffee",
  "video_id": "video_001",
  "observed_steps": [0, 1, 2, 3, ...],
  "matched_edges": [[0, 1], [1, 2], ...],
  "label": 1
}
```

## Usage

### Training

```bash
python train.py --config configs/dagnn_config.yaml
```

### Evaluation

```bash
python evaluate.py --checkpoint results/checkpoints/best_model.pt
```

### In Colab

```python
# Train with data from Google Drive
!python train.py \
    --observed_graphs_dir /content/drive/MyDrive/AML_Project/substep3_outputs \
    --model_type dagnn \
    --num_epochs 50
```

## Model Architecture

- **DAGNN** (Directed Acyclic Graph Neural Network) - Recommended
- **GCN** (Graph Convolutional Network)
- **GraphSAGE**

## Evaluation Strategy

Leave-One-Out (LOO) validation:
- Train on (k-1) recipes
- Test on the k-th recipe
- Repeat for all 24 recipes

## Results

Results and checkpoints will be saved in `results/` directory.

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

For questions, please contact: [Your Email]
