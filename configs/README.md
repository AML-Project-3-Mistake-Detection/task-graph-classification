# Configuration Guide

## What is `configs/config.py`?

The `config.py` file is a centralized place to manage all project settings. Instead of typing parameters every time you run the code, you can set them once in this file.

## Key Settings Explained

### 📂 **Paths** (Most Important!)

```python
ANNOTATIONS_DIR = "annotations/task_graphs"     # Standard recipe graphs (24 recipes)
OBSERVED_GRAPHS_DIR = "data/observed_graphs"    # YOUR Substep 3 outputs go here
RESULTS_DIR = "results"                          # Where models are saved
```

**For Colab users:** Set this to your Google Drive path:
```python
DRIVE_OBSERVED_GRAPHS_DIR = "/content/drive/MyDrive/AML_Project/substep3_outputs"
```

---

### 🧠 **Model Settings**

```python
MODEL_TYPE = "dagnn"      # Which GNN to use (dagnn, gcn, graphsage)
HIDDEN_DIM = 128          # Size of hidden layers (bigger = more powerful, but slower)
NUM_LAYERS = 3            # Number of GNN layers (deeper = more expressive)
DROPOUT = 0.3             # Regularization (prevents overfitting)
```

**When to change:**
- Try `HIDDEN_DIM = 256` if model underfits
- Try `NUM_LAYERS = 4` for more complex graphs
- Increase `DROPOUT = 0.5` if overfitting

---

### 📊 **Training Settings**

```python
BATCH_SIZE = 8            # How many graphs to process at once
NUM_EPOCHS = 100          # How many times to go through the data
LEARNING_RATE = 0.001     # How fast the model learns
```

**When to change:**
- Reduce `BATCH_SIZE` if GPU memory runs out
- Increase `NUM_EPOCHS = 200` if not converged
- Decrease `LEARNING_RATE = 0.0001` if training unstable

---

### 🎯 **Node Features**

```python
NODE_FEATURE_TYPE = "text_embedding"  # How to represent steps
EMBEDDING_DIM = 768                    # Size of step embeddings
```

**Options:**
1. `"one_hot"`: Simple binary encoding (faster, less powerful)
2. `"text_embedding"`: Use BERT/EgoVLP embeddings (better, requires Substep 3)
3. `"learned"`: Let model learn features (experimental)

---

## How to Use Config

### Method 1: Edit config.py directly (Recommended)

```python
# In configs/config.py
HIDDEN_DIM = 256
NUM_EPOCHS = 200
```

Then just run:
```bash
python train.py
```

### Method 2: Override from command line

```bash
python train.py --hidden_dim 256 --num_epochs 200 --lr 0.0001
```

### Method 3: In Colab

```python
# Override Google Drive path
!python train.py \
    --observed_graphs_dir /content/drive/MyDrive/AML_Project/substep3_outputs \
    --num_epochs 200 \
    --device cuda
```

---

## Quick Start Checklist

- [ ] Put Substep 3 outputs in `data/observed_graphs/`
- [ ] Check paths in `configs/config.py`
- [ ] Adjust `BATCH_SIZE` based on your GPU
- [ ] Set `NUM_EPOCHS` (start with 50 for testing)
- [ ] Run `python train.py`

---

## Example Configurations

### 🏃 **Fast Testing**
```python
BATCH_SIZE = 16
NUM_EPOCHS = 10
HIDDEN_DIM = 64
NUM_LAYERS = 2
```

### 🎯 **Best Performance**
```python
BATCH_SIZE = 4
NUM_EPOCHS = 200
HIDDEN_DIM = 256
NUM_LAYERS = 4
DROPOUT = 0.4
```

### 💻 **CPU Only**
```python
DEVICE = "cpu"
BATCH_SIZE = 1
HIDDEN_DIM = 64
```

---

## Common Issues

**Q: "No data loaded"**
→ Check `OBSERVED_GRAPHS_DIR` path is correct

**Q: "CUDA out of memory"**
→ Reduce `BATCH_SIZE` or `HIDDEN_DIM`

**Q: "Model not improving"**
→ Try increasing `NUM_EPOCHS` or `LEARNING_RATE`

---

## Need Help?

The config file has comments explaining each parameter. If unsure, start with the defaults!
