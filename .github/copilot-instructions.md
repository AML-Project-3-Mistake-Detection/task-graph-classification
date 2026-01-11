# Task Graph Classification - AI Coding Instructions

## Project Overview
Graph Neural Network (GNN) classifier for verifying recipe execution correctness by analyzing observed task-graph structures. **Part of CaptainCook4D Mistake Detection (Substep 4)**. Consumes outputs from Substep 3 (observed graphs from video analysis) and standard recipe task graphs.

## Architecture Pattern
**Data Flow:** Standard recipe graphs (annotations/) + Observed execution graphs (data/observed_graphs/) → PyG Dataset → GNN models (DAGNN/GCN) → Binary classification (correct/incorrect execution)

Key components:
- **[utils/data_loader.py](utils/data_loader.py)**: `TaskGraphDataset` loads both standard and observed graphs, extracts node features (one-hot encoding by default, supports embeddings), converts to PyTorch Geometric `Data` objects
- **[models/dagnn.py](models/dagnn.py)**: `DAGNN` (custom directed acyclic graph conv) and `GCNClassifier` implementations with global pooling and MLP classifier
- **[configs/config.py](configs/config.py)**: Centralized configuration class - all paths, hyperparameters, and settings in one place
- **[train.py](train.py)**: Training loop with train/val split (TODO: leave-one-out cross-validation for 24 recipes)

## Critical Data Conventions
1. **Observed graph format** (Substep 3 output in `data/observed_graphs/*.json`):
   ```json
   {
     "recipe_id": "coffee",
     "video_id": "video_001", 
     "observed_steps": [1, 2, 3, ...],  // Step IDs without START/END
     "edges": [[1, 2], [2, 3], ...],    // Integer edge list
     "label": 1  // 1=correct, 0=incorrect
   }
   ```

2. **Standard task graphs** (annotations/task_graphs/*.json):
   - Contains `"steps"` dict with step descriptions
   - Includes START (0) and END nodes
   - Standard recipes: 24 total from CaptainCook4D annotations (git submodule)

3. **Node features**: `TaskGraphDataset._extract_node_features()` supports:
   - `one_hot`: One-hot encoding based on `max_steps` (default, currently used)
   - `text_embeddings`: From Substep 3 if provided in JSON (`step_embeddings` key)
   - `learned`: Embedding layer (implementation TODO)

## Development Workflows

### Training from scratch:
```bash
python train.py --hidden_dim 128 --num_epochs 100
# Or modify configs/config.py and run:
python train.py
```

### Mock data generation (for testing without Substep 3):
```bash
python generate_mock_data.py
# Creates synthetic correct/incorrect graphs in data/observed_graphs/
# Error types: missing_edge, extra_edge, wrong_order, missing_step
```

### Google Colab workflow:
- Mount Drive → set `Config.DRIVE_OBSERVED_GRAPHS_DIR` 
- Clone with `--recursive` flag to get annotations submodule
- Install requirements via pip

## Project-Specific Patterns

1. **Path Resolution**: Always use `Path.resolve()` for data paths ([train.py](train.py#L90-L92)) - handles relative paths and Colab environments consistently

2. **Config Override Pattern**: Command-line args override `Config` class defaults via `Config.from_args()` - only non-None args applied ([configs/config.py](configs/config.py#L60-L66))

3. **Graph Feature Dimensions**: Dataset calculates `max_steps` across all standard recipes at init to ensure consistent feature dimensions ([utils/data_loader.py](utils/data_loader.py#L44-L49))

4. **Model Selection**: Factory pattern in [train.py](train.py#L140-L160) - `config.MODEL_TYPE` switches between DAGNN/GCN implementations

5. **Checkpoint Saving**: Best model saved to `results/checkpoints/best_model.pt` with full state dict, config, and metrics ([train.py](train.py#L193-L202))

## Integration Points

- **Annotations**: Git submodule at `annotations/` pointing to CaptainCook4D task graphs - update with `git submodule update --remote`
- **Substep 3 Interface**: Expects JSON files in `data/observed_graphs/` with specific schema (see Data Conventions)
- **Future Integration**: Evaluation script `evaluate.py` marked as TODO - should implement leave-one-out cross-validation

## Key Gotchas

1. **Empty Dataset**: If dataset size is 0, check paths exist and `observed_graphs/` has JSON files - [train.py](train.py#L113-L116) logs diagnostic info
2. **Annotations Submodule**: Clone with `--recursive` or run `git submodule init && git submodule update` - required for standard task graphs
3. **Labels Format**: Must be scalar int (0/1) in JSON, not tensor - converted to `torch.long` in `TaskGraphDataset.get()` ([utils/data_loader.py](utils/data_loader.py#L130))
4. **Node Feature Consistency**: All graphs use same `max_steps` for feature dimension, calculated at dataset initialization

## Common Tasks

**Add new model architecture**: Create class in [models/dagnn.py](models/dagnn.py), add choice to [train.py](train.py#L140) factory, update `MODEL_TYPE` choices in [configs/config.py](configs/config.py#L28)

**Change node features**: Modify `TaskGraphDataset._extract_node_features()` or set `Config.NODE_FEATURE_TYPE` (current: one_hot)

**Implement LOO cross-validation**: Replace simple train/val split in [train.py](train.py#L119-L124) with 24-fold LOO based on recipe_id

**Debug data loading**: Check `annotations_dir.exists()` and `observed_graphs_dir.exists()` outputs, verify JSON format matches expected schema
