# Data Available for Substep 4: GNN Classification

## Overview
This document describes the data produced by Substeps 1-3 that will be used in Substep 4 for training a GNN-based classifier to predict correct vs. incorrect task execution.

---

## Key Files for Substep 4

### 1. **matched_pairs.json** 
**Location:** `outputs/matched_features/matched_pairs.json`

**Description:** Contains 358 matched pairs of (task_graph_node, visual_step) with recording metadata.

**Structure per pair:**
```json
{
  "task_embedding": [256 floats],          // Task graph node embedding
  "visual_embedding": [256 floats],        // Visual step embedding  
  "task_name": "blenderbananapancakes",    // Recipe name
  "task_idx": 0,                           // Index in task graph
  "visual_idx": 22,                        // Index in visual features (task-specific)
  "recording_id": "21_17",                 // ⭐ Video recording ID
  "video_label": 0,                        // ⭐ 0=correct, 1=incorrect
  "video_idx": 146,                        // Global video index
  "step_idx_in_video": 2,                  // Step position within video
  "similarity": 0.1734,                    // Matching similarity score
  "description": "Add-1/2 tsp baking..."   // Human-readable step description
}
```

**Key for Substep 4:**
- `recording_id`: Identifies which video execution this step belongs to
- `video_label`: Ground truth label (0=correct, 1=incorrect) for GNN training
- `task_name`: Groups steps into task-specific graphs

---

### 2. **Task Graph Embeddings**
**Location:** `outputs/task_graph_encodings/task_graph_embeddings.npz`

**Description:** Original task graph node embeddings (before fusion).

**Structure:** 
- Organized by task name: `data['blenderbananapancakes']` → shape (14, 256)
- Each task has variable number of nodes

---

### 3. **Task Graph Metadata**
**Location:** `outputs/task_graph_encodings/task_graph_metadata.json`

**Description:** Task graph structure and step descriptions.

**Structure per task:**
```json
{
  "blenderbananapancakes": {
    "descriptions": [...],      // Step descriptions
    "steps": {...},            // Node ID to description mapping
    "edges": [[14,2], [7,3]...] // Graph edges (directed)
  }
}
```

---

### 4. **Fusion Model (Trained)**
**Location:** `outputs/fusion_model/best_fusion_model.pth`

**Description:** Trained fusion module that combines task and visual embeddings.

**Contains:**
- Model weights for the fusion network
- Training hyperparameters
- Embedding dimension info

**Usage for Substep 4:** 
Use this model to update task graph nodes with fused features:
```python
# Load fusion model
checkpoint = torch.load('outputs/fusion_model/best_fusion_model.pth')
fusion_model = FeatureFusionModule(...)
fusion_model.load_state_dict(checkpoint['model_state_dict'])

# For each matched pair:
task_emb = torch.tensor(pair['task_embedding'])
visual_emb = torch.tensor(pair['visual_embedding'])
fused_emb = fusion_model(task_emb, visual_emb)

# Use fused_emb as updated node feature in task graph
```

---

### 5. **Visual Features Metadata**
**Location:** `visual_features/visual_features_mapping.json`

**Description:** Mapping between videos, tasks, and labels.

**Contents:**
- `video_to_task`: Maps video indices to task names
- `video_to_label`: Maps video indices to correct/incorrect labels
- `task_step_metadata`: Step-level metadata with recording IDs
- `task_stats`: Statistics per task

---

### 6. **Video Metadata**
**Location:** `visual_features/video_metadata.json`

**Description:** Detailed metadata for each recording.

**Structure:**
```json
{
  "146": {
    "recording_id": "21_17",
    "task": "blenderbananapancakes",
    "label": 0,
    "boundaries": [[...]], 
    "feature_file": "21_17_360p_224.mp4_1s_1s.npz"
  }
}
```

---

## How to Use This Data for Substep 4

### Step 1: Group Matched Pairs by Recording
```python
import json
from collections import defaultdict

with open('outputs/matched_features/matched_pairs.json', 'r') as f:
    pairs = json.load(f)

# Group by recording
recording_to_pairs = defaultdict(list)
for pair in pairs:
    recording_to_pairs[pair['recording_id']].append(pair)

print(f"Total recordings with matched steps: {len(recording_to_pairs)}")
```

### Step 2: Construct Per-Recording Task Graphs
```python
import torch
import numpy as np

# For each recording, build a task graph
for recording_id, matched_steps in recording_to_pairs.items():
    task_name = matched_steps[0]['task_name']
    label = matched_steps[0]['video_label']  # 0 or 1
    
    # Load base task graph structure
    tg_data = np.load('outputs/task_graph_encodings/task_graph_embeddings.npz')
    with open('outputs/task_graph_encodings/task_graph_metadata.json', 'r') as f:
        tg_meta = json.load(f)
    
    # Get edges for this task
    edges = tg_meta[task_name]['edges']
    
    # Initialize node features (use fusion model to update matched nodes)
    node_features = tg_data[task_name].copy()  # Shape: (num_nodes, 256)
    
    # Update matched nodes with fused features
    for pair in matched_steps:
        task_idx = pair['task_idx']
        task_emb = torch.tensor(pair['task_embedding'])
        visual_emb = torch.tensor(pair['visual_embedding'])
        
        # Get fused embedding
        fused_emb = fusion_model(task_emb.unsqueeze(0), 
                                visual_emb.unsqueeze(0))
        
        # Update node feature
        node_features[task_idx] = fused_emb.detach().numpy()
    
    # Now you have:
    # - node_features: (num_nodes, 256) with updated features
    # - edges: list of [source, target] pairs
    # - label: 0 (correct) or 1 (incorrect)
    
    # Build PyTorch Geometric graph
    # ... (use torch_geometric to create Data object)
```

### Step 3: Train GNN Classifier
```python
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# Create dataset of graph objects
graphs = []
for recording_id in recording_to_pairs:
    # Build graph as shown above
    graph = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long).t(),
        y=torch.tensor([label], dtype=torch.long)
    )
    graphs.append(graph)

# Train GNN classifier
# ... (standard PyG training loop)
```

---

## Statistics

- **Total matched pairs:** 358
- **From correct executions:** 148 pairs
- **From incorrect executions:** 210 pairs
- **Tasks:** 24 different recipes
- **Recordings:** 384 total videos (many recordings have 0 matched steps)

---

## Important Notes

1. **Not all recordings will have matched steps** - The Hungarian matching only matches the most similar task nodes to visual steps. Many recordings might have 0 matched pairs.

2. **Variable graph sizes** - Different tasks have different numbers of nodes (7-25 nodes).

3. **Imbalanced classes** - More incorrect executions (210) than correct (148) in matched pairs.

4. **Fusion model is trained** - Use the saved fusion model to generate fused embeddings rather than training from scratch.

---

## Files to Share with Substep 4 Project

Essential files:
1. `outputs/matched_features/matched_pairs.json` (358 pairs with metadata)
2. `outputs/task_graph_encodings/task_graph_embeddings.npz` (base graph embeddings)
3. `outputs/task_graph_encodings/task_graph_metadata.json` (graph structure)
4. `outputs/fusion_model/best_fusion_model.pth` (trained fusion model)
5. `visual_features/video_metadata.json` (recording metadata)

Optional (for reference):
6. `train_fusion.py` (fusion model architecture definition)
7. This README

---

## Questions or Issues?

If any data is missing or unclear, check:
- Visual features were reorganized with metadata preservation
- Matching was performed with updated script
- All 358 pairs have `recording_id` and `video_label` fields
