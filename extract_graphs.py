"""
Extract Observed Task Graphs using Matched Pairs from Substep 3

This script generates PyG Data objects (one per recording) by:
1. Loading the pre-computed matched_pairs.json
2. Grouping the matches by recording_id
3. Fusing matched pairs using the trained Fusion Model
4. Constructing task graphs with fused node features for matched nodes,
   and base task embeddings for unmatched nodes.

Usage:
    python extract_graphs.py \
    --matched_pairs data/matched_features/matched_pairs.json \
    --task_embeddings data/task_graph_encodings_256/task_graph_embeddings.npz \
    --task_metadata data/task_graph_encodings_256/task_graph_metadata.json \
    --fusion_model data/fusion_model/best_fusion_model.pth \
    --output data/processed_graphs.pt
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.utils import add_self_loops

class FeatureFusionModule(nn.Module):
    """Fusion model from Substep 3."""
    def __init__(self, embedding_dim=256, hidden_dim=512, output_dim=256, fusion_type='concat'):
        super().__init__()
        self.fusion_type = fusion_type
        self.embedding_dim = embedding_dim
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif fusion_type == 'cross_attention':
            self.query_proj = nn.Linear(embedding_dim, embedding_dim)
            self.key_proj = nn.Linear(embedding_dim, embedding_dim)
            self.value_proj = nn.Linear(embedding_dim, embedding_dim)
            self.out_proj = nn.Linear(embedding_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(embedding_dim, output_dim)
            self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, task_features, visual_features):
        if self.fusion_type == 'concat':
            combined = torch.cat([task_features, visual_features], dim=-1)
            fused = self.fusion(combined)
        elif self.fusion_type == 'cross_attention':
            Q = self.query_proj(task_features)
            K = self.key_proj(visual_features)
            V = self.value_proj(visual_features)
            attn_weights = torch.softmax(Q @ K.T / np.sqrt(self.embedding_dim), dim=-1)
            attended = attn_weights @ V
            fused = self.norm(self.out_proj(attended))
        elif self.fusion_type == 'gated':
            gate_input = torch.cat([task_features, visual_features], dim=-1)
            gate = self.gate(gate_input)
            combined = gate * task_features + (1 - gate) * visual_features
            fused = self.norm(self.proj(combined))
        return fused


def _detect_fusion_type_from_state_dict(state_dict: dict) -> str:
    """Detect fusion type from state_dict keys."""
    state_keys = set(state_dict.keys())
    
    if any('gate' in k for k in state_keys):
        return 'gated'
    elif any('query_proj' in k for k in state_keys):
        return 'cross_attention'
    elif any('fusion' in k for k in state_keys):
        return 'concat'
    else:
        return 'gated'  # default fallback


def load_fusion_model(
    checkpoint_path: str,
    device: str,
    fusion_type: str = 'gated',
    output_dim: int = 256,
) -> FeatureFusionModule:
    """Load fusion model from checkpoint.

    Behavior:
    - If checkpoint contains `fusion_type` or `output_dim` keys, those values
      will be preferred. Otherwise the provided `fusion_type` and
      `output_dim` arguments are used to construct the module.
    - If checkpoint is a raw state_dict, automatically detects the fusion type
      from the state_dict keys.
    - If loading fails, raises an informative error.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', 256)
        # Prefer checkpoint-stored fusion_type, then detect from state_dict, then use args
        if 'fusion_type' in checkpoint:
            ckpt_fusion_type = checkpoint['fusion_type']
        else:
            ckpt_fusion_type = _detect_fusion_type_from_state_dict(state_dict)
        ckpt_output_dim = checkpoint.get('output_dim', output_dim)
        hidden_dim = checkpoint.get('hidden_dim', 512)

        model = FeatureFusionModule(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=ckpt_output_dim,
            fusion_type=ckpt_fusion_type,
        )
        model.load_state_dict(state_dict)
    else:
        # Construct with detected or requested output_dim and fusion_type
        if isinstance(checkpoint, dict):
            # Try to detect fusion type from state_dict
            detected_fusion_type = _detect_fusion_type_from_state_dict(checkpoint)
            model = FeatureFusionModule(
                fusion_type=detected_fusion_type,
                output_dim=output_dim
            )
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError as e:
                print(f"⚠ Failed to load state_dict with detected fusion type '{detected_fusion_type}'.")
                print(f"  State dict keys: {list(checkpoint.keys())[:5]}...")
                raise e
        else:
            model = checkpoint

    model.to(device).eval()
    return model

def build_dataset(args):
    print("="*80)
    print("Loading resources for Substep 4...")
    print("="*80)
    
    # 1. Load matched pairs
    print("[1/4] Loading matched pairs...")
    with open(args.matched_pairs, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # If it's a dict, check if it has a 'pairs' key or if keys are recording IDs
        if 'pairs' in data:
            pairs = data['pairs']
        elif 'matches' in data:
            pairs = data['matches']
        else:
            # Assume it's keyed by recording_id; convert to list format
            pairs = []
            for recording_id, pair_list in data.items():
                if isinstance(pair_list, list):
                    pairs.extend(pair_list)
                else:
                    # Single pair per recording
                    pair_list['recording_id'] = recording_id
                    pairs.append(pair_list)
    else:
        # It's already a list
        pairs = data
    
    print(f"✓ Loaded {len(pairs)} matched pairs")

    recording_to_pairs = defaultdict(list)
    for pair in pairs:
        if isinstance(pair, dict) and 'recording_id' in pair:
            recording_to_pairs[pair['recording_id']].append(pair)
        else:
            print(f"⚠ Skipping invalid pair: {pair}")
    print(f"✓ Found {len(recording_to_pairs)} unique recordings")

    # 1.5 Load true labels from original annotations
    print("[1.5/4] Loading true labels from original annotations...")
    with open('annotations/annotation_json/error_annotations.json', 'r') as f:
        err_ann = json.load(f)
    rec_to_label = {}
    for entry in err_ann:
        rec_to_label[entry['recording_id']] = 0 if entry['is_error'] else 1
    print(f"✓ Loaded {len(rec_to_label)} true labels")

    # 2. Load task graph embeddings (Base nodes)
    print("[2/4] Loading base task graph embeddings...")
    tg_data = np.load(args.task_embeddings, allow_pickle=True)
    
    # 3. Load task graph metadata (Edges)
    print("[3/4] Loading task graph metadata...")
    with open(args.task_metadata, 'r') as f:
        tg_meta = json.load(f)
        
    # 4. Load Fusion Model
    if args.fusion_model:
        print("[4/4] Loading trained fusion model...")
        fusion_model = load_fusion_model(
            args.fusion_model,
            args.device,
            fusion_type=args.fusion_type,
            output_dim=args.fusion_output_dim,
        )
    else:
        print("[4/4] No fusion model provided. Using base features.")
        fusion_model = None

    all_graphs = []
    
    print("\n" + "="*80)
    print(f"Constructing Task Graphs...")
    print("="*80)

    for recording_id, matched_steps in tqdm(recording_to_pairs.items(), desc="Processing recordings"):
        task_name = matched_steps[0]['task_name']
        
        # Override the label from substep 3 with the original annotation if available
        if recording_id in rec_to_label:
            label = rec_to_label[recording_id]
        else:
            label = matched_steps[0].get('video_label', 1)
        
        # Make sure task base data exists
        if task_name not in tg_data.files or task_name not in tg_meta:
            continue
            
        # Get base edges and adjust if necessary
        standard_edges = tg_meta[task_name].get('edges', [])
        
        # Initialize node features with base task embeddings (Shape: num_nodes, 256)
        node_features = tg_data[task_name].copy()
        num_std_steps = node_features.shape[0]

        # Update matched nodes with fused features
        with torch.no_grad():
            for pair in matched_steps:
                task_idx = pair['task_idx']
                
                # Check valid index bounds
                if task_idx < 0 or task_idx >= num_std_steps:
                    continue
                    
                task_emb = torch.tensor(pair['task_embedding'], dtype=torch.float32).to(args.device)
                visual_emb = torch.tensor(pair['visual_embedding'], dtype=torch.float32).to(args.device)
                
                # Forward pass: shape requires [1, 256]
                fused_emb = fusion_model(task_emb.unsqueeze(0), visual_emb.unsqueeze(0))
                
                # Overwrite original embedding with fused embedding
                node_features[task_idx] = fused_emb.cpu().squeeze().numpy()

        # Build PyG Graph
        # Ensure edges refer to valid bounds
        valid_edges = []
        for src, dst in standard_edges:
            s, d = int(src), int(dst)
            if 0 <= s < num_std_steps and 0 <= d < num_std_steps:
                valid_edges.append([s, d])
        
        # Fallback to sequential edges if empty
        if not valid_edges:
            valid_edges = [[i, i+1] for i in range(num_std_steps - 1)]

        if valid_edges:
            edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        # Add self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_std_steps)

        x_tensor = torch.tensor(node_features, dtype=torch.float32)
        y_tensor = torch.tensor([label], dtype=torch.float)

        data = Data(
            x=x_tensor,
            edge_index=edge_index,
            y=y_tensor,
            recording_id=recording_id,
            task_name=task_name
        )
        all_graphs.append(data)

    # Save outputs
    torch.save(all_graphs, args.output)
    print(f"\n✓ Extracted and saved {len(all_graphs)} graphs to {args.output}")
    print("\n" + "="*80)
    print("✅ Done! Ready for GNN training.")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build PyG graphs using Substep 3 Matched Pairs')
    parser.add_argument(
        '--matched_pairs',
        type=str,
        default='data/extension3_outputs/matched_features/matched_pairs.json',
        help='Path to matched_pairs.json containing pre-computed matches'
    )
    parser.add_argument(
        '--task_embeddings',
        type=str,
        default='data/extension3_outputs/task_graph_encodings/task_graph_embeddings.npz',
        help='Path to base task_graph_embeddings.npz'
    )
    parser.add_argument(
        '--task_metadata',
        type=str,
        default='data/extension3_outputs/task_graph_encodings/task_graph_metadata.json',
        help='Path to task_graph_metadata.json'
    )
    parser.add_argument(
        '--fusion_model',
        type=str,
        default='data/extension3_outputs/fusion_model/best_fusion_model.pth',
        help='Path to pretrained visual-text best_fusion_model.pth'
    )
    parser.add_argument(
        '--fusion_type',
        type=str,
        default='gated',
        choices=['concat', 'gated'],
        help='Fusion strategy to use when fusing task and visual embeddings (default: gated)'
    )
    parser.add_argument(
        '--fusion_output_dim',
        type=int,
        default=256,
        help='Output embedding dimension of the fusion module (default: 256)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_graphs.pt',
        help='Output path for the PyG Dataset'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Compute device'
    )
    
    args = parser.parse_args()
    build_dataset(args)
