"""
Data loader for Task Graph Classification
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch_geometric.data import Data, Dataset
import networkx as nx
import numpy as np


class TaskGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for Task Graph Classification
    
    Loads:
    1. Standard task graphs from annotations/task_graphs/
    2. Observed graphs from Substep 3 outputs
    3. Video embeddings from Extension3 outputs (hiero_step_embeddings_256.npz)
    """
    
    def __init__(self, 
                 task_graphs_dir: str,
                 observed_graphs_dir: str,
                 extension3_dir: str = None,
                 use_hiero_embeddings: bool = True,
                 transform=None,
                 pre_transform=None):
        """
        Args:
            task_graphs_dir: Path to standard task graphs (from annotations/)
            observed_graphs_dir: Path to observed graphs (from Substep 3)
            extension3_dir: Path to Extension3 outputs (for hiero embeddings)
            use_hiero_embeddings: Whether to use 256D hiero embeddings vs one-hot
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform
        """

        self.task_graphs_dir = Path(task_graphs_dir).resolve()
        self.observed_graphs_dir = Path(observed_graphs_dir).resolve()
        self.use_hiero_embeddings = use_hiero_embeddings
        
        # Load hiero embeddings if available
        self.hiero_embeddings = None
        if use_hiero_embeddings and extension3_dir:
            self._load_hiero_embeddings(extension3_dir)
        
        # Load standard task graphs
        self.standard_graphs = self._load_standard_task_graphs()
        
        # Calculate max steps across all recipes (for consistent feature dimensions)
        self.max_steps = max(
            len(graph.get('steps', {})) 
            for graph in self.standard_graphs.values()
        ) if self.standard_graphs else 20
        
        if self.hiero_embeddings is not None:
            print(f"Using 256D hiero embeddings from Extension3")
        else:
            print(f"Using one-hot encoding with max_steps={self.max_steps}")
        
        # Load Substep 3 outputs
        self.observed_graphs = self._load_substep3_outputs()
        
        super().__init__(None, transform, pre_transform)
    
    def _load_hiero_embeddings(self, extension3_dir: str):
        """Load hiero_step_embeddings_256.npz from Extension3"""
        extension3_path = Path(extension3_dir).resolve()
        hiero_path = extension3_path / "visual_features" / "hiero_step_embeddings_256.npz"
        
        if hiero_path.exists():
            try:
                hiero_data = np.load(hiero_path, allow_pickle=True)
                self.hiero_embeddings = {
                    'step_embeddings': hiero_data['step_embeddings'],  # (384, 61, 256)
                    'step_mask': hiero_data['step_mask'],              # (384, 61) bool
                    'labels': hiero_data['labels'],                    # (384,)
                    'video_ids': hiero_data['video_ids']               # (384,) str
                }
                print(f"✓ Loaded hiero embeddings from {hiero_path}")
                print(f"  Total videos: {len(self.hiero_embeddings['labels'])}")
                print(f"  Embedding shape per video: {self.hiero_embeddings['step_embeddings'].shape}")
            except Exception as e:
                print(f"Warning: Failed to load hiero embeddings: {e}")
                self.hiero_embeddings = None
        else:
            print(f"Warning: hiero_step_embeddings_256.npz not found at {hiero_path}")
    
    def _load_standard_task_graphs(self) -> Dict[str, Dict]:
        """Load all standard task graphs from annotations"""
        task_graphs = {}
        
        for json_file in self.task_graphs_dir.glob("*.json"):
            recipe_name = json_file.stem
            with open(json_file, 'r') as f:
                task_graphs[recipe_name] = json.load(f)
        
        print(f"Loaded {len(task_graphs)} standard task graphs")
        return task_graphs
    
    def _load_substep3_outputs(self) -> List[Dict]:
        """Load observed graphs from Substep 3 outputs"""
        observed_graphs = []
        
        if not self.observed_graphs_dir.exists():
            print(f"ERROR: {self.observed_graphs_dir} does not exist")
            return observed_graphs
        
        json_files = list(self.observed_graphs_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in {self.observed_graphs_dir}")
        
        for json_file in json_files:
            # Skip metadata/stats files
            if json_file.name.startswith('_'):
                continue
                
            try:
                with open(json_file, 'r') as f:
                    graph_data = json.load(f)
                    # Validate required fields
                    if 'recipe_id' in graph_data and 'observed_steps' in graph_data:
                        observed_graphs.append(graph_data)
                    else:
                        print(f"Warning: Skipping {json_file.name} - missing required fields")
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
                continue
        
        print(f"Loaded {len(observed_graphs)} observed graphs")
        return observed_graphs
    
    def len(self) -> int:
        """Return number of graphs"""
        return len(self.observed_graphs)
    
    def get(self, idx: int) -> Data:
        """
        Get a single graph as PyTorch Geometric Data object
        
        Args:
            idx: Index of the graph
            
        Returns:
            PyG Data object with:
                - x: Node features [num_nodes, feature_dim]
                - edge_index: Edge connectivity [2, num_edges]
                - y: Label (1=correct, 0=incorrect)
                - recipe_id: Recipe identifier
        """
        observed_graph = self.observed_graphs[idx]
        recipe_id = observed_graph['recipe_id']
        
        # Get standard task graph for this recipe
        standard_graph = self.standard_graphs[recipe_id]
        
        # Extract node features
        x = self._extract_node_features(observed_graph, standard_graph)
        
        # Extract edges
        edge_index = self._extract_edges(observed_graph)
        
        # Get label (as scalar, not tensor)
        y = observed_graph.get('label', 1)
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            recipe_id=recipe_id,
            video_id=observed_graph.get('video_id', 'unknown')
        )
        
        return data
    
    def _extract_node_features(self, 
                               observed_graph: Dict,
                               standard_graph: Dict) -> torch.Tensor:
        """
        Extract node features for the observed graph
        
        Three options:
        1. Use hiero 256D embeddings from Extension3 (recommended)
        2. Pre-computed text embeddings (from observed_graph)
        3. One-hot encoding of step IDs (fallback)
        """
        observed_steps = observed_graph.get('observed_steps', [])
        num_nodes = len(observed_steps)
        
        if num_nodes == 0:
            # Empty graph - return dummy feature
            feature_dim = 256 if self.hiero_embeddings else self.max_steps
            return torch.zeros((1, feature_dim), dtype=torch.float)
        
        # Option 1: Use hiero 256D embeddings (if available)
        if self.hiero_embeddings is not None:
            video_id_str = observed_graph.get('video_id', None)
            if video_id_str is not None:
                try:
                    video_id = int(video_id_str)
                    # Get embeddings for this video
                    video_embeddings = self.hiero_embeddings['step_embeddings'][video_id]  # (61, 256)
                    step_mask = self.hiero_embeddings['step_mask'][video_id]                # (61,) bool
                    
                    # Extract features for observed steps
                    features = []
                    for step_idx in observed_steps:
                        step_idx_int = int(step_idx) if isinstance(step_idx, str) else step_idx
                        if step_idx_int < len(video_embeddings) and step_mask[step_idx_int]:
                            features.append(video_embeddings[step_idx_int])
                        else:
                            # Fallback: use zero vector
                            features.append(np.zeros(256))
                    
                    if features:
                        return torch.tensor(np.array(features), dtype=torch.float)
                except Exception as e:
                    print(f"Warning: Failed to extract hiero embeddings: {e}")
        
        # Option 2: Check if embeddings are provided in observed_graph
        if 'step_embeddings' in observed_graph:
            embeddings = observed_graph['step_embeddings']
            features = []
            for step_id in observed_steps:
                if str(step_id) in embeddings:
                    features.append(embeddings[str(step_id)])
            if features:
                return torch.tensor(features, dtype=torch.float)
        
        # Option 3: One-hot encoding using global max_steps for consistency (fallback)
        features = torch.zeros((num_nodes, self.max_steps), dtype=torch.float)
        for i, step_id in enumerate(observed_steps):
            try:
                step_idx = int(step_id)
                if step_idx < self.max_steps:
                    features[i, step_idx] = 1.0
            except (ValueError, TypeError):
                continue
        
        return features
    
    def _extract_edges(self, observed_graph: Dict) -> torch.Tensor:
        """
        Extract edge connectivity from observed graph
        
        Returns:
            edge_index: [2, num_edges] tensor with node indices (0 to num_nodes-1)
        """
        observed_steps = observed_graph.get('observed_steps', [])
        
        # Create mapping from step_id to node index
        step_to_idx = {int(step_id): idx for idx, step_id in enumerate(observed_steps)}
        
        # Try to get edges from the graph
        edges = observed_graph.get('edges', [])
        
        if not edges:
            # Fallback: Create a simple chain if no edges provided
            edges = [[i, i+1] for i in range(len(observed_steps)-1)]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Remap edges from step_id to node index
            remapped_edges = []
            for src, dst in edges:
                src_int = int(src)
                dst_int = int(dst)
                # Only add edge if both nodes exist in observed_steps
                if src_int in step_to_idx and dst_int in step_to_idx:
                    remapped_edges.append([step_to_idx[src_int], step_to_idx[dst_int]])
            
            if remapped_edges:
                edge_index = torch.tensor(remapped_edges, dtype=torch.long).t().contiguous()
            else:
                # Empty graph case
                edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return edge_index


def load_task_graphs(task_graphs_dir: str) -> Dict[str, Dict]:
    """
    Utility function to load all task graphs
    
    Args:
        task_graphs_dir: Path to task graphs directory
        
    Returns:
        Dictionary mapping recipe names to task graphs
    """
    task_graphs = {}
    task_graphs_path = Path(task_graphs_dir)
    
    for json_file in task_graphs_path.glob("*.json"):
        recipe_name = json_file.stem
        with open(json_file, 'r') as f:
            task_graphs[recipe_name] = json.load(f)
    
    return task_graphs
