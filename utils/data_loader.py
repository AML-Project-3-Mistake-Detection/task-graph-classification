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
    """
    
    def __init__(self, 
                 task_graphs_dir: str,
                 observed_graphs_dir: str,
                 transform=None,
                 pre_transform=None):
        """
        Args:
            task_graphs_dir: Path to standard task graphs (from annotations/)
            observed_graphs_dir: Path to observed graphs (from Substep 3)
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform
        """
        self.task_graphs_dir = Path(task_graphs_dir).resolve()
        self.observed_graphs_dir = Path(observed_graphs_dir).resolve()
        
        # Load standard task graphs
        self.standard_graphs = self._load_standard_task_graphs()
        
        # Calculate max steps across all recipes (for consistent feature dimensions)
        self.max_steps = max(
            len(graph.get('steps', {})) 
            for graph in self.standard_graphs.values()
        ) if self.standard_graphs else 20
        print(f"Max steps across all recipes: {self.max_steps}")
        
        # Load Substep 3 outputs
        self.observed_graphs = self._load_substep3_outputs()
        
        super().__init__(None, transform, pre_transform)
    
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
        
        Options:
        1. One-hot encoding of step IDs (using global max_steps)
        2. Pre-computed text embeddings (from Substep 3)
        3. Learnable embeddings
        """
        observed_steps = observed_graph.get('observed_steps', [])
        num_nodes = len(observed_steps)
        
        # Option 1: Check if embeddings are provided
        if 'step_embeddings' in observed_graph:
            embeddings = observed_graph['step_embeddings']
            features = []
            for step_id in observed_steps:
                features.append(embeddings[str(step_id)])
            return torch.tensor(features, dtype=torch.float)
        
        # Option 2: One-hot encoding using global max_steps for consistency
        features = torch.zeros((num_nodes, self.max_steps), dtype=torch.float)
        for i, step_id in enumerate(observed_steps):
            # Convert step_id to int if it's a string
            try:
                step_idx = int(step_id)
                if step_idx < self.max_steps:
                    features[i, step_idx] = 1.0
            except (ValueError, TypeError):
                # If conversion fails, skip this step
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
