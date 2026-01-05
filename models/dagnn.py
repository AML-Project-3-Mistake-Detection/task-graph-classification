"""
DAGNN (Directed Acyclic Graph Neural Network) implementation
for Task Graph Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


class DAGNNConv(MessagePassing):
    """
    DAGNN Convolutional Layer
    Designed for directed acyclic graphs
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add', flow='source_to_target')
        self.lin = nn.Linear(in_channels, out_channels)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, x, edge_index):
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Transform node features
        x = self.lin(x)
        
        # Propagate messages
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # x_j: features of neighbor nodes
        return x_j


class DAGNN(nn.Module):
    """
    DAGNN model for graph classification
    
    Architecture:
    - Multiple DAGNN convolutional layers
    - Global pooling (mean/max)
    - MLP classifier
    """
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 num_classes=2,
                 dropout=0.3,
                 pooling='mean'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.pooling = pooling
        
        # DAGNN layers
        self.convs = nn.ModuleList()
        self.convs.append(DAGNNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(DAGNNConv(hidden_channels, hidden_channels))
        
        # Batch normalization
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]
            
        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # DAGNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x = self._global_pool(x, batch)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def _global_pool(self, x, batch):
        """
        Global pooling: aggregate node features to graph-level
        
        Args:
            x: Node features [num_nodes, hidden_channels]
            batch: Batch assignment [num_nodes]
            
        Returns:
            Graph-level features [batch_size, hidden_channels]
        """
        from torch_geometric.nn import global_mean_pool, global_max_pool
        
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'mean_max':
            return torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")


class GCNClassifier(nn.Module):
    """
    Simple GCN-based classifier for comparison
    """
    
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 num_classes=2,
                 dropout=0.3):
        super().__init__()
        
        from torch_geometric.nn import GCNConv, global_mean_pool
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout
        self.global_pool = global_mean_pool
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.global_pool(x, batch)
        return self.classifier(x)
