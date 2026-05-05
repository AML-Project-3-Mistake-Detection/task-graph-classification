"""
Configuration file for Task Graph Classification

This file contains all hyperparameters and paths for the project.
You can modify values here instead of passing command-line arguments.

Usage:
    from configs.config import Config
    config = Config()
    
Or override from command line:
    python train.py --hidden_dim 256 --num_epochs 200
"""
import os
from pathlib import Path

class Config:
    """Configuration class for training and evaluation"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    ANNOTATIONS_DIR = PROJECT_ROOT / "annotations" / "task_graphs"
    DATA_DIR = PROJECT_ROOT / "data"
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    # Data paths - RENAMED for clarity
    OBSERVED_GRAPHS_DIR = DATA_DIR / "observed_graphs"  # Observation graphs with Extension3 data
    EXTENSION3_DIR = DATA_DIR / "extension3_outputs"    # Extension3 data (for hiero embeddings)
    
    # Google Drive path (for Colab) - can be overridden
    DRIVE_OBSERVED_GRAPHS_DIR = "/content/drive/MyDrive/AML_Project/substep3_outputs"
    
    # Extension3 settings
    USE_EXTENSION3_EMBEDDINGS = True  # Use 32D hiero embeddings vs one-hot
    
    # Model settings
    MODEL_TYPE = "dagnn"  # Options: dagnn, gcn, graphsage
    HIDDEN_DIM = 32      # Hidden layer dimension
    NUM_LAYERS = 3        # Number of GNN layers
    DROPOUT = 0.3         # Dropout rate for regularization
    
    # Node feature settings
    NODE_FEATURE_TYPE = "one_hot"  # Options: one_hot, text_embedding, learned
    EMBEDDING_DIM = 768  # For text embeddings (BERT/EgoVLP size)
    
    # Training settings
    BATCH_SIZE = 8        # Graphs per batch
    NUM_EPOCHS = 100      # Total training epochs
    LEARNING_RATE = 0.001 # Adam learning rate
    WEIGHT_DECAY = 1e-4   # L2 regularization
    
    # Evaluation settings
    EVAL_STRATEGY = "leave_one_out"  # LOO cross-validation (24 folds)
    NUM_RECIPES = 24      # Total number of recipes in dataset
    
    # Device and reproducibility
    DEVICE = "cuda"       # "cuda" or "cpu"
    SEED = 42             # Random seed for reproducibility
    
    # Logging and checkpointing
    LOG_INTERVAL = 10     # Log every N epochs
    SAVE_CHECKPOINT = True # Save best model checkpoint
    
    @classmethod
    def from_args(cls, args):
        """Update config from command line arguments"""
        config = cls()
        for key, value in vars(args).items():
            # Only override if value is not None (i.e., was explicitly provided)
            if value is not None and hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        return config
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__class__.__dict__.items() 
                 if not k.startswith('_') and not callable(v)}
        return f"Config({attrs})"
