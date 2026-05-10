import torch
import torch.nn as nn

class FeatureFusionModule(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=512, output_dim=256, fusion_type='gated'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, task_features, visual_features):
        combined = torch.cat([task_features, visual_features], dim=-1)
        gate = self.gate(combined)
        fused = gate * task_features + (1 - gate) * visual_features
        return self.norm(self.proj(fused))
