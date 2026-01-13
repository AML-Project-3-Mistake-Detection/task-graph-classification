"""
Extract Observed Task Graphs with On-the-Fly Hungarian Matching

This script generates 384 PyG Data objects (one per video) by:
1. Loading hiero embeddings for all 384 videos
2. For each video, performing Hungarian matching between visual steps and task steps
3. Fusing matched pairs using the trained Fusion Model
4. Constructing graphs with observed steps only

Usage:
    python extract_graphs.py --device cpu --output data/processed_graphs.pt
"""

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from torch_geometric.data import Data
from tqdm import tqdm
from torch_geometric.utils import add_self_loops

class FeatureFusionModule(nn.Module):
    """Fusion model from teammate's code."""
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


class VideoGraphExtractor:
    """Extract graphs for 384 individual videos with on-the-fly matching."""
    
    def __init__(self,
                 hiero_path: str,
                 visual_mapping_path: str,
                 task_embeddings_path: str,
                 metadata_path: str,
                 fusion_checkpoint: str,
                 device: str = 'cpu'):
        
        self.device = device
        
        print("="*80)
        print("Loading resources...")
        print("="*80)
        
        # 1. Load hiero embeddings (384 videos)
        print("[1/5] Loading hiero embeddings...")
        hiero_data = np.load(hiero_path, allow_pickle=True)
        self.hiero_step_embeddings = hiero_data['step_embeddings']  # (384, 61, 256)
        self.hiero_step_mask = hiero_data['step_mask']              # (384, 61)
        self.hiero_labels = hiero_data['labels']                    # (384,)
        self.hiero_video_ids = hiero_data['video_ids']              # (384,)
        print(f"✓ Loaded {len(self.hiero_labels)} videos")
        
        # 2. Load video-to-task mapping
        print("[2/5] Loading video-to-task mapping...")
        with open(visual_mapping_path) as f:
            mapping = json.load(f)
        self.video_to_task = mapping['video_to_task']  # {0: 'pinwheels', ...}
        print(f"✓ Loaded mapping for {len(self.video_to_task)} videos")
        
        # 3. Load task embeddings (per recipe)
        print("[3/5] Loading task embeddings...")
        self.task_embeddings_npz = np.load(task_embeddings_path, allow_pickle=True)
        print(f"✓ Loaded embeddings for {len(self.task_embeddings_npz.files)} recipes")
        
        # 4. Load metadata (edges, structure)
        print("[4/5] Loading metadata...")
        with open(metadata_path) as f:
            self.metadata = json.load(f)
        print(f"✓ Loaded metadata for {len(self.metadata)} recipes")
        
        # 5. Load Fusion model
        print("[5/5] Loading Fusion model...")
        self.fusion_model = self._load_fusion_model(fusion_checkpoint)
        self.fusion_model.to(device).eval()
        print(f"✓ Fusion model loaded\n")
    
    def _load_fusion_model(self, checkpoint_path: str) -> FeatureFusionModule:
        """Load fusion model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            embedding_dim = checkpoint.get('embedding_dim', 256)
            fusion_type = checkpoint.get('fusion_type', 'concat')
            hidden_dim = checkpoint.get('hidden_dim', 512)
            output_dim = checkpoint.get('output_dim', 256)
            
            model = FeatureFusionModule(
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                fusion_type=fusion_type
            )
            model.load_state_dict(state_dict)
        else:
            model = FeatureFusionModule(fusion_type='concat')
            if isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
        
        return model
    
    def _compute_cosine_similarity(self, embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix.
        
        Args:
            embeddings_a: (N, D)
            embeddings_b: (M, D)
        
        Returns:
            similarity: (N, M) cosine similarity matrix
        """
        # Normalize
        norm_a = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-8)
        norm_b = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(norm_a, norm_b.T)
        return similarity
    
    def _hungarian_matching(self, visual_steps: np.ndarray, task_steps: np.ndarray) -> List[Tuple[int, int]]:
        """
        Perform Hungarian matching between visual and task steps.
        
        Args:
            visual_steps: (N_vis, 256) visual embeddings
            task_steps: (N_task, 256) task embeddings
        
        Returns:
            matches: List of (visual_idx, task_idx) pairs
        """
        # Compute similarity
        similarity = self._compute_cosine_similarity(visual_steps, task_steps)
        
        # Convert to cost (maximize similarity = minimize negative similarity)
        cost_matrix = -similarity
        
        # Hungarian algorithm
        vis_indices, task_indices = linear_sum_assignment(cost_matrix)
        
        matches = list(zip(vis_indices.tolist(), task_indices.tolist()))
        return matches
     
    def extract_all_videos(self) -> List[Data]:
        """Extract graphs for all 384 videos using Zero-Filling for missing steps."""
        all_graphs = []
        
        print("="*80)
        print(f"Extracting graphs for {len(self.video_to_task)} videos...")
        print("="*80 + "\n")
        
        for video_idx in tqdm(range(len(self.video_to_task)), desc="Processing videos"):
            try:
                # 1. 基础信息获取
                video_id_str = str(video_idx)
                if video_id_str not in self.video_to_task: continue
                recipe_name = self.video_to_task[video_id_str]
                if recipe_name not in self.metadata: continue

                # 2. 获取该视频有效的视觉片段 (Visual Features)
                step_mask = self.hiero_step_mask[video_idx]
                valid_indices = np.where(step_mask)[0]
                if len(valid_indices) == 0: continue # 空视频跳过
                visual_steps = self.hiero_step_embeddings[video_idx, valid_indices] # Shape: [M, 256]

                # 3. 获取该食谱的标准文本步骤 (Task Features)
                if recipe_name not in self.task_embeddings_npz.files: continue
                task_steps_all = self.task_embeddings_npz[recipe_name] # Shape: [N, 256] (N是标准步骤数)
                num_std_steps = task_steps_all.shape[0]

                # 4. 匈牙利匹配 (Matching)
                # 注意：这里计算所有视觉片段和所有文本步骤的匹配
                matches = self._hungarian_matching(visual_steps, task_steps_all)
                # 转换成字典方便查找: {task_idx: visual_idx}
                task_to_vis_map = {t_idx: v_idx for v_idx, t_idx in matches}

                # 5. 【核心修正】构建节点特征 (Node Features) - 填空逻辑
                node_features_list = []
                with torch.no_grad():
                    # 必须遍历 0 到 N-1 所有的标准步骤
                    for t_idx in range(num_std_steps):
                        # A. 拿文本特征
                        t_emb = torch.FloatTensor(task_steps_all[t_idx]).to(self.device).unsqueeze(0)
                        
                        # B. 检查是否匹配上了视觉特征
                        if t_idx in task_to_vis_map:
                            # 有匹配：融合task+visual
                            v_idx = task_to_vis_map[t_idx]
                            v_emb = torch.FloatTensor(visual_steps[v_idx]).to(self.device).unsqueeze(0)
                            fused = self.fusion_model(t_emb, v_emb)
                            node_features_list.append(fused.cpu())
                        else:
                            # 没匹配：输出全0向量（表示缺失的步骤）
                            zero_feat = torch.zeros(1, 256, dtype=torch.float32)
                            node_features_list.append(zero_feat)
                
                # 拼成 [N, 256] 的矩阵，注意 N 是标准步骤数，永远固定！
                x = torch.cat(node_features_list, dim=0)

                # 6. 【核心修正】构建边 (Edges) - 使用标准元数据
                # 因为节点现在是对齐标准的，所以直接用 metadata 里的边即可，不需要过滤！
                standard_edges = self.metadata[recipe_name]['edges']
                
                # 过滤掉涉及 START/END 的边 (如果有的话)，并对齐索引
                valid_edges = []
                for src, dst in standard_edges:
                    s, d = int(src), int(dst)
                    # 假设 metadata 里 steps 包含 '0'(START) 和 'END'，且步骤从1开始编号
                    # 我们需要根据你的 metadata 实际格式调整。
                    # 通常处理方式：只保留 1..N 的边，并减1映射到 0..N-1
                    # 这里假设 metadata 里的数字是 0-based 且直接对应 task_steps_all 的索引
                    if 0 <= s < num_std_steps and 0 <= d < num_std_steps:
                        valid_edges.append([s, d])
                
                if not valid_edges:
                    # 保底：时间顺序连线
                    valid_edges = [[i, i+1] for i in range(num_std_steps - 1)]

                if valid_edges:
                    edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
                else:
                    edge_index = torch.zeros((2, 0), dtype=torch.long)
                
                # 添加自环
                edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])

                # 7. 获取标签
                label = int(self.hiero_labels[video_idx])
                # 注意 hiero label 定义：通常 1 是 correct, 0 是 error (请根据实际情况确认)
                # 你的描述里说 hiero labels 是 correct/incorrect 标签

                # 8. 封装
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=torch.tensor([label], dtype=torch.float), # 改成 float 适合 BCEWithLogitsLoss
                    video_id=video_id_str,
                    hiero_id=self.hiero_video_ids[video_idx],
                    task_name=recipe_name
                )
                all_graphs.append(data)

            except Exception as e:
                # print(f"Error: {e}") # 调试时可以打开
                continue
        
        print(f"\n✓ Successfully extracted {len(all_graphs)} graphs")
        return all_graphs

    def save(self, graphs: List[Data], output_path: str):
        """Save graphs to disk."""
        torch.save(graphs, output_path)
        print(f"\n✓ Saved {len(graphs)} graphs to {output_path}")


def main(args):
    extractor = VideoGraphExtractor(
        hiero_path=args.hiero_embeddings,
        visual_mapping_path=args.visual_mapping,
        task_embeddings_path=args.task_embeddings,
        metadata_path=args.metadata,
        fusion_checkpoint=args.fusion_checkpoint,
        device=args.device
    )
    
    graphs = extractor.extract_all_videos()
    extractor.save(graphs, args.output)
    
    print("\n" + "="*80)
    print("✅ Done! Ready for GNN training.")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract video graphs with on-the-fly Hungarian matching')
    
    parser.add_argument(
        '--hiero_embeddings',
        type=str,
        default='data/extension3_outputs/visual_features/hiero_step_embeddings_256.npz',
        help='Path to hiero_step_embeddings_256.npz'
    )
    parser.add_argument(
        '--visual_mapping',
        type=str,
        default='data/extension3_outputs/visual_features/visual_features_mapping.json',
        help='Path to visual_features_mapping.json'
    )
    parser.add_argument(
        '--task_embeddings',
        type=str,
        default='data/extension3_outputs/task_graph_encodings/task_graph_embeddings.npz',
        help='Path to task_graph_embeddings.npz'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='data/extension3_outputs/task_graph_encodings/task_graph_metadata.json',
        help='Path to task_graph_metadata.json'
    )
    parser.add_argument(
        '--fusion_checkpoint',
        type=str,
        default='data/extension3_outputs/fusion_model/best_fusion_model.pth',
        help='Path to best_fusion_model.pth'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed_graphs.pt',
        help='Output path'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device'
    )
    
    args = parser.parse_args()
    main(args)
