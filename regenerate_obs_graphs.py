#!/usr/bin/env python3
"""
Regenerate observation graphs using correct numeric video_id (hiero index 0-383).
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def load_task_graph(recipe_name, annotations_dir):
    """Load a standard task graph from annotations."""
    graph_path = Path(annotations_dir) / f"{recipe_name}.json"
    if not graph_path.exists():
        return None
    
    with open(graph_path) as f:
        graph_data = json.load(f)
    
    return graph_data


def generate_observation_graphs(
    hiero_embeddings_path: str,
    visual_features_mapping_path: str,
    annotations_dir: str,
    output_dir: str
):
    """
    Generate observation graphs.

    Args:
        hiero_embeddings_path: Path to hiero_step_embeddings_256.npz
        visual_features_mapping_path: Path to visual_features_mapping.json
        annotations_dir: Path to task graphs directory
        output_dir: Output directory for observation graphs
    """
    # 1. Load hiero embeddings
    print("Loading hiero_step_embeddings_256.npz...")
    hiero_path = Path(hiero_embeddings_path)
    if not hiero_path.exists():
        raise FileNotFoundError(f"Hiero embeddings not found: {hiero_path}")
    
    hiero_data = np.load(hiero_path, allow_pickle=True)
    
    hiero_video_ids = hiero_data['video_ids']  # (384,) 例如 '10_16'
    hiero_labels = hiero_data['labels']        # (384,)
    step_embeddings = hiero_data['step_embeddings']  # (384, 61, 256)
    step_mask = hiero_data['step_mask']        # (384, 61)
    
    print(f"Total videos: {len(hiero_video_ids)}")
    print(f"Label distribution: 0={np.sum(hiero_labels==0)}, 1={np.sum(hiero_labels==1)}")
    
    # 2. Load video_to_task mapping
    print("\nLoading visual_features_mapping.json...")
    mapping_path = Path(visual_features_mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Visual features mapping not found: {mapping_path}")
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    video_to_task = mapping['video_to_task']
    
    # 3. Load standard task graphs
    print("\nLoading standard task graphs...")
    annotations_path = Path(annotations_dir)
    task_graphs = {}
    recipes = list(set(video_to_task.values()))
    for recipe in recipes:
        graph_data = load_task_graph(recipe, annotations_path)
        if graph_data:
            task_graphs[recipe] = graph_data
    print(f"Loaded {len(task_graphs)} task graphs")
    
    # 4. Generate observation graphs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating observation_graphs to {output_path}...")
    
    failed = 0
    success = 0
    
    for video_id in range(len(hiero_video_ids)):
        recipe_name = video_to_task[str(video_id)]
        label = int(hiero_labels[video_id])  # 0 or 1
        
        if recipe_name not in task_graphs:
            print(f"  ⚠ Recipe {recipe_name} not found")
            failed += 1
            continue
        
        task_graph = task_graphs[recipe_name]
        
        # Get valid steps for this video
        valid_mask = step_mask[video_id]  # (61,) bool
        valid_steps = np.where(valid_mask)[0].tolist()  # indices of valid steps
        
        # Skip if no valid steps
        if not valid_steps:
            failed += 1
            continue
        
        # Build observation graph
        obs_graph = {
            "recipe_id": recipe_name,
            "video_id": str(video_id),  # important: numeric string as video_id
            "hiero_video_id": hiero_video_ids[video_id],  # original hiero video_id
            "label": label,  # 0=correct, 1=incorrect
            "observed_steps": valid_steps,  # executed step indices
            "steps": task_graph.get("steps", {}),  # step descriptions
            "edges": task_graph.get("edges", []),  # task graph edges
            "metadata": {
                "num_observed_steps": len(valid_steps),
                "hiero_embeddings_shape": list(step_embeddings[video_id].shape),
                "step_mask_sum": int(np.sum(valid_mask))
            }
        }
        
        # Save as JSON
        filename = f"{recipe_name}_{video_id}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w') as f:
            json.dump(obs_graph, f, indent=2)
        
        success += 1
        
        if success % 50 == 0:
            print(f"  Generated {success} observation graphs...")
    
    print(f"\n✓ Successfully generated: {success}")
    print(f"✗ Failed: {failed}")
    print(f"Total: {success + failed}")
    
    # 5. Generate summary statistics
    stats = {
        "total_videos": len(hiero_video_ids),
        "successfully_generated": success,
        "failed": failed,
        "label_distribution": {
            "correct": int(np.sum(hiero_labels == 0)),
            "incorrect": int(np.sum(hiero_labels == 1))
        },
        "output_directory": str(output_path)
    }
    
    stats_path = output_path / "_dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Statistics saved to {stats_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate observation graphs from Extension3 data")
    parser.add_argument(
        "--hiero_embeddings",
        type=str,
        default="data/extension3_outputs/visual_features/hiero_step_embeddings_256.npz",
        help="Path to hiero_step_embeddings_256.npz"
    )
    parser.add_argument(
        "--visual_mapping",
        type=str,
        default="data/extension3_outputs/visual_features/visual_features_mapping.json",
        help="Path to visual_features_mapping.json"
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default="annotations/task_graphs",
        help="Path to task graphs directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/observed_graphs",
        help="Output directory for observation graphs"
    )
    
    args = parser.parse_args()
    
    generate_observation_graphs(
        hiero_embeddings_path=args.hiero_embeddings,
        visual_features_mapping_path=args.visual_mapping,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir
    )
