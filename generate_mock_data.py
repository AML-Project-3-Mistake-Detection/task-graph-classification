"""
Generate Mock Data for Task Graph Classification
Simulates Substep 3 outputs (observed task graphs)
"""
import json
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union


def load_standard_task_graph(json_path: Union[str, Path]) -> Dict:
    """Load a standard task graph"""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_correct_graph(recipe_name: str, task_graph: Dict) -> Dict:
    """Create a correct observed graph (matches standard)"""
    steps = task_graph["steps"].copy()
    # Remove START and END for observed_steps (filter by both key and value)
    observed_steps = [
        k for k, v in steps.items() 
        if k not in ("0", "END") and v not in ("START", "END")
    ]
    
    # Convert edges to integers
    edges = [[int(src), int(dst)] for src, dst in task_graph["edges"]]
    
    return {
        "recording_id": f"{recipe_name}_correct_{random.randint(1000, 9999)}",
        "recipe_id": recipe_name,
        "steps": steps,
        "edges": edges,
        "observed_steps": observed_steps,
        "label": 1,  # Correct execution
        "metadata": {
            "execution_time": random.randint(180, 600),
            "num_steps": len(observed_steps)
        }
    }


def create_incorrect_graph(recipe_name: str, task_graph: Dict, error_type: str = "random") -> Dict:
    """
    Create an incorrect observed graph
    
    Error types:
    - missing_edge: Remove some edges (skipped steps)
    - extra_edge: Add wrong dependencies
    - wrong_order: Swap some edges
    - missing_step: Remove a step
    """
    steps = task_graph["steps"].copy()
    # Convert edges to integers
    edges = [[int(src), int(dst)] for src, dst in task_graph["edges"]]
    error_applied = False
    
    if error_type == "missing_edge" and len(edges) > 1:
        # Remove 1-2 edges (skipped steps)
        num_remove = min(random.randint(1, 2), len(edges) - 1)
        for _ in range(num_remove):
            if edges:
                edges.pop(random.randint(0, len(edges) - 1))
                error_applied = True
    
    elif error_type == "extra_edge":
        # Add 1-2 wrong edges
        step_ids = [int(k) for k, v in steps.items() if k not in ("0", "END") and v not in ("START", "END")]
        if len(step_ids) >= 2:
            num_add = random.randint(1, 2)
            added = 0
            max_attempts = 50  # Prevent infinite loop
            attempts = 0
            while added < num_add and attempts < max_attempts:
                src = random.choice(step_ids)
                dst = random.choice(step_ids)
                if src != dst and [src, dst] not in edges:
                    edges.append([src, dst])
                    added += 1
                    error_applied = True
                attempts += 1
    
    elif error_type == "wrong_order":
        # Reverse 1-2 edges
        if len(edges) >= 2:
            num_swap = min(random.randint(1, 2), len(edges))
            for _ in range(num_swap):
                idx = random.randint(0, len(edges) - 1)
                edges[idx] = [edges[idx][1], edges[idx][0]]  # Reverse edge
                error_applied = True
    
    elif error_type == "missing_step":
        # Remove a step (not START/END)
        step_ids = [k for k, v in steps.items() if k not in ("0", "END") and v not in ("START", "END")]
        if step_ids:
            remove_id = random.choice(step_ids)
            del steps[remove_id]
            # Remove edges connected to this step
            remove_id_int = int(remove_id)
            edges = [e for e in edges if e[0] != remove_id_int and e[1] != remove_id_int]
            error_applied = True
    
    # Remove START and END for observed_steps (filter by both key and value)
    observed_steps = [
        k for k, v in steps.items() 
        if k not in ("0", "END") and v not in ("START", "END")
    ]
    
    # Warn if error could not be applied
    if not error_applied:
        print(f"⚠️  Warning: Could not apply error_type='{error_type}' to {recipe_name} (graph too small). Treating as label noise.")
    
    return {
        "recording_id": f"{recipe_name}_incorrect_{error_type}_{random.randint(1000, 9999)}",
        "recipe_id": recipe_name,
        "steps": steps,
        "edges": edges,
        "observed_steps": observed_steps,
        "label": 0,  # Incorrect execution
        "error_type": error_type,
        "error_applied": error_applied,  # Track whether error was successfully applied
        "metadata": {
            "execution_time": random.randint(120, 700),
            "num_steps": len(observed_steps)
        }
    }


def generate_mock_dataset(
    annotations_dir: str,
    output_dir: str,
    num_correct_per_recipe: int = 3,
    num_incorrect_per_recipe: int = 3
):
    """
    Generate mock dataset
    
    Args:
        annotations_dir: Path to annotations/task_graphs/
        output_dir: Output directory for mock data
        num_correct_per_recipe: Number of correct samples per recipe
        num_incorrect_per_recipe: Number of incorrect samples per recipe
    """
    annotations_path = Path(annotations_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all task graphs
    task_graph_files = list(annotations_path.glob("*.json"))
    print(f"Found {len(task_graph_files)} task graphs")
    
    error_types = ["missing_edge", "extra_edge", "wrong_order", "missing_step"]
    total_samples = 0
    
    for json_file in task_graph_files:
        recipe_name = json_file.stem
        task_graph = load_standard_task_graph(json_file)
        
        # Generate correct samples
        for i in range(num_correct_per_recipe):
            correct_graph = create_correct_graph(recipe_name, task_graph)
            output_file = output_path / f"{recipe_name}_correct_{i}.json"
            with open(output_file, 'w') as f:
                json.dump(correct_graph, f, indent=2)
            total_samples += 1
        
        # Generate incorrect samples
        for i in range(num_incorrect_per_recipe):
            error_type = random.choice(error_types)
            incorrect_graph = create_incorrect_graph(recipe_name, task_graph, error_type)
            output_file = output_path / f"{recipe_name}_incorrect_{error_type}_{i}.json"
            with open(output_file, 'w') as f:
                json.dump(incorrect_graph, f, indent=2)
            total_samples += 1
    
    print(f"\n✓ Generated {total_samples} mock samples")
    print(f"  - {len(task_graph_files) * num_correct_per_recipe} correct samples")
    print(f"  - {len(task_graph_files) * num_incorrect_per_recipe} incorrect samples")
    print(f"  - Saved to: {output_path}")
    
    # Generate summary statistics
    stats = {
        "num_recipes": len(task_graph_files),
        "num_correct_per_recipe": num_correct_per_recipe,
        "num_incorrect_per_recipe": num_incorrect_per_recipe,
        "total_samples": total_samples,
        "error_types": error_types
    }
    
    stats_file = output_path / "_dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✓ Dataset statistics saved to: {stats_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate mock data for task graph classification")
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
        help="Output directory for mock data"
    )
    parser.add_argument(
        "--num_correct",
        type=int,
        default=3,
        help="Number of correct samples per recipe"
    )
    parser.add_argument(
        "--num_incorrect",
        type=int,
        default=3,
        help="Number of incorrect samples per recipe"
    )
    
    args = parser.parse_args()
    
    generate_mock_dataset(
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        num_correct_per_recipe=args.num_correct,
        num_incorrect_per_recipe=args.num_incorrect
    )
