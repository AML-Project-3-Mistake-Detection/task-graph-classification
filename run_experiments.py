"""Hyperparameter Grid Search with Resume Support

Features:
- Incremental saving after each experiment (survives crashes)
- Skip already-completed configurations (resume support)
- Verification via .done marker file with checksums
- Support both LOO (precise) and Standard (fast) evaluation modes

Usage:
  python run_experiments_resume.py --mode standard  # Fast search (~30 min for 108 configs)
  python run_experiments_resume.py --mode loo       # Precise eval (~10+ hours for 108 configs)
"""
import subprocess
import itertools
import time
import pandas as pd
from pathlib import Path
import json
import argparse

# ==========================================
# Define hyperparameter grid
# ==========================================
param_grid = {
    'model_type': ['gcn', 'dagnn'],
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4],
    'lr': [0.001, 0.0005, 0.0001],
    'dropout': [0.3, 0.5],
    'batch_size': [16]
}

# Output paths
RESULTS_DIR = Path("results")
GRID_RESULTS_PATH = RESULTS_DIR / "loo_results_grid.csv"
PROGRESS_PATH = RESULTS_DIR / "grid_progress.json"

# Parse arguments
parser = argparse.ArgumentParser(description="Grid search for optimal hyperparameters")
parser.add_argument('--mode', type=str, default='loo', choices=['loo', 'standard'],
                    help='Evaluation mode: loo (precise, ~6min/config), standard (fast, ~20s/config)')
parser.add_argument('--data_path', type=str, default='data/processed_graphs.pt',
                    help='Path to preprocessed graphs (.pt file)')

args = parser.parse_args()

eval_mode = args.mode
data_path = args.data_path
# Extract dataset name from path for CSV naming (e.g., processed_graphs_258.pt -> _258)
dataset_suffix = Path(data_path).stem.replace('processed_graphs', '')

if eval_mode == 'standard':
    GRID_RESULTS_PATH = RESULTS_DIR / f"grid_search_standard{dataset_suffix}.csv"
    print(f"⚡ FAST MODE: Standard 80/20 split")
    print(f"📊 Using dataset: {data_path}")
    print(f"💾 Results will be saved to: {GRID_RESULTS_PATH}")
else:
    GRID_RESULTS_PATH = RESULTS_DIR / f"loo_results_grid{dataset_suffix}.csv"
    print(f"🎯 PRECISE MODE: LOO cross-validation")
    print(f"📊 Using dataset: {data_path}")
    print(f"💾 Results will be saved to: {GRID_RESULTS_PATH}")

print()

# Generate all combinations
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"🚀 Total experiments: {len(combinations)}")

# ==========================================
# Load existing progress (only count rows with metrics as completed)
# ==========================================
completed_configs = set()
all_results = []

if GRID_RESULTS_PATH.exists():
    print(f"\n📂 Found existing results: {GRID_RESULTS_PATH}")
    existing_df = pd.read_csv(GRID_RESULTS_PATH)

    # Only keep rows that already have metrics; drop blank/pending rows to avoid persisting empty lines
    if eval_mode == 'loo':
        valid_mask = existing_df['avg_f1'].notna()
    else:
        valid_mask = existing_df['best_f1'].notna()
    
    completed_df = existing_df[valid_mask]
    pending_df = existing_df[~valid_mask]
    all_results = completed_df.to_dict('records')

    for _, result in completed_df.iterrows():
        config_key = (
            result['model_type'],
            result['hidden_dim'],
            result['num_layers'],
            result['lr'],
            result['dropout'],
            result['batch_size']
        )
        completed_configs.add(config_key)

    print(f"✅ Completed (with metrics): {len(completed_configs)}/{len(combinations)}")
    print(f"🟡 Pending/failed rows dropped from cache: {len(pending_df)}")
    print(f"🔄 Remaining to run: {len(combinations) - len(completed_configs)}")
else:
    print("\n🆕 Starting fresh grid search")

print("="*60)

# ==========================================
# Run experiments
# ==========================================
for i, params in enumerate(combinations):
    # Check if already completed
    config_key = (
        params['model_type'],
        params['hidden_dim'],
        params['num_layers'],
        params['lr'],
        params['dropout'],
        params['batch_size']
    )
    
    if config_key in completed_configs:
        print(f"[{i+1}/{len(combinations)}] ⏭️  Skipping (already done): {params}")
        continue
    
    print(f"\n[{i+1}/{len(combinations)}] 🏃 Running: {params}")
    
    # Build command
    cmd = [
        "python", "train.py",
        "--eval_mode", eval_mode,  # ⚡ Use --mode argument
        "--data_path", data_path,
        "--model_type", params['model_type'],
        "--hidden_dim", str(params['hidden_dim']),
        "--num_layers", str(params['num_layers']),
        "--lr", str(params['lr']),
        "--dropout", str(params['dropout']),
        "--batch_size", str(params['batch_size']),
        "--num_epochs", "50",
        "--patience", "15"
    ]
    
    # Run command
    try:
        start_time = time.time()
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"✅ Completed in {duration:.2f}s")
        
        # 🔑 Step 1: Read and verify .done marker file (train.py wrote this)
        if eval_mode == 'loo':
            results_path = RESULTS_DIR / "loo_results.csv"
            done_marker_path = RESULTS_DIR / "loo_results.done"
            result_key = 'avg_f1'
        else:
            results_path = RESULTS_DIR / "experiment_results.csv"
            done_marker_path = RESULTS_DIR / "experiment_results.done"
            result_key = 'best_f1'
        
        success = False
        
        expected_f1 = None
        if done_marker_path.exists():
            try:
                with open(done_marker_path) as f:
                    done_data = json.load(f)
                expected_f1 = float(done_data['avg_f1'])  # .done 文件统一使用 avg_f1
                expected_num_recipes = done_data['num_recipes']
                print(f"✅ Verification from .done: {expected_num_recipes} recipes, F1={expected_f1:.6f}")
            except Exception as e:
                print(f"⚠️  Failed to read .done marker: {e}")
                done_data = None
                expected_f1 = None
        else:
            print(f"⚠️  .done marker not found, proceeding with CSV validation only")
            done_data = None
        
        # 🔑 Step 2: Read and validate CSV
        if results_path.exists():
            try:
                df = pd.read_csv(results_path)
                
                if eval_mode == 'loo':
                    # LOO mode: column names are "Recipe", "F1", "AUC", etc.
                    avg_row = df[df['Recipe'] == 'Average']
                    if not avg_row.empty and len(avg_row) == 1:
                        csv_f1 = float(avg_row['F1'].values[0])
                        csv_recipes = len(df) - 1
                        metrics = {
                            'avg_accuracy': float(avg_row['Accuracy'].values[0]),
                            'avg_f1': csv_f1,
                            'avg_auc': float(avg_row['AUC'].values[0]),
                            'avg_precision': float(avg_row['Precision'].values[0]),
                            'avg_recall': float(avg_row['Recall'].values[0]),
                        }
                        expected_f1_key = 'avg_f1'
                else:
                    # Standard mode: column names are "Best_F1", "Best_Acc", "Best_AUC", etc. (last row)
                    last_row = df.iloc[-1]
                    csv_f1 = float(last_row['Best_F1'])
                    csv_recipes = 1
                    metrics = {
                        'best_accuracy': float(last_row['Best_Acc']),
                        'best_f1': csv_f1,
                        'best_auc': float(last_row['Best_AUC']),
                        'best_threshold': float(last_row.get('Best_Threshold', 0.5)) if 'Best_Threshold' in last_row else 0.5,
                    }
                    expected_f1_key = 'best_f1'
                
                if all(not pd.isna(v) for v in metrics.values()):
                    # 🔑 Step 3: Cross-verify with .done marker if available
                    if expected_f1 is not None:
                        if abs(csv_f1 - expected_f1) < 1e-4:
                            print(f"✅ CSV verified against .done marker")
                            success = True
                        else:
                            print(f"❌ F1 mismatch: {csv_f1:.6f} vs {expected_f1:.6f}")
                    else:
                        print(f"✅ CSV looks valid (no .done marker to verify against)")
                        success = True
                else:
                    print(f"❌ NaN found in metrics: {metrics}")
            except Exception as e:
                print(f"❌ Error reading CSV: {e}")
        else:
            print(f"❌ CSV file not found: {results_path}")
        
        if success:
            result = {
                **params,
                **metrics,
                'duration_sec': duration
            }
            all_results.append(result)
            completed_configs.add(config_key)
            
            # Save immediately after each experiment (only rows with metrics)
            results_df = pd.DataFrame(all_results)
            if eval_mode == 'loo':
                results_with_metrics = results_df[results_df['avg_f1'].notna()]
            else:
                results_with_metrics = results_df[results_df['best_f1'].notna()]
            results_with_metrics.to_csv(GRID_RESULTS_PATH, index=False)
            print(f"💾 Saved to {GRID_RESULTS_PATH} ({len(results_with_metrics)} rows)")
            
            # Clean up .done marker
            try:
                done_marker_path.unlink(missing_ok=True)
            except:
                pass
        else:
            print(f"❌ Experiment result validation failed")
            # Do not persist blank rows; just continue so the combo will rerun next time
                
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        result = {
            **params,
            'avg_accuracy': None,
            'avg_f1': None,
            'avg_auc': None,
            'avg_precision': None,
            'avg_recall': None,
            'duration_sec': None
        }
        all_results.append(result)
        
        # Save even failed experiments, but DO NOT mark as completed
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(GRID_RESULTS_PATH, index=False)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        print(f"Progress saved to {GRID_RESULTS_PATH}")
        print(f"Run this script again to resume from where you left off.")
        break

# ==========================================
# Final summary
# ==========================================
print("\n" + "="*60)
if len(completed_configs) == len(combinations):
    print("🎉 All experiments completed!")
else:
    print(f"⏸️  Progress: {len(completed_configs)}/{len(combinations)} completed")
    print(f"Run again to continue with remaining {len(combinations) - len(completed_configs)} experiments")
print("="*60)

# Print top results
if all_results:
    results_df = pd.DataFrame(all_results)
    
    if eval_mode == 'loo':
        valid_results = results_df[results_df['avg_f1'].notna()]
        metric_col = 'avg_f1'
        display_cols = ['model_type', 'hidden_dim', 'num_layers', 'lr', 'dropout', 'avg_accuracy', 'avg_f1', 'avg_auc']
    else:
        valid_results = results_df[results_df['best_f1'].notna()]
        metric_col = 'best_f1'
        display_cols = ['model_type', 'hidden_dim', 'num_layers', 'lr', 'dropout', 'best_accuracy', 'best_f1', 'best_auc']
    
    if not valid_results.empty:
        print("\n🏆 Top 5 configurations by F1 score:")
        top5 = valid_results.nlargest(5, metric_col)
        print(top5[display_cols].to_string(index=False))
        
        print(f"\n📊 Full results: {GRID_RESULTS_PATH}")
