import subprocess
import itertools
import time

# ==========================================
# 定义你想尝试的参数网格
# ==========================================
param_grid = {
    'model_type': ['gcn', 'dagnn'],        # 比较两个模型
    'hidden_dim': [64, 128, 256],          # 尝试不同宽度
    'num_layers': [2, 3],                  # 尝试不同深度
    'lr': [0.001, 0.0005, 0.0001],         # 尝试不同学习率
    'dropout': [0.3, 0.5],                 # 尝试不同正则化力度
    'batch_size': [16]                     # 固定 Batch Size
}

# 生成所有组合
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"🚀 总共需要运行 {len(combinations)} 组实验...")
print(f"预计耗时: {len(combinations) * 1.5:.1f} 分钟 (假设每组跑90秒)")
print("="*60)

for i, params in enumerate(combinations):
    print(f"\n[{i+1}/{len(combinations)}] Running experiment with: {params}")
    
    # 构建命令
    cmd = [
        "python", "train.py",
        "--model_type", params['model_type'],
        "--hidden_dim", str(params['hidden_dim']),
        "--num_layers", str(params['num_layers']),
        "--lr", str(params['lr']),
        "--dropout", str(params['dropout']),
        "--batch_size", str(params['batch_size']),
        "--num_epochs", "50",      # 统一跑50轮
        "--patience", "15"         # 早停耐心
    ]
    
    # 运行命令
    try:
        start_time = time.time()
        subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        print(f"✅ 完成! 耗时: {duration:.2f}s")
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验失败: {e}")

print("\n" + "="*60)
print("🎉 所有实验运行完毕！请查看 results/experiment_results.csv")
print("="*60)