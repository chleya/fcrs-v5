"""
实验1.1: 预测机制有效性验证
比较：PredictiveFCRS vs RandomBaseline
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5/predictive')
from core_predictive import PredictiveFCRS, RandomBaseline


class SimpleEnv:
    """简单环境"""
    def __init__(self, input_dim=10, complexity=5):
        self.input_dim = input_dim
        self.class_centers = {i: np.random.randn(input_dim) * 2 for i in range(complexity)}
    
    def generate_input(self):
        cls = np.random.randint(0, len(self.class_centers))
        center = self.class_centers[cls]
        return center + np.random.randn(self.input_dim) * 0.3


def experiment_1_1():
    """实验1.1：预测机制有效性验证"""
    print("=" * 60)
    print("实验1.1：预测机制有效性验证")
    print("=" * 60)
    
    # 参数
    input_dim = 10
    compress_dim = 3
    steps = 1000
    
    results = {'predictive': [], 'random': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        # PredictiveFCRS
        env = SimpleEnv(input_dim, complexity=5)
        system = PredictiveFCRS(
            input_dim=input_dim,
            compress_dim=compress_dim,
            learning_rate=0.01,
            exploration_rate=0.1
        )
        system.run(env, steps)
        stats = system.get_statistics()
        results['predictive'].append(stats['mean_prediction_error'])
        
        # Random (高探索率模拟)
        np.random.seed(run * 100)
        env2 = SimpleEnv(input_dim, complexity=5)
        system2 = PredictiveFCRS(
            input_dim=input_dim,
            compress_dim=compress_dim,
            learning_rate=0.01,
            exploration_rate=0.9  # 几乎随机
        )
        system2.run(env2, steps)
        stats2 = system2.get_statistics()
        results['random'].append(stats2['mean_prediction_error'])
    
    # 对比
    pred_mean = np.mean(results['predictive'])
    rand_mean = np.mean(results['random'])
    improvement = (rand_mean - pred_mean) / rand_mean * 100
    
    print(f"\n结果对比:")
    print(f"  PredictiveFCRS: {pred_mean:.4f}")
    print(f"  RandomBaseline: {rand_mean:.4f}")
    print(f"  改进幅度: {improvement:.1f}%")
    
    if improvement > 15:
        print(f"  [PASS] 预测机制有效！")
    else:
        print(f"  [FAIL] 预测机制效果不明显")
    
    return {'predictive': pred_mean, 'random': rand_mean, 'improvement': improvement}


def experiment_1_2():
    """实验1.2：压缩维度影响验证"""
    print("\n" + "=" * 60)
    print("实验1.2：压缩维度影响验证")
    print("=" * 60)
    
    input_dim = 10
    compress_dims = [1, 2, 3, 5, 7, 9]
    results = []
    
    for compress_dim in compress_dims:
        np.random.seed(42)
        env = SimpleEnv(input_dim, complexity=5)
        system = PredictiveFCRS(
            input_dim=input_dim,
            compress_dim=compress_dim,
            learning_rate=0.01,
            exploration_rate=0.1
        )
        system.run(env, 1000)
        stats = system.get_statistics()
        
        results.append({
            'compress_dim': compress_dim,
            'compression_ratio': input_dim / compress_dim,
            'prediction_error': stats['mean_prediction_error']
        })
        
        print(f"compress_dim={compress_dim}: pred_error={stats['mean_prediction_error']:.4f}")
    
    best = min(results, key=lambda x: x['prediction_error'])
    print(f"\n最优: compress_dim={best['compress_dim']}, 压缩比={best['compression_ratio']:.1f}x")
    
    return results


def experiment_1_3():
    """实验1.3：探索率影响验证"""
    print("\n" + "=" * 60)
    print("实验1.3：探索率影响验证")
    print("=" * 60)
    
    input_dim = 10
    exploration_rates = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    results = []
    
    for exp_rate in exploration_rates:
        np.random.seed(42)
        env = SimpleEnv(input_dim, complexity=5)
        system = PredictiveFCRS(
            input_dim=input_dim,
            compress_dim=3,
            learning_rate=0.01,
            exploration_rate=exp_rate
        )
        system.run(env, 1000)
        stats = system.get_statistics()
        
        results.append({
            'exploration_rate': exp_rate,
            'prediction_error': stats['mean_prediction_error']
        })
        
        print(f"exploration_rate={exp_rate}: pred_error={stats['mean_prediction_error']:.4f}")
    
    best = min(results, key=lambda x: x['prediction_error'])
    print(f"\n最优: exploration_rate={best['exploration_rate']}")
    
    return results


# ==================== Main ====================
if __name__ == "__main__":
    print("FCRS-v5 Predictive Experiments\n")
    
    exp_1_1_results = experiment_1_1()
    exp_1_2_results = experiment_1_2()
    exp_1_3_results = experiment_1_3()
    
    print("\n" + "=" * 60)
    print("All Experiments Complete!")
    print("=" * 60)
