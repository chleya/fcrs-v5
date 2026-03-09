"""
预测导向有限竞争表征系统 v2
修复版
"""

import numpy as np
from typing import List, Optional


class PredictiveCompressor:
    """预测导向压缩器"""
    
    def __init__(self, input_dim: int, compress_dim: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.lr = learning_rate
        
        # 编码器
        self.W = np.random.randn(input_dim, compress_dim) * 0.1
        self.b = np.zeros(compress_dim)
        
        # 预测器
        self.W_pred = np.eye(compress_dim) * 0.9  # 初始为单位矩阵
        self.b_pred = np.zeros(compress_dim)
        
        self.error_history = []
    
    def compress(self, x: np.ndarray) -> np.ndarray:
        """压缩：x -> f"""
        f = x @ self.W + self.b
        f = np.maximum(0, f)  # ReLU
        norm = np.linalg.norm(f)
        if norm > 1e-6:
            f = f / norm
        return f
    
    def predict(self, f_curr: np.ndarray) -> np.ndarray:
        """预测：f_t -> f_{t+1}"""
        f_next = f_curr @ self.W_pred + self.b_pred
        return f_next
    
    def update(self, f_curr: np.ndarray, f_next: np.ndarray):
        """更新预测器"""
        pred = self.predict(f_curr)
        error = f_next - pred
        self.error_history.append(np.linalg.norm(error))
        
        # 简单更新
        self.W_pred += self.lr * np.outer(f_curr, error)
        self.b_pred += self.lr * error
    
    def get_errors(self) -> dict:
        if not self.error_history:
            return {'mean': 0, 'std': 0}
        recent = self.error_history[-100:]
        return {'mean': np.mean(recent), 'std': np.std(recent)}


class PredictiveFCRS:
    """预测导向FCRS"""
    
    def __init__(self, input_dim=10, compress_dim=5, capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.capacity = capacity
        self.explore = explore
        
        self.compressor = PredictiveCompressor(input_dim, compress_dim, lr)
        
        # 表征池
        self.reps = [np.random.randn(input_dim) for _ in range(3)]
        
        self.step = 0
        self.pred_errors = []
        self.recon_errors = []
    
    def select(self, x: np.ndarray) -> int:
        """ε-贪心选择"""
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        # 选择重构误差最小的
        scores = []
        for rep in self.reps:
            err = np.linalg.norm(rep - x)
            scores.append(-err)
        
        return np.argmax(scores)
    
    def forward(self, x: np.ndarray):
        self.step += 1
        
        # 压缩
        f = self.compressor.compress(x)
        
        # 预测 & 更新
        if self.step > 1:
            pred = self.compressor.predict(self.prev_f)
            self.compressor.update(self.prev_f, f)
            self.pred_errors.append(np.linalg.norm(pred - f))
        
        self.prev_f = f
        
        # 选择表征
        idx = self.select(x)
        rep = self.reps[idx]
        
        # 重构误差
        recon_err = np.linalg.norm(rep - x)
        self.recon_errors.append(recon_err)
        
        # 更新表征
        self.reps[idx] = rep + 0.01 * (x - rep)
        self.reps[idx] = self.reps[idx] / (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        return recon_err
    
    def run(self, env, steps):
        for _ in range(steps):
            self.forward(env.gen())
        
        return {
            'pred_error': np.mean(self.pred_errors[-100:]) if self.pred_errors else 0,
            'recon_error': np.mean(self.recon_errors[-100:]) if self.recon_errors else 0,
            'pool_size': len(self.reps)
        }


class Env:
    """环境"""
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3
    
    def generate_input(self):
        return self.gen()


# ==================== 实验 ====================
def exp_1_1():
    """预测机制有效性"""
    print("=" * 60)
    print("实验1.1: 预测机制有效性")
    print("=" * 60)
    
    results = {'pred': [], 'rand': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        # Predictive
        env = Env(10, 5)
        sys = PredictiveFCRS(10, 3, 10, 0.01, 0.1)
        r1 = sys.run(env, 1000)
        results['pred'].append(r1['pred_error'])
        
        # Random (高探索)
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys2 = PredictiveFCRS(10, 3, 10, 0.01, 0.9)
        r2 = sys2.run(env, 1000)
        results['rand'].append(r2['pred_error'])
    
    pred_mean = np.mean(results['pred'])
    rand_mean = np.mean(results['rand'])
    improvement = (rand_mean - pred_mean) / rand_mean * 100
    
    print(f"\nPredictive: {pred_mean:.4f}")
    print(f"Random: {rand_mean:.4f}")
    print(f"改进: {improvement:.1f}%")
    
    return pred_mean, rand_mean, improvement


def exp_1_2():
    """压缩维度"""
    print("\n" + "=" * 60)
    print("实验1.2: 压缩维度影响")
    print("=" * 60)
    
    results = []
    
    for dim in [1, 2, 3, 5, 7]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS(10, dim, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        results.append((dim, r['pred_error']))
        print(f"dim={dim}: error={r['pred_error']:.4f}")
    
    best = min(results, key=lambda x: x[1])
    print(f"\n最优: dim={best[0]}")
    
    return results


def exp_1_3():
    """探索率"""
    print("\n" + "=" * 60)
    print("实验1.3: 探索率影响")
    print("=" * 60)
    
    results = []
    
    for exp in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS(10, 3, 10, 0.01, exp)
        r = sys.run(env, 1000)
        results.append((exp, r['pred_error']))
        print(f"exp={exp}: error={r['pred_error']:.4f}")
    
    best = min(results, key=lambda x: x[1])
    print(f"\n最优: exp={best[0]}")
    
    return results


# ==================== Main ====================
if __name__ == "__main__":
    print("FCRS-v5 Predictive Experiments v2\n")
    
    exp_1_1()
    exp_1_2()
    exp_1_3()
    
    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
