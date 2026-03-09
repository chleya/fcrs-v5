"""
预测导向FCRS - 进一步优化版
让预测器真正学习预测能力
"""

import numpy as np


class PredictiveCompressor:
    def __init__(self, input_dim, compress_dim, lr=0.01):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.lr = lr
        
        # 编码器
        self.W = np.random.randn(input_dim, compress_dim) * 0.1
        self.b = np.zeros(compress_dim)
        
        # 预测器 - 用多个历史预测
        self.W_pred = np.eye(compress_dim) * 0.5
        self.b_pred = np.zeros(compress_dim)
        
        # 历史
        self.f_history = []
    
    def compress(self, x):
        f = x @ self.W + self.b
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def predict(self, f_curr):
        return f_curr @ self.W_pred + self.b_pred
    
    def update(self, f_curr, f_next):
        pred = self.predict(f_curr)
        error = f_next - pred
        
        # 记录历史预测误差
        pred_error = np.linalg.norm(error)
        
        # 更新
        self.W_pred += self.lr * np.outer(f_curr, error)
        self.b_pred += self.lr * error
        
        # 记录
        self.f_history.append(f_curr)
        if len(self.f_history) > 10:
            self.f_history.pop(0)
        
        return pred_error
    
    def get_prediction_ability(self):
        """评估预测能力 - 用历史数据验证"""
        if len(self.f_history) < 3:
            return 0.5
        
        errors = []
        for i in range(len(self.f_history) - 1):
            pred = self.predict(self.f_history[i])
            actual = self.f_history[i + 1]
            errors.append(np.linalg.norm(pred - actual))
        
        return np.mean(errors) if errors else 0.5


class PredictiveFCRS:
    def __init__(self, input_dim=10, compress_dim=5, capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        self.compressor = PredictiveCompressor(input_dim, compress_dim, lr)
        
        # 表征池
        self.reps = [np.random.randn(input_dim) for _ in range(3)]
        
        self.step = 0
        self.pred_errors = []
        
        # 每个表征的预测得分
        self.rep_pred_scores = {i: [] for i in range(3)}
    
    def evaluate_rep_prediction(self, rep, current_f):
        """评估一个表征的预测能力"""
        # 用这个表征压缩当前输入
        f = self.compressor.compress(rep)
        
        # 预测下一时刻
        pred_f = self.compressor.predict(f)
        
        # 好的预测 = 预测结果稳定 + 与实际变化方向一致
        # 简单：预测误差
        return np.linalg.norm(pred_f - current_f)
    
    def select(self, x, current_f):
        """基于预测能力选择"""
        # 探索
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        # 计算每个表征的预测得分
        scores = []
        for i, rep in enumerate(self.reps):
            # 用表征来预测
            pred_ability = self.evaluate_rep_prediction(rep, current_f)
            scores.append(-pred_ability)  # 负误差 = 高得分
        
        return np.argmax(scores)
    
    def forward(self, x):
        self.step += 1
        
        # 压缩
        f = self.compressor.compress(x)
        
        # 预测 & 更新
        if self.step > 1:
            pred_error = self.compressor.update(self.prev_f, f)
            self.pred_errors.append(pred_error)
        
        self.prev_f = f
        
        # 选择表征
        idx = self.select(x, f)
        rep = self.reps[idx]
        
        # 更新表征
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        self.reps[idx] = new_rep
        
        return np.linalg.norm(rep - x)
    
    def run(self, env, steps):
        for _ in range(steps):
            self.forward(env.gen())
        
        return {
            'pred_error': np.mean(self.pred_errors[-100:]) if self.pred_errors else 0,
            'pool_size': len(self.reps)
        }


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 测试 ====================
def test_prediction_vs_random():
    print("=" * 60)
    print("Test: Prediction vs Random vs Recon")
    print("=" * 60)
    
    results = {'pred': [], 'rand': [], 'recon': []}
    
    for run in range(10):
        # Predictive
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = PredictiveFCRS(10, 3, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        results['pred'].append(r['pred_error'])
        
        # Random
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys2 = PredictiveFCRS(10, 3, 10, 0.01, 0.9)
        r2 = sys2.run(env, 1000)
        results['rand'].append(r2['pred_error'])
        
        # Recon-only
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        class ReconSys:
            def __init__(self):
                self.reps = [np.random.randn(10) for _ in range(3)]
                self.errors = []
            
            def forward(self, x):
                scores = [-np.linalg.norm(r - x) for r in self.reps]
                idx = np.argmax(scores)
                err = np.linalg.norm(self.reps[idx] - x)
                self.reps[idx] += 0.01 * (x - self.reps[idx])
                self.reps[idx] /= np.linalg.norm(self.reps[idx]) + 1e-8
                self.errors.append(err)
            
            def run(self, env, steps):
                for _ in range(steps):
                    self.forward(env.gen())
                return np.mean(self.errors[-100:]) if self.errors else 0
        
        sys3 = ReconSys()
        results['recon'].append(sys3.run(env, 1000))
    
    print(f"\nPredictive: {np.mean(results['pred']):.4f}")
    print(f"Random: {np.mean(results['rand']):.4f}")
    print(f"Recon: {np.mean(results['recon']):.4f}")
    
    print(f"\nPred vs Rand: {(np.mean(results['rand']) - np.mean(results['pred'])) / np.mean(results['rand']) * 100:+.1f}%")
    print(f"Pred vs Recon: {(np.mean(results['recon']) - np.mean(results['pred'])) / np.mean(results['recon']) * 100:+.1f}%")


def test_compression_dim():
    print("\n" + "=" * 60)
    print("Compression Dim Test")
    print("=" * 60)
    
    for dim in [1, 2, 3, 5]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS(10, dim, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        print(f"dim={dim}: {r['pred_error']:.4f}")


# ==================== Main ====================
test_prediction_vs_random()
test_compression_dim()
print("\nDone!")
