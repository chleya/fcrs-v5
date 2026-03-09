"""
FCRS-v5 Predictive: 方向A
每个表征有自己的预测器

核心改进：预测器与选择器整合
"""

import numpy as np


class PerRepPredictor:
    """每个表征的独立预测器"""
    
    def __init__(self, compress_dim, lr=0.01):
        self.compress_dim = compress_dim
        self.lr = lr
        
        # 预测器权重
        self.W = np.eye(compress_dim) * 0.5
        self.b = np.zeros(compress_dim)
        
        # 历史
        self.f_history = []
        self.errors = []
    
    def predict(self, f_curr):
        """预测下一时刻"""
        return f_curr @ self.W + self.b
    
    def update(self, f_curr, f_next):
        """更新预测器"""
        pred = self.predict(f_curr)
        error = f_next - pred
        self.errors.append(np.linalg.norm(error))
        
        # 更新权重
        self.W += self.lr * np.outer(f_curr, error)
        self.b += self.lr * error
        
        # 记录历史
        self.f_history.append(f_curr)
        if len(self.f_history) > 5:
            self.f_history.pop(0)
        
        return np.linalg.norm(error)
    
    def get_error(self):
        """获取平均预测误差"""
        if len(self.errors) < 3:
            return 1.0  # 初始给高误差
        return np.mean(self.errors[-5:])


class PredictiveFCRS_A:
    """
    方向A: 每个表征有自己的预测器
    
    选择机制：选择预测误差最小的表征
    """
    
    def __init__(self, input_dim=10, compress_dim=5, capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        # 编码器
        self.W = np.random.randn(input_dim, compress_dim) * 0.1
        self.b = np.zeros(compress_dim)
        
        # 表征池：每个表征有自己的预测器
        self.reps = []
        self.rep_predictors = {}
        
        # 初始化
        for i in range(3):
            rep = np.random.randn(input_dim)
            rep = rep / (np.linalg.norm(rep) + 1e-8)
            self.reps.append(rep)
            self.rep_predictors[i] = PerRepPredictor(compress_dim, lr)
        
        self.step = 0
        self.pred_errors = []
        self.recon_errors = []
        
        self.prev_f = None
    
    def compress(self, x):
        """压缩输入"""
        f = x @ self.W + self.b
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def select_by_prediction(self, x, current_f):
        """
        核心：基于每个表征的预测误差选择
        """
        # 探索
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        # 计算每个表征的预测误差
        scores = {}
        for i, rep in enumerate(self.reps):
            # 用这个表征压缩当前输入
            rep_f = self.compress(rep)
            
            # 用这个表征的预测器预测
            pred_f = self.rep_predictors[i].predict(rep_f)
            
            # 预测误差作为得分（负值=越小越好）
            # 但我们需要和current_f比较
            # 好的预测 = 预测结果与实际结果接近
            
            # 更直接的方式：评估这个表征对当前输入的"预测价值"
            # 即：用这个表征的历史预测表现
            error = self.rep_predictors[i].get_error()
            scores[i] = -error  # 负误差 = 高得分
        
        # 选择得分最高的（预测误差最小的）
        best_idx = max(scores, key=scores.get)
        return best_idx
    
    def forward(self, x):
        self.step += 1
        
        # 压缩
        f = self.compress(x)
        
        # 更新每个表征的预测器
        if self.prev_f is not None:
            for i in range(len(self.reps)):
                # 用前一个压缩表示更新预测器
                error = self.rep_predictors[i].update(self.prev_f, f)
        
        # 记录
        self.prev_f = f
        
        # 选择表征（基于预测误差）
        idx = self.select_by_prediction(x, f)
        rep = self.reps[idx]
        
        # 重构误差
        recon_err = np.linalg.norm(rep - x)
        self.recon_errors.append(recon_err)
        
        # 更新表征
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        self.reps[idx] = new_rep
        
        return recon_err
    
    def run(self, env, steps):
        for _ in range(steps):
            self.forward(env.gen())
        
        return {
            'pred_error': np.mean([p.get_error() for p in self.rep_predictors.values()]),
            'recon_error': np.mean(self.recon_errors[-100:]) if self.recon_errors else 0,
            'pool_size': len(self.reps)
        }


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 实验 ====================
def test_A_vs_baseline():
    """测试方向A vs 基线"""
    print("=" * 60)
    print("Test A: Per-Representor Predictor")
    print("=" * 60)
    
    results = {'A': [], 'recon': [], 'random': []}
    
    for run in range(10):
        # 方向A
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        results['A'].append(r['pred_error'])
        
        # 重构选择
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
                return {'pred_error': np.mean(self.errors[-100:])}
        
        sys2 = ReconSys()
        results['recon'].append(sys2.run(env, 1000)['pred_error'])
        
        # 随机选择
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        class RandomSys:
            def __init__(self):
                self.reps = [np.random.randn(10) for _ in range(3)]
                self.errors = []
            
            def forward(self, x):
                idx = np.random.randint(len(self.reps))
                err = np.linalg.norm(self.reps[idx] - x)
                self.reps[idx] += 0.01 * (x - self.reps[idx])
                self.reps[idx] /= np.linalg.norm(self.reps[idx]) + 1e-8
                self.errors.append(err)
            
            def run(self, env, steps):
                for _ in range(steps):
                    self.forward(env.gen())
                return {'pred_error': np.mean(self.errors[-100:])}
        
        sys3 = RandomSys()
        results['random'].append(sys3.run(env, 1000)['pred_error'])
    
    print(f"\n方向A (per-rep预测): {np.mean(results['A']):.4f}")
    print(f"重构选择: {np.mean(results['recon']):.4f}")
    print(f"随机选择: {np.mean(results['random']):.4f}")
    
    print(f"\n方向A vs 重构: {(np.mean(results['recon']) - np.mean(results['A'])) / np.mean(results['recon']) * 100:+.1f}%")
    print(f"方向A vs 随机: {(np.mean(results['random']) - np.mean(results['A'])) / np.mean(results['random']) * 100:+.1f}%")


def test_exploration():
    """测试探索率"""
    print("\n" + "=" * 60)
    print("Exploration Rate Test")
    print("=" * 60)
    
    for exp in [0.0, 0.05, 0.1, 0.2, 0.3]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS_A(10, 3, 10, 0.01, exp)
        r = sys.run(env, 1000)
        print(f"exp={exp}: pred_error={r['pred_error']:.4f}")


def test_compression_dim():
    """测试压缩维度"""
    print("\n" + "=" * 60)
    print("Compression Dim Test")
    print("=" * 60)
    
    for dim in [1, 2, 3, 5]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS_A(10, dim, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        print(f"dim={dim}: pred_error={r['pred_error']:.4f}")


# ==================== Main ====================
if __name__ == "__main__":
    test_A_vs_baseline()
    test_exploration()
    test_compression_dim()
    print("\nDone!")
