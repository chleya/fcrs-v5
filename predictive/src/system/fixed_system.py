"""
FCRS-v5: 修复后的系统
真正基于预测的选择
"""

import numpy as np
from collections import defaultdict


class FixedPredictor:
    """修复后的预测器"""
    def __init__(self, dim):
        self.dim = dim
        self.W = np.eye(dim) * 0.5
        self.b = np.zeros(dim)
        self.errors = []
        self.history_f = []
    
    def predict(self, f):
        return f @ self.W + self.b
    
    def update(self, f_curr, f_next):
        pred = self.predict(f_curr)
        error = f_next - pred
        
        # 记录
        self.history_f.append(f_curr.copy())
        if len(self.history_f) > 20:
            self.history_f.pop(0)
        
        # 更新权重
        self.W += 0.01 * np.outer(f_curr, error)
        self.b += 0.01 * error
        
        self.errors.append(np.linalg.norm(error))
        
        return np.linalg.norm(error)
    
    def get_score(self):
        """获取预测质量分数"""
        if len(self.errors) < 5:
            return 0.5  # 默认中等
        
        # 最近误差的平均
        recent = np.mean(self.errors[-10:])
        
        # 转换为分数 (误差越小，分数越高)
        return 1.0 / (1.0 + recent)


class FixedSystem:
    """修复后的完整系统"""
    
    def __init__(self, input_dim=10, compress_dim=5, capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        # 编码器
        self.W_enc = np.random.randn(input_dim, compress_dim) * 0.1
        self.b_enc = np.zeros(compress_dim)
        
        # 表征池 + 预测器
        self.reps = []
        self.predictors = []
        
        # 初始化
        for _ in range(3):
            rep = np.random.randn(input_dim)
            rep = rep / (np.linalg.norm(rep) + 1e-8)
            self.reps.append(rep)
            self.predictors.append(FixedPredictor(compress_dim))
        
        # 统计
        self.step = 0
        self.prev_rep_idx = 0
        self.prev_f = None
        
        self.errors = []
        self.selection_log = []
    
    def encode(self, x):
        """压缩"""
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def select_by_prediction(self, x):
        """
        核心修复：真正基于预测选择！
        """
        f = self.encode(x)
        
        # ε-贪心探索
        if np.random.random() < self.explore:
            idx = np.random.randint(len(self.reps))
            self.selection_log.append(('explore', idx))
            return idx
        
        # 计算每个表征的预测分数
        scores = []
        for i, (rep, pred) in enumerate(zip(self.reps, self.predictors)):
            # 表征的压缩
            rep_f = self.encode(rep)
            
            # 用这个表征的预测器预测
            # 预测: 给定当前rep，下一状态是什么？
            predicted = pred.predict(rep_f)
            
            # 预测质量分数
            score = pred.get_score()
            
            # 同时考虑重构误差
            recon_err = np.linalg.norm(rep - x)
            
            # 综合分数: 预测分数 * 0.7 + (1/重构误差) * 0.3
            composite = score * 0.7 + (1.0 / (recon_err + 0.1)) * 0.3
            scores.append(composite)
        
        # 选择分数最高的
        idx = np.argmax(scores)
        self.selection_log.append(('exploit', idx))
        
        return idx
    
    def forward(self, x):
        self.step += 1
        
        # 压缩
        f = self.encode(x)
        
        # 选择表征
        idx = self.select_by_prediction(x)
        
        # 更新选择的预测器
        if self.prev_f is not None and self.prev_rep_idx == idx:
            # 继续用同一个表征，更新其预测器
            error = self.predictors[idx].update(self.prev_f, f)
        
        # 更新编码器 (简单)
        self.W_enc += self.lr * np.outer(x - self.W_enc @ f, f)
        
        # 更新表征
        rep = self.reps[idx]
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        self.reps[idx] = new_rep
        
        # 记录
        recon_err = np.linalg.norm(rep - x)
        self.errors.append(recon_err)
        
        self.prev_f = f
        self.prev_rep_idx = idx
        
        return recon_err
    
    def diagnose(self):
        """诊断"""
        if len(self.selection_log) < 10:
            return "数据不足"
        
        # 分析选择
        exploits = [idx for mode, idx in self.selection_log[-50:] if mode == 'exploit']
        explores = [idx for mode, idx in self.selection_log[-50:] if mode == 'explore']
        
        counts = [exploits.count(i) for i in range(len(self.reps))]
        
        # 预测器质量
        pred_scores = [p.get_score() for p in self.predictors]
        
        return {
            'selection_counts': counts,
            'explore_ratio': len(explores) / 50,
            'predictor_scores': pred_scores
        }


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


def test_fixed():
    """测试修复"""
    print("="*60)
    print("Test: Fixed System")
    print("="*60)
    
    results = {'fixed': [], 'old': []}
    
    for run in range(20):
        # 修复后的系统
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = FixedSystem(10, 5, 10, 0.01, 0.1)
        
        for _ in range(500):
            sys.forward(env.gen())
        
        results['fixed'].append(np.mean(sys.errors[-100:]))
        
        # 旧系统 (基于重构)
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        class OldSystem:
            def __init__(self):
                self.reps = [np.random.randn(10) for _ in range(3)]
                self.errors = []
            
            def forward(self, x):
                idx = np.argmax([-np.linalg.norm(r - x) for r in self.reps])
                err = np.linalg.norm(self.reps[idx] - x)
                self.reps[idx] += 0.01 * (x - self.reps[idx])
                self.errors.append(err)
                return err
        
        old = OldSystem()
        for _ in range(500):
            old.forward(env.gen())
        
        results['old'].append(np.mean(old.errors[-100:]))
    
    print(f"\nFixed (预测选择): {np.mean(results['fixed']):.4f} +/- {np.std(results['fixed']):.4f}")
    print(f"Old (重构选择): {np.mean(results['old']):.4f} +/- {np.std(results['old']):.4f}")
    
    diff = (np.mean(results['old']) - np.mean(results['fixed'])) / np.mean(results['old']) * 100
    print(f"\n改进: {diff:+.1f}%")
    
    if diff > 0:
        print("修复成功!")
    else:
        print("修复无效")


def test_selection():
    """测试选择机制"""
    print("\n" + "="*60)
    print("Test: Selection Mechanism")
    print("="*60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = FixedSystem(10, 5, 10, 0.01, 0.1)
    
    for _ in range(500):
        sys.forward(env.gen())
    
    diag = sys.diagnine()
    print(f"\n诊断: {diag}")


# ==================== Main ====================
test_fixed()
test_selection()
print("\nDone!")
