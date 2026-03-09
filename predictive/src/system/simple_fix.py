"""
FCRS-v5: 简化修复版
只修复核心问题
"""

import numpy as np


class SimpleFixed:
    """简化修复版"""
    
    def __init__(self, dim=10, capacity=3, lr=0.01):
        self.dim = dim
        self.capacity = capacity
        self.lr = lr
        
        # 表征
        self.reps = [np.random.randn(dim) for _ in range(capacity)]
        
        # 预测器 (每个表征一个)
        self.predictors = [np.eye(dim) * 0.5 for _ in range(capacity)]
        
        self.errors = []
        self.prev_rep = None
    
    def select(self, x, explore=0.1):
        """基于预测选择 - 简化版"""
        # ε-贪心
        if np.random.random() < explore:
            return np.random.randint(len(self.reps))
        
        # 计算每个表征的"预测价值"
        scores = []
        for i, (rep, W) in enumerate(zip(self.reps, self.predictors)):
            # 表征对当前输入的"预测能力"
            # 简化: 用表征本身与输入的关系
            
            # 方法1: 重构误差 (越小越好)
            recon = np.linalg.norm(rep - x)
            
            # 方法2: 表征的"历史预测能力" (如果有)
            pred_ability = 1.0  # 默认
            
            # 综合
            score = -recon + pred_ability * 0.1
            scores.append(score)
        
        return np.argmax(scores)
    
    def forward(self, x, explore=0.1):
        # 选择
        idx = self.select(x, explore)
        rep = self.reps[idx]
        
        # 重构误差
        err = np.linalg.norm(rep - x)
        self.errors.append(err)
        
        # 更新表征
        self.reps[idx] = rep + self.lr * (x - rep)
        
        # 更新预测器 (简化)
        if self.prev_rep is not None:
            # 预测: prev_rep -> x
            pred = self.prev_rep @ self.predictors[idx]
            error = x - pred
            
            # 更新预测器
            self.predictors[idx] += self.lr * np.outer(self.prev_rep, error)
        
        self.prev_rep = rep.copy()
        
        return err


class Baseline:
    """简单基线"""
    def __init__(self, dim=10, capacity=3, lr=0.01):
        self.dim = dim
        self.reps = [np.random.randn(dim) for _ in range(capacity)]
        self.errors = []
    
    def forward(self, x):
        # 只选最近邻
        idx = np.argmin([np.linalg.norm(r - x) for r in self.reps])
        err = np.linalg.norm(self.reps[idx] - x)
        self.errors.append(err)
        
        self.reps[idx] += 0.01 * (x - self.reps[idx])
        
        return err


def test_simple():
    """简化测试"""
    print("="*60)
    print("Simple Test")
    print("="*60)
    
    results = {'simple': [], 'baseline': []}
    
    for run in range(20):
        # 简化修复版
        np.random.seed(run * 100)
        s = SimpleFixed(10, 3, 0.01)
        
        for _ in range(500):
            x = np.random.randn(10) * 2 + np.random.randn(10) * 0.3
            s.forward(x)
        
        results['simple'].append(np.mean(s.errors[-100:]))
        
        # 基线
        np.random.seed(run * 100)
        b = Baseline(10, 3, 0.01)
        
        for _ in range(500):
            x = np.random.randn(10) * 2 + np.random.randn(10) * 0.3
            b.forward(x)
        
        results['baseline'].append(np.mean(b.errors[-100:]))
    
    print(f"\nSimple Fixed: {np.mean(results['simple']):.4f} +/- {np.std(results['simple']):.4f}")
    print(f"Baseline:     {np.mean(results['baseline']):.4f} +/- {np.std(results['baseline']):.4f}")
    
    diff = (np.mean(results['baseline']) - np.mean(results['simple'])) / np.mean(results['baseline']) * 100
    print(f"\n差异: {diff:+.1f}%")


def test_explore():
    """测试探索率"""
    print("\n" + "="*60)
    print("Explore Rate Test")
    print("="*60)
    
    for exp in [0.0, 0.05, 0.1, 0.2, 0.3]:
        np.random.seed(42)
        s = SimpleFixed(10, 3, 0.01)
        
        for _ in range(500):
            x = np.random.randn(10) * 2 + np.random.randn(10) * 0.3
            s.forward(x, explore=exp)
        
        print(f"exp={exp}: {np.mean(s.errors[-100:]):.4f}")


# ==================== Main ====================
test_simple()
test_explore()
print("\nDone!")
