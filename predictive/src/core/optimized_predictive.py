# -*- coding: utf-8 -*-
"""
FCRS-v5 Predictive: 核心模块 (优化版 v5.3)

优化内容 (第一阶段):
1. 多步预测评估 (multi_step_horizon)
2. 动态权衡机制 (adaptive alpha)
3. 双目标联合损失
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ==================== 配置 ====================

class Config:
    """统一配置"""
    INPUT_DIM = 10
    COMPRESS_DIM = 3
    POOL_CAPACITY = 10
    
    LEARNING_RATE = 0.01
    EXPLORATION_RATE = 0.1
    
    # 优化参数
    MULTI_STEP_HORIZON = 3  # 多步预测视野
    Xavier_FACTOR = 1.0
    
    # 双目标损失权重
    LAMBDA_RECON = 0.3
    LAMBDA_PRED = 0.7
    
    @classmethod
    def set(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


# ==================== 工具函数 ====================

def xavier_init(fan_in: int, fan_out: int, factor: float = 1.0) -> np.ndarray:
    limit = factor * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def stable_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    return x / norm


# ==================== 优化预测器 ====================

class OptimizedPredictor:
    """
    优化预测器: 带L2正则和历史滚动窗口
    """
    
    def __init__(self, dim: int, lr: float = 0.01, l2_reg: float = 0.001):
        self.dim = dim
        self.lr = lr
        self.l2_reg = l2_reg
        
        self.W = xavier_init(dim, dim)
        self.b = np.zeros(dim)
        
        # 历史记录 (有限窗口)
        self.window_size = 100
        self.state_history = []
        self.error_history = []
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        out = state @ self.W + self.b
        out = stable_normalize(out)
        out = np.maximum(0, out)
        out = stable_normalize(out)
        return out
    
    def predict_multi_step(self, state: np.ndarray, horizon: int) -> List[np.ndarray]:
        """多步预测"""
        trajectory = [state]
        current = state
        
        for _ in range(horizon):
            current = self.predict(current)
            trajectory.append(current)
        
        return trajectory
    
    def update(self, state_curr: np.ndarray, state_next: np.ndarray):
        pred = self.predict(state_curr)
        error = state_next - pred
        
        # L2正则
        reg_term = self.l2_reg * self.W
        
        # 梯度更新
        self.W -= self.lr * (np.outer(state_curr, error) + reg_term)
        self.b -= self.lr * error
        
        # 记录
        self.state_history.append((state_curr, state_next))
        self.error_history.append(np.linalg.norm(error))
        
        # 滑动窗口
        if len(self.state_history) > self.window_size:
            self.state_history = self.state_history[-self.window_size:]
        if len(self.error_history) > self.window_size:
            self.error_history = self.error_history[-self.window_size:]
    
    def get_mean_error(self) -> float:
        if len(self.error_history) == 0:
            return 1.0
        return np.mean(self.error_history[-10:])
    
    def get_multi_step_error(self, horizon: int) -> float:
        """计算多步累计预测误差"""
        if len(self.state_history) < horizon:
            return 1.0
        
        total_error = 0
        for i in range(min(horizon, len(self.state_history))):
            curr, target = self.state_history[-(i+1)]
            pred = self.predict(curr)
            total_error += np.linalg.norm(pred - target)
        
        return total_error / horizon


# ==================== 优化表征 ====================

class OptimizedRepresentation:
    """优化表征: 带多步预测能力"""
    
    def __init__(self, dim: int, lr: float = 0.01):
        self.dim = dim
        
        self.vector = xavier_init(dim, 1).flatten()
        self.vector = stable_normalize(self.vector)
        
        self.predictor = OptimizedPredictor(dim, lr)
    
    def get_single_step_error(self) -> float:
        return self.predictor.get_mean_error()
    
    def get_multi_step_error(self, horizon: int) -> float:
        return self.predictor.get_multi_step_error(horizon)


# ==================== 优化表征池 ====================

class OptimizedPool:
    """
    优化表征池: 
    - 多步预测评估
    - 动态权衡
    - 双目标训练
    """
    
    def __init__(self, dim: int, capacity: int, lr: float = 0.01):
        self.dim = dim
        self.capacity = capacity
        self.lr = lr
        
        self.reps: List[OptimizedRepresentation] = []
        self.selection_counts = []
        
        # 环境稳定性跟踪
        self.stability_history = []
    
    def add(self, rep: OptimizedRepresentation):
        if len(self.reps) < self.capacity:
            self.reps.append(rep)
    
    def initialize(self, n: int = 3):
        for _ in range(n):
            rep = OptimizedRepresentation(self.dim, self.lr)
            self.add(rep)
    
    def compute_alpha(self) -> float:
        """
        动态计算alpha:
        - 环境稳定时，alpha低，更看重未来预测
        - 环境变化时，alpha高，更看重当前匹配
        """
        if len(self.stability_history) < 10:
            return 0.5
        
        # 基于预测误差方差计算稳定性
        recent = self.stability_history[-10:]
        variance = np.var(recent) if recent else 0
        
        # 方差大=不稳定=高alpha
        alpha = np.clip(np.exp(-variance * 10), 0.2, 0.8)
        return alpha
    
    def select_optimized(self, x: np.ndarray, exploration: float = 0.1,
                        horizon: int = 3, use_dual_objective: bool = True) -> int:
        """
        优化选择策略:
        1. 多步预测评估
        2. 动态权衡
        3. 双目标
        """
        # ε-贪心
        if np.random.random() < exploration:
            idx = np.random.randint(len(self.reps))
            self.selection_counts.append(idx)
            return idx
        
        # 计算每个表征的得分
        scores = []
        alpha = self.compute_alpha()  # 动态alpha
        
        for rep in self.reps:
            # 当前匹配度 (重构误差的负值)
            recon_error = np.linalg.norm(rep.vector - x)
            recon_score = 1.0 / (recon_error + 0.1)
            
            # 未来预测能力 (多步预测误差)
            if use_dual_objective:
                pred_error = rep.get_multi_step_error(horizon)
                pred_score = 1.0 / (pred_error + 0.1)
            else:
                pred_error = rep.get_single_step_error()
                pred_score = 1.0 / (pred_error + 0.1)
            
            # 动态权衡
            score = alpha * recon_score + (1 - alpha) * pred_score
            scores.append(score)
        
        best_idx = np.argmax(scores)
        self.selection_counts.append(best_idx)
        
        return best_idx
    
    def update(self, idx: int, x: np.ndarray, state_curr: np.ndarray, state_next: np.ndarray,
              lambda_recon: float = 0.3, lambda_pred: float = 0.7):
        """双目标更新"""
        rep = self.reps[idx]
        
        # 重构目标更新
        rep.vector += self.lr * lambda_recon * (x - rep.vector)
        rep.vector = stable_normalize(rep.vector)
        
        # 预测目标更新
        rep.predictor.update(state_curr, state_next)
        
        # 记录稳定性
        pred_err = rep.predictor.get_mean_error()
        self.stability_history.append(pred_err)
        
        # 滑动窗口
        if len(self.stability_history) > 100:
            self.stability_history = self.stability_history[-100:]


# ==================== 优化系统 ====================

class OptimizedSystem:
    """优化预测导向系统"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        self.pool = OptimizedPool(
            dim=self.config.INPUT_DIM,
            capacity=self.config.POOL_CAPACITY,
            lr=self.config.LEARNING_RATE
        )
        self.pool.initialize(3)
        
        self.state_history = []
        self.recon_errors = []
        self.pred_errors = []
    
    def step(self, x: np.ndarray, explore: float = None,
            horizon: int = None, dual_objective: bool = True) -> dict:
        explore = explore or self.config.EXPLORATION_RATE
        horizon = horizon or self.config.MULTI_STEP_HORIZON
        
        # 获取当前状态
        if len(self.state_history) > 0:
            state_curr = self.state_history[-1]
        else:
            state_curr = np.zeros(self.config.INPUT_DIM)
        
        # 优化选择
        idx = self.pool.select_optimized(x, exploration=explore, 
                                         horizon=horizon, 
                                         use_dual_objective=dual_objective)
        rep = self.pool.reps[idx]
        
        # 重构误差
        recon_err = np.linalg.norm(rep.vector - x)
        self.recon_errors.append(recon_err)
        
        # 状态更新
        state_next = stable_normalize(x)
        
        # 双目标更新
        self.pool.update(idx, x, state_curr, state_next,
                        lambda_recon=self.config.LAMBDA_RECON,
                        lambda_pred=self.config.LAMBDA_PRED)
        
        # 记录
        self.state_history.append(state_next)
        
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return {
            "recon_error": recon_err,
            "pred_error": rep.predictor.get_mean_error(),
            "selected_idx": idx
        }
    
    def run(self, env, steps: int, explore: float = None, 
           horizon: int = None, dual_objective: bool = True) -> dict:
        for _ in range(steps):
            x = env.generate_input()
            self.step(x, explore, horizon, dual_objective)
        return self.get_statistics()
    
    def get_statistics(self) -> dict:
        recon = np.array(self.recon_errors[-100:]) if self.recon_errors else np.array([0])
        return {
            "mean_recon_error": float(np.mean(recon)),
            "std_recon_error": float(np.std(recon)),
            "pool_size": len(self.pool.reps)
        }


# ==================== 基线系统 ====================

class ReconstructionBaseline2:
    """重构基线: 只优化重构误差"""
    
    def __init__(self, dim: int, capacity: int, lr: float = 0.01):
        self.dim = dim
        self.reps = [xavier_init(dim, 1).flatten() for _ in range(capacity)]
        for r in self.reps:
            r /= (np.linalg.norm(r) + 1e-8)
        self.lr = lr
        self.errors = []
    
    def step(self, x: np.ndarray) -> float:
        idx = np.argmin([np.linalg.norm(r - x) for r in self.reps])
        err = np.linalg.norm(self.reps[idx] - x)
        self.errors.append(err)
        
        # 只更新重构
        self.reps[idx] += self.lr * (x - self.reps[idx])
        self.reps[idx] /= (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        return err


class RandomBaseline3:
    """随机基线"""
    
    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.reps = [xavier_init(dim, 1).flatten() for _ in range(capacity)]
        for r in self.reps:
            r /= (np.linalg.norm(r) + 1e-8)
        self.errors = []
    
    def step(self, x: np.ndarray) -> float:
        idx = np.random.randint(len(self.reps))
        err = np.linalg.norm(self.reps[idx] - x)
        self.errors.append(err)
        return err


class SimpleEnv:
    """简单环境"""
    
    def __init__(self, dim: int = 10, n_classes: int = 5, noise: float = 0.3):
        self.dim = dim
        self.n_classes = n_classes
        self.noise = noise
        
        np.random.seed(42)
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def generate_input(self) -> np.ndarray:
        c = self.centers[np.random.randint(0, self.n_classes)]
        return c + np.random.randn(self.dim) * self.noise


# ==================== 测试 ====================

def test_optimization():
    """测试优化效果"""
    print("="*60)
    print("Optimization Test")
    print("="*60)
    
    results = {
        'optimized': [],
        'recon': [],
        'random': []
    }
    
    for run in range(20):
        np.random.seed(run * 100)
        
        # 优化系统 (多步预测 + 动态权衡 + 双目标)
        Config.set(LEARNING_RATE=0.01, EXPLORATION_RATE=0.1, 
                  MULTI_STEP_HORIZON=3, LAMBDA_RECON=0.3, LAMBDA_PRED=0.7)
        env = SimpleEnv(10, 5, 0.3)
        sys = OptimizedSystem()
        
        for _ in range(500):
            x = env.generate_input()
            sys.step(x, horizon=3, dual_objective=True)
        
        results['optimized'].append(np.mean(sys.recon_errors[-100:]))
        
        # 重构基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        baseline = ReconstructionBaseline2(10, 3, 0.01)
        
        for _ in range(500):
            x = env.generate_input()
            baseline.step(x)
        
        results['recon'].append(np.mean(baseline.errors[-100:]))
        
        # 随机基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        rand = RandomBaseline3(10, 3)
        
        for _ in range(500):
            x = env.generate_input()
            rand.step(x)
        
        results['random'].append(np.mean(rand.errors[-100:]))
    
    print(f"\nOptimized:     {np.mean(results['optimized']):.4f} +/- {np.std(results['optimized']):.4f}")
    print(f"Reconstruction: {np.mean(results['recon']):.4f} +/- {np.std(results['recon']):.4f}")
    print(f"Random:        {np.mean(results['random']):.4f} +/- {np.std(results['random']):.4f}")
    
    opt_vs_recon = (np.mean(results['recon']) - np.mean(results['optimized'])) / np.mean(results['recon']) * 100
    opt_vs_random = (np.mean(results['random']) - np.mean(results['optimized'])) / np.mean(results['random']) * 100
    
    print(f"\nOptimized vs Reconstruction: {opt_vs_recon:+.1f}%")
    print(f"Optimized vs Random: {opt_vs_random:+.1f}%")
    
    if opt_vs_recon > 5:
        print("\n[OK] Optimized > Reconstruction!")
    elif opt_vs_recon > 0:
        print("\n[~] Optimized > Reconstruction (marginal)")
    else:
        print("\n[X] Optimized <= Reconstruction")


def test_multi_step_horizon():
    """测试不同horizon"""
    print("\n" + "="*60)
    print("Multi-Step Horizon Test")
    print("="*60)
    
    for horizon in [1, 2, 3, 5]:
        errors = []
        
        for run in range(10):
            np.random.seed(run * 100)
            
            Config.set(LEARNING_RATE=0.01, EXPLORATION_RATE=0.1, MULTI_STEP_HORIZON=horizon)
            env = SimpleEnv(10, 5, 0.3)
            sys = OptimizedSystem()
            
            for _ in range(500):
                x = env.generate_input()
                sys.step(x, horizon=horizon, dual_objective=True)
            
            errors.append(np.mean(sys.recon_errors[-100:]))
        
        print(f"horizon={horizon}: {np.mean(errors):.4f}")


def test_dual_objective():
    """测试双目标效果"""
    print("\n" + "="*60)
    print("Dual Objective Test")
    print("="*60)
    
    # 只用预测目标
    Config.set(LAMBDA_RECON=0.0, LAMBDA_PRED=1.0)
    env = SimpleEnv(10, 5, 0.3)
    sys1 = OptimizedSystem()
    
    for _ in range(500):
        x = env.generate_input()
        sys1.step(x, dual_objective=True)
    
    print(f"Pred only: {np.mean(sys1.recon_errors[-100:]):.4f}")
    
    # 双目标
    Config.set(LAMBDA_RECON=0.3, LAMBDA_PRED=0.7)
    env = SimpleEnv(10, 5, 0.3)
    sys2 = OptimizedSystem()
    
    for _ in range(500):
        x = env.generate_input()
        sys2.step(x, dual_objective=True)
    
    print(f"Dual: {np.mean(sys2.recon_errors[-100:]):.4f}")
    
    # 只用重构目标
    Config.set(LAMBDA_RECON=1.0, LAMBDA_PRED=0.0)
    env = SimpleEnv(10, 5, 0.3)
    sys3 = OptimizedSystem()
    
    for _ in range(500):
        x = env.generate_input()
        sys3.step(x, dual_objective=True)
    
    print(f"Recon only: {np.mean(sys3.recon_errors[-100:]):.4f}")


if __name__ == "__main__":
    test_optimization()
    test_multi_step_horizon()
    test_dual_objective()
    print("\nDone!")
