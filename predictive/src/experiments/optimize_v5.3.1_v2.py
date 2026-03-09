# -*- coding: utf-8 -*-
"""
FCRS-v5.3.1 优化版 v2: 调优参数

调整内容:
1. 降低UCB探索常数
2. 增加训练轮数
3. 调整学习率
"""

import numpy as np
from typing import Dict
import sys
import os

# 添加core目录到路径
core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)

from grid_world import GridWorldEnv
from metrics import Metrics


# ==================== 线性预测器 (更稳定) ====================

class LinearPredictor:
    """线性预测器 - 更稳定"""
    
    def __init__(self, dim: int, lr: float = 0.01):
        self.dim = dim
        self.lr = lr
        
        self.W = np.eye(dim) * 0.5
        self.errors = []
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        out = state @ self.W
        norm = np.linalg.norm(out)
        if norm > 1e-8:
            out = out / norm
        return out
    
    def update(self, state_curr: np.ndarray, state_next: np.ndarray):
        pred = self.predict(state_curr)
        error = state_next - pred
        self.errors.append(np.linalg.norm(error))
        
        self.W += self.lr * np.outer(state_curr, error)
        
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_mean_error(self) -> float:
        if not self.errors:
            return 1.0
        return np.mean(self.errors[-10:])


# ==================== Agent ====================

class OptimizedAgentV2:
    """优化版Agent v2"""
    
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',
                 n_reps: int = 5,  # 增加表征数量
                 lr: float = 0.05,  # 增大学习率
                 explore: float = 0.3):  # 增加探索率
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selection_mode = selection_mode
        self.n_reps = n_reps
        self.lr = lr
        self.explore = explore
        
        # 表征池
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        
        # 线性预测器
        self.predictors = [LinearPredictor(state_dim, lr) for _ in range(n_reps)]
        
        # 策略网络
        self.W_policy = np.random.randn(state_dim, action_dim) * 0.1
        self.b_policy = np.zeros(action_dim)
        
        # 历史
        self.state_history = []
        
        # 指标
        self.metrics = Metrics()
    
    def select_representation(self, state: np.ndarray) -> int:
        """选择表征"""
        if self.selection_mode == 'prediction':
            # 选择历史预测误差最小的
            errors = [p.get_mean_error() for p in self.predictors]
            idx = np.argmin(errors)
        
        elif self.selection_mode == 'reconstruction':
            errors = [np.linalg.norm(r - state) for r in self.reps]
            idx = np.argmin(errors)
        
        else:  # random
            idx = np.random.randint(self.n_reps)
        
        return idx
    
    def select_action(self, representation: np.ndarray) -> int:
        """选择动作"""
        q_values = representation @ self.W_policy + self.b_policy
        
        # 贪心 + 探索
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim)
        
        return np.argmax(q_values)
    
    def learn(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, rep_idx: int):
        """学习"""
        # 表征更新
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        # 预测器更新
        if len(self.state_history) > 0:
            prev = self.state_history[-1]
            self.predictors[rep_idx].update(prev, state)
        
        # 策略更新
        rep = self.reps[rep_idx]
        q_values = rep @ self.W_policy + self.b_policy
        
        for a in range(self.action_dim):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward - q_values[a]) * rep
                self.b_policy[a] += self.lr * (reward - q_values[a])
            else:
                self.W_policy[:, a] -= 0.01 * q_values[a] * rep
        
        # 历史
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        # 指标
        recon_err = np.linalg.norm(self.reps[rep_idx] - state)
        self.metrics.add_recon_error(recon_err)


def run_experiment(selection_mode: str, n_episodes: int = 200,  # 增加训练轮数
                 n_runs: int = 10) -> Dict:
    """运行实验"""
    results = {'success_rates': [], 'mean_rewards': [], 'mean_recon_errors': []}
    
    for run in range(n_runs):
        np.random.seed(run * 100)
        
        env = GridWorldEnv(size=5, n_obstacles=3)
        
        agent = OptimizedAgentV2(
            state_dim=env.input_dim,
            action_dim=env.action_dim,
            selection_mode=selection_mode,
            n_reps=5,  # 5个表征
            lr=0.05,   # 更高学习率
            explore=0.3
        )
        
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                rep_idx = agent.select_representation(state)
                action = agent.select_action(agent.reps[rep_idx])
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, rep_idx)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            agent.metrics.add_episode_result(total_reward, steps, done)
        
        results['success_rates'].append(agent.metrics.get_success_rate())
        results['mean_rewards'].append(agent.metrics.get_mean_reward())
        results['mean_recon_errors'].append(agent.metrics.get_mean_recon_error())
    
    return {
        'success_rate': np.mean(results['success_rates']) * 100,
        'std_success': np.std(results['success_rates']) * 100,
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_recon_error': np.mean(results['mean_recon_errors'])
    }


def main():
    print("="*60)
    print("FCRS-v5.3.1 Optimized v2")
    print("="*60)
    
    modes = ['prediction', 'reconstruction', 'random']
    all_results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        all_results[mode] = run_experiment(mode, n_episodes=200, n_runs=10)
        print(f"  Success: {all_results[mode]['success_rate']:.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = all_results[mode]
        print(f"{mode:<20} {r['success_rate']:.1f}%  {r['mean_reward']:.2f}  {r['mean_recon_error']:.4f}")
    
    pred = all_results['prediction']
    recon = all_results['reconstruction']
    
    print(f"\nPrediction vs Reconstruction: {pred['success_rate'] - recon['success_rate']:+.1f}%")
    
    if pred['success_rate'] >= 60:
        print("\n[OK] Target 60%+ achieved!")
    else:
        print(f"\n[~] Current: {pred['success_rate']:.1f}%, Target: 60%")


if __name__ == "__main__":
    main()
