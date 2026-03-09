# -*- coding: utf-8 -*-
"""
FCRS-v5.3.1 优化版: UCB选择 + MLP预测器

优化内容:
1. UCB选择策略 (替代ε-贪心)
2. MLP非线性预测器 (替代线性预测器)
3. 调优双目标损失权重
"""

import numpy as np
from typing import Dict, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from grid_world import GridWorldEnv
from metrics import Metrics


# ==================== MLP预测器 ====================

class MLPPredictor:
    """
    MLP非线性预测器
    输入: 表征
    输出: 预测的下一状态
    隐藏层: 32
    """
    
    def __init__(self, dim: int, hidden_dim: int = 32, lr: float = 0.01):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        
        # 初始化权重 (Xavier)
        self.W1 = np.random.randn(dim, hidden_dim) * np.sqrt(2.0 / dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(dim)
        
        # 历史
        self.error_history = []
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 隐藏层 (ReLU)
        h = np.tanh(state @ self.W1 + self.b1)
        # 输出层
        out = h @ self.W2 + self.b2
        # 归一化
        norm = np.linalg.norm(out)
        if norm > 1e-8:
            out = out / norm
        return out
    
    def update(self, state_curr: np.ndarray, state_next: np.ndarray):
        """反向传播更新 (简化版)"""
        # 前向
        h = np.tanh(state_curr @ self.W1 + self.b1)
        pred = h @ self.W2 + self.b2
        
        # 误差
        error = state_next - pred
        self.error_history.append(np.linalg.norm(error))
        
        # 简化梯度更新
        lr = self.lr
        
        # 输出层梯度
        d_pred = -error
        d_W2 = np.outer(h, d_pred)
        d_b2 = d_pred
        
        # 隐藏层梯度
        d_h = d_pred @ self.W2.T * (1 - h**2)
        d_W1 = np.outer(state_curr, d_h)
        d_b1 = d_h
        
        # 更新
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
        
        # 滑动窗口
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]
    
    def get_mean_error(self) -> float:
        if len(self.error_history) == 0:
            return 1.0
        return np.mean(self.error_history[-10:])


# ==================== UCB选择 ====================

class UCBSelector:
    """
    UCB (Upper Confidence Bound) 选择器
    平衡探索与利用
    """
    
    def __init__(self, n_arms: int, c: float = 1.0):
        self.n_arms = n_arms
        self.c = c  # 探索常数
        
        self.counts = [0] * n_arms
        self.values = [0.0] * n_arms
        self.total_counts = 0
    
    def select(self) -> int:
        """选择臂"""
        # 初始化：先尝试所有臂
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                self.counts[i] += 1
                self.total_counts += 1
                return i
        
        # UCB计算
        ucb_values = []
        for i in range(self.n_arms):
            # 置信上界
            bonus = self.c * np.sqrt(np.log(self.total_counts) / self.counts[i])
            ucb = self.values[i] + bonus
            ucb_values.append(ucb)
        
        # 选择最大UCB
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """更新臂的价值"""
        n = self.counts[arm]
        self.values[arm] = (self.values[arm] * n + reward) / (n + 1)
        self.counts[arm] += 1
        self.total_counts += 1


# ==================== 优化Agent ====================

class OptimizedAgent:
    """
    优化版Agent:
    - MLP预测器
    - UCB选择
    - 双目标训练
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',
                 n_reps: int = 3, lr: float = 0.01,
                 ucb_c: float = 1.0):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selection_mode = selection_mode
        self.n_reps = n_reps
        self.lr = lr
        
        # 表征池
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        
        # MLP预测器 (替代线性)
        self.predictors = [MLPPredictor(state_dim, hidden_dim=32, lr=lr) for _ in range(n_reps)]
        
        # UCB选择器
        self.ucb = UCBSelector(n_reps, c=ucb_c)
        
        # 策略网络
        self.W_policy = np.random.randn(state_dim, action_dim) * 0.1
        self.b_policy = np.zeros(action_dim)
        
        # 历史
        self.state_history = []
        self.selection_history = []
        
        # 指标
        self.metrics = Metrics()
    
    def select_representation(self, state: np.ndarray) -> int:
        """UCB + 预测选择"""
        if self.selection_mode == 'prediction':
            # 计算每个表征的预测得分
            scores = []
            for i in range(self.n_reps):
                pred_err = self.predictors[i].get_mean_error()
                # UCB值 = -预测误差 (越小越好)
                ucb_val = -pred_err
                scores.append(ucb_val)
            
            # UCB选择
            idx = self.ucb.select()
            
            # 更新UCB
            if len(self.state_history) > 0:
                # 用预测误差作为reward
                reward = -scores[idx]
                self.ucb.update(idx, reward)
        
        elif self.selection_mode == 'reconstruction':
            # 重构导向
            scores = [np.linalg.norm(r - state) for r in self.reps]
            idx = np.argmin(scores)
            # UCB更新
            reward = -scores[idx]
            self.ucb.update(idx, -reward)
        
        else:  # random
            idx = np.random.randint(self.n_reps)
        
        self.selection_history.append(idx)
        return idx
    
    def select_action(self, representation: np.ndarray, explore: float = 0.1) -> int:
        """基于表征选择动作"""
        q_values = representation @ self.W_policy + self.b_policy
        
        if np.random.random() < explore:
            return np.random.randint(self.action_dim)
        
        return np.argmax(q_values)
    
    def learn(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, rep_idx: int):
        """学习"""
        # 表征更新
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        # MLP预测器更新
        if len(self.state_history) > 0:
            prev = self.state_history[-1]
            self.predictors[rep_idx].update(prev, state)
        
        # 策略更新
        selected_rep = self.reps[rep_idx]
        q_values = selected_rep @ self.W_policy + self.b_policy
        target_q = reward
        
        for a in range(self.action_dim):
            if a == action:
                self.W_policy[:, a] += self.lr * (target_q - q_values[a]) * selected_rep
                self.b_policy[a] += self.lr * (target_q - q_values[a])
            else:
                self.W_policy[:, a] -= 0.01 * q_values[a] * selected_rep
        
        # 历史
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        # 指标
        recon_err = np.linalg.norm(self.reps[rep_idx] - state)
        self.metrics.add_recon_error(recon_err)


# ==================== 实验 ====================

def run_experiment(selection_mode: str, n_episodes: int = 100,
                   n_runs: int = 10) -> Dict:
    """运行实验"""
    results = {
        'success_rates': [],
        'mean_rewards': [],
        'mean_lengths': [],
        'mean_recon_errors': []
    }
    
    for run in range(n_runs):
        np.random.seed(run * 100)
        
        # 环境
        env = GridWorldEnv(size=5, n_obstacles=3)
        
        # Agent
        agent = OptimizedAgent(
            state_dim=env.input_dim,
            action_dim=env.action_dim,
            selection_mode=selection_mode,
            n_reps=3,
            lr=0.01,
            ucb_c=1.0
        )
        
        # 训练
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                # 表征选择
                rep_idx = agent.select_representation(state)
                
                # 动作选择
                action = agent.select_action(agent.reps[rep_idx], explore=0.2)
                
                # 执行
                next_state, reward, done = env.step(action)
                
                # 学习
                agent.learn(state, action, reward, next_state, rep_idx)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            agent.metrics.add_episode_result(total_reward, steps, done)
        
        # 结果
        results['success_rates'].append(agent.metrics.get_success_rate())
        results['mean_rewards'].append(agent.metrics.get_mean_reward())
        results['mean_lengths'].append(agent.metrics.get_mean_length())
        results['mean_recon_errors'].append(agent.metrics.get_mean_recon_error())
    
    return {
        'success_rate': np.mean(results['success_rates']) * 100,
        'std_success': np.std(results['success_rates']) * 100,
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_length': np.mean(results['mean_lengths']),
        'mean_recon_error': np.mean(results['mean_recon_errors'])
    }


def main():
    """主实验"""
    print("="*60)
    print("FCRS-v5.3.1 Optimized: UCB + MLP Predictor")
    print("="*60)
    
    modes = ['prediction', 'reconstruction', 'random']
    all_results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        all_results[mode] = run_experiment(mode, n_episodes=100, n_runs=10)
        print(f"  Success: {all_results[mode]['success_rate']:.1f}%")
    
    # 对比
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\n{'Mode':<20} {'Success':<10} {'Reward':<10} {'Recon Err':<12}")
    print("-"*60)
    
    for mode in modes:
        r = all_results[mode]
        print(f"{mode:<20} {r['success_rate']:<9.1f}% {r['mean_reward']:<10.2f} {r['mean_recon_error']:<12.4f}")
    
    # 核心对比
    pred = all_results['prediction']
    recon = all_results['reconstruction']
    
    print(f"\nPrediction vs Reconstruction:")
    print(f"  Success Rate: {pred['success_rate'] - recon['success_rate']:+.1f}%")
    
    if pred['success_rate'] >= 60:
        print("\n[OK] Target 60%+ achieved!")
    elif pred['success_rate'] > recon['success_rate']:
        print("\n[~] Improvement confirmed, target not reached")
    else:
        print("\n[X] No improvement")


if __name__ == "__main__":
    main()
