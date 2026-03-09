# -*- coding: utf-8 -*-
"""
FCRS-v5.3.1 消融实验
每个参数单独变化，测试贡献度
"""

import numpy as np
from typing import Dict
import sys
import os

core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)

from grid_world import GridWorldEnv
from metrics import Metrics


class LinearPredictor:
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


class AblationAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',
                 n_reps: int = 3, lr: float = 0.01, explore: float = 0.2,
                 seed: int = 100):
        
        np.random.seed(seed)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selection_mode = selection_mode
        self.n_reps = n_reps
        self.lr = lr
        self.explore = explore
        
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [LinearPredictor(state_dim, lr) for _ in range(n_reps)]
        
        self.W_policy = np.random.randn(state_dim, action_dim) * 0.1
        self.b_policy = np.zeros(action_dim)
        
        self.state_history = []
        self.metrics = Metrics()
    
    def select_representation(self, state: np.ndarray) -> int:
        if self.selection_mode == 'prediction':
            errors = [p.get_mean_error() for p in self.predictors]
            idx = np.argmin(errors)
        elif self.selection_mode == 'reconstruction':
            errors = [np.linalg.norm(r - state) for r in self.reps]
            idx = np.argmin(errors)
        else:
            idx = np.random.randint(self.n_reps)
        return idx
    
    def select_action(self, representation: np.ndarray) -> int:
        q_values = representation @ self.W_policy + self.b_policy
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim)
        return np.argmax(q_values)
    
    def learn(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, rep_idx: int):
        
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        if len(self.state_history) > 0:
            self.predictors[rep_idx].update(self.state_history[-1], state)
        
        rep = self.reps[rep_idx]
        q_values = rep @ self.W_policy + self.b_policy
        
        for a in range(self.action_dim):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward - q_values[a]) * rep
                self.b_policy[a] += self.lr * (reward - q_values[a])
            else:
                self.W_policy[:, a] -= 0.01 * q_values[a] * rep
        
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        self.metrics.add_recon_error(np.linalg.norm(self.reps[rep_idx] - state))


def run_ablation(selection_mode: str, n_reps: int = 3, lr: float = 0.01, 
                explore: float = 0.2, n_episodes: int = 200) -> float:
    """运行消融实验"""
    np.random.seed(100)
    
    env = GridWorldEnv(size=5, n_obstacles=3)
    
    agent = AblationAgent(
        state_dim=env.input_dim,
        action_dim=env.action_dim,
        selection_mode=selection_mode,
        n_reps=n_reps,
        lr=lr,
        explore=explore,
        seed=100
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
    
    return agent.metrics.get_success_rate() * 100


def main():
    print("="*60)
    print("FCRS-v5.3.1 Ablation Experiment")
    print("="*60)
    
    # 基线 (v5.3.0配置)
    baseline = run_ablation('prediction', n_reps=3, lr=0.01, explore=0.2)
    print(f"\n基线 (v5.3.0): {baseline:.1f}%")
    
    # 消融实验
    print("\n--- 消融实验 ---")
    
    # 1. 仅增加表征池
    r1 = run_ablation('prediction', n_reps=5, lr=0.01, explore=0.2)
    print(f"仅n_reps=5: {r1:.1f}% (+{r1-baseline:.1f}%)")
    
    # 2. 仅增大学习率
    r2 = run_ablation('prediction', n_reps=3, lr=0.05, explore=0.2)
    print(f"仅lr=0.05: {r2:.1f}% (+{r2-baseline:.1f}%)")
    
    # 3. 仅增加探索率
    r3 = run_ablation('prediction', n_reps=3, lr=0.01, explore=0.3)
    print(f"仅explore=0.3: {r3:.1f}% (+{r3-baseline:.1f}%)")
    
    # 4. 增加表征池+学习率
    r4 = run_ablation('prediction', n_reps=5, lr=0.05, explore=0.2)
    print(f"n_reps=5 + lr=0.05: {r4:.1f}% (+{r4-baseline:.1f}%)")
    
    # 5. 全部三个参数
    r5 = run_ablation('prediction', n_reps=5, lr=0.05, explore=0.3)
    print(f"全部优化 (v5.3.1): {r5:.1f}% (+{r5-baseline:.1f}%)")
    
    # 总结
    print("\n" + "="*60)
    print("消融实验总结")
    print("="*60)
    print(f"\n{'配置':<25} {'成功率':<10} {'提升':<10}")
    print("-"*50)
    print(f"{'基线 (v5.3.0)':<25} {baseline:.1f}% {'-':<10}")
    print(f"{'仅n_reps=5':<25} {r1:.1f}% {'+' + str(round(r1-baseline,1)) + '%':<10}")
    print(f"{'仅lr=0.05':<25} {r2:.1f}% {'+' + str(round(r2-baseline,1)) + '%':<10}")
    print(f"{'仅explore=0.3':<25} {r3:.1f}% {'+' + str(round(r3-baseline,1)) + '%':<10}")
    print(f"{'n_reps+lr':<25} {r4:.1f}% {'+' + str(round(r4-baseline,1)) + '%':<10}")
    print(f"{'全部 (v5.3.1)':<25} {r5:.1f}% {'+' + str(round(r5-baseline,1)) + '%':<10}")
    
    # 核心发现
    print("\n核心发现:")
    improvements = [r1-baseline, r2-baseline, r3-baseline]
    best = max(improvements)
    if best == r1-baseline:
        print("- 表征池(n_reps)贡献最大")
    elif best == r2-baseline:
        print("- 学习率(lr)贡献最大")
    else:
        print("- 探索率(explore)贡献最大")


if __name__ == "__main__":
    main()
