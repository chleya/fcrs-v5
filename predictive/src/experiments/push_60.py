# -*- coding: utf-8 -*-
"""
FCRS-v5.3.2 冲击60%目标
基于消融发现: 表征池贡献最大
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


class OptimizedAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',
                 n_reps: int = 5, lr: float = 0.05, explore: float = 0.3,
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


def run_exp(selection_mode: str, n_reps: int = 5, lr: float = 0.05, 
           explore: float = 0.3, n_episodes: int = 200, seed: int = 100) -> float:
    np.random.seed(seed)
    
    env = GridWorldEnv(size=5, n_obstacles=3)
    
    agent = OptimizedAgent(
        state_dim=env.input_dim,
        action_dim=env.action_dim,
        selection_mode=selection_mode,
        n_reps=n_reps,
        lr=lr,
        explore=explore,
        seed=seed
    )
    
    for ep in range(n_episodes):
        state = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 50:
            rep_idx = agent.select_representation(state)
            action = agent.select_action(agent.reps[rep_idx])
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, rep_idx)
            state = next_state
            steps += 1
        
        agent.metrics.add_episode_result(0, steps, done)
    
    return agent.metrics.get_success_rate() * 100


def main():
    print("="*60)
    print("FCRS-v5.3.2 冲击60%")
    print("="*60)
    
    # 基于消融: n_reps+lr组合最优
    configs = [
        # 扩大表征池
        ("n_reps=6, lr=0.05", 6, 0.05, 0.3, 200),
        ("n_reps=7, lr=0.05", 7, 0.05, 0.3, 200),
        ("n_reps=8, lr=0.05", 8, 0.05, 0.3, 200),
        # 调整学习率
        ("n_reps=5, lr=0.06", 5, 0.06, 0.3, 200),
        ("n_reps=5, lr=0.07", 5, 0.07, 0.3, 200),
        # 增加训练轮数
        ("n_reps=5, lr=0.05, ep=300", 5, 0.05, 0.3, 300),
        ("n_reps=6, lr=0.05, ep=300", 6, 0.05, 0.3, 300),
        # 组合
        ("n_reps=6, lr=0.06, ep=250", 6, 0.06, 0.3, 250),
    ]
    
    results = []
    for name, n_reps, lr, exp, ep in configs:
        print(f"\nTesting: {name}")
        pred = run_exp('prediction', n_reps, lr, exp, ep)
        recon = run_exp('reconstruction', n_reps, lr, exp, ep)
        results.append((name, pred, recon, pred - recon))
        print(f"  Prediction: {pred:.1f}%, Reconstruction: {recon:.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Config':<25} {'Pred':<10} {'Recon':<10} {'Diff':<10}")
    print("-"*60)
    
    for name, pred, recon, diff in sorted(results, key=lambda x: -x[1]):
        print(f"{name:<25} {pred:<10.1f} {recon:<10.1f} {diff:<10.1f}")
    
    best = max(results, key=lambda x: x[1])
    print(f"\nBest: {best[0]} = {best[1]}%")


if __name__ == "__main__":
    main()
