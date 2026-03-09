# -*- coding: utf-8 -*-
"""
FCRS-v5.3.1 优化版 v3: 进一步调优
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


class OptimizedAgentV3:
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',
                 n_reps: int = 5, lr: float = 0.1,  # 更高学习率
                 explore: float = 0.2):  # 更低探索
        
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
        
        gamma = 0.9
        for a in range(self.action_dim):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward + gamma * np.max(q_values) - q_values[a]) * rep
                self.b_policy[a] += self.lr * (reward + gamma * np.max(q_values) - q_values[a])
        
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        self.metrics.add_recon_error(np.linalg.norm(self.reps[rep_idx] - state))


def run_experiment(selection_mode: str, n_episodes: int = 300, n_runs: int = 10) -> Dict:
    results = {'success_rates': [], 'mean_rewards': [], 'mean_recon_errors': []}
    
    for run in range(n_runs):
        np.random.seed(run * 100)
        
        env = GridWorldEnv(size=5, n_obstacles=3)
        
        agent = OptimizedAgentV3(
            state_dim=env.input_dim,
            action_dim=env.action_dim,
            selection_mode=selection_mode,
            n_reps=5,
            lr=0.1,
            explore=0.2
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
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_recon_error': np.mean(results['mean_recon_errors'])
    }


def main():
    print("="*60)
    print("FCRS-v5.3.1 Optimized v3")
    print("="*60)
    
    modes = ['prediction', 'reconstruction', 'random']
    all_results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        all_results[mode] = run_experiment(mode, n_episodes=300, n_runs=10)
        print(f"  Success: {all_results[mode]['success_rate']:.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = all_results[mode]
        print(f"{mode:<20} {r['success_rate']:.1f}%  {r['mean_reward']:.2f}")
    
    pred = all_results['prediction']
    recon = all_results['reconstruction']
    
    print(f"\nPrediction vs Reconstruction: {pred['success_rate'] - recon['success_rate']:+.1f}%")
    
    if pred['success_rate'] >= 60:
        print("\n[OK] Target 60%+ achieved!")


if __name__ == "__main__":
    main()
