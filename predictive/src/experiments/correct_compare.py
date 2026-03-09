# -*- coding: utf-8 -*-
"""
FCRS vs RL基线 - 使用正确的v5.3.1实现
"""

import numpy as np
import sys
import os
sys.path.insert(0, 'F:/fcrs-v5/predictive/src/core')

from grid_world import GridWorld


class LinearPredictor:
    """正确的预测器实现 (来自v5.3.1)"""
    def __init__(self, dim, lr=0.01):
        self.dim = dim
        self.lr = lr
        self.W = np.eye(dim) * 0.5
        self.errors = []
    
    def predict(self, state):
        out = state @ self.W
        norm = np.linalg.norm(out)
        if norm > 1e-8:
            out = out / norm
        return out
    
    def update(self, sc, sn):
        pred = self.predict(sc)
        error = sn - pred
        self.errors.append(np.linalg.norm(error))
        self.W += self.lr * np.outer(sc, error)
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_mean_error(self):
        if not self.errors:
            return 1.0
        return np.mean(self.errors[-10:])


class FCRSCorrect:
    """正确的FCRS v5.3.1实现"""
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.05, explore=0.25):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        self.explore = explore
        
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [LinearPredictor(state_dim, lr) for _ in range(n_reps)]
        self.W = np.random.randn(state_dim, action_dim) * 0.3
        self.b = np.zeros(action_dim)
        self.history = []
    
    def act(self, state):
        # 使用平均预测误差选择表征
        errors = [p.get_mean_error() for p in self.predictors]
        idx = np.argmin(errors)
        
        # 表征归一化 (关键！)
        rep = self.reps[idx]
        rep = rep / (np.linalg.norm(rep) + 1e-8)
        
        q = rep @ self.W + self.b
        action = np.random.randint(self.action_dim) if np.random.random() < self.explore else np.argmax(q)
        return action, idx
    
    def update(self, state, action, reward, next_state, idx):
        # 表征更新 + 归一化
        self.reps[idx] += self.lr * (state - self.reps[idx])
        self.reps[idx] /= (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        # 预测器更新
        if self.history:
            self.predictors[idx].update(self.history[-1], state)
        
        # 策略更新
        rep = self.reps[idx]
        rep = rep / (np.linalg.norm(rep) + 1e-8)
        
        gamma = 0.95
        q = rep @ self.W + self.b
        
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * rep
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(state)


class LinearQAgent:
    def __init__(self, state_dim, action_dim, lr=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.explore = 0.3
    
    def act(self, state):
        q = state @ self.W + self.b
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim), 0
        return np.argmax(q), 0
    
    def update(self, state, action, reward, next_state, idx):
        q = state @ self.W + self.b
        q_next = np.max(next_state @ self.W + self.b)
        td = reward + 0.99 * q_next - q[action]
        self.W[:, action] += self.lr * td * state
        self.b[action] += self.lr * td


class RandomAgent:
    def __init__(self, *args, **kwargs):
        pass
    
    def act(self, state):
        return np.random.randint(4), 0
    
    def update(self, *args):
        pass


def run_exp(AgentClass, n_ep=200):
    np.random.seed(100)
    env = GridWorld(size=5, n_obstacles=3, seed=42)
    agent = AgentClass(25, 4)
    
    for _ in range(n_ep):
        state = env.reset()
        for _ in range(50):
            action, idx = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            if done:
                break
    
    # Test
    success = 0
    for _ in range(50):
        state = env.reset()
        for _ in range(50):
            action, _ = agent.act(state)
            state, reward, done = env.step(action)
            if done:
                success += 1
                break
    
    return success / 50 * 100


def main():
    print("="*60)
    print("FCRS v5.3.1 (Correct Implementation) vs RL Baselines")
    print("="*60)
    
    results = {}
    
    for name, cls in [("FCRS v5.3.1", FCRSCorrect), 
                      ("Linear Q", LinearQAgent), 
                      ("Random", RandomAgent)]:
        print(f"[{name}]", end=" ")
        rate = run_exp(cls)
        results[name] = rate
        print(f"Success: {rate:.0f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name}: {rate:.0f}%")


if __name__ == "__main__":
    main()
