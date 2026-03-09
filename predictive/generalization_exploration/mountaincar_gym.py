# -*- coding: utf-8 -*-
"""
FCRS-v5 高维输入测试 - MountainCar (多步规划)
"""

import numpy as np
import gymnasium as gym


class FCRSMC:
    """FCRS for MountainCar"""
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.history = []
        self.explore = 0.3
    
    def act(self, state):
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - state) for p in self.predictors]
            idx = np.argmin(errors)
        else:
            idx = 0
        
        q = self.reps[idx] @ self.W + self.b
        action = np.random.randint(self.action_dim) if np.random.random() < self.explore else np.argmax(q)
        return action, idx
    
    def update(self, state, action, reward, next_state, idx):
        self.reps[idx] += self.lr * (state - self.reps[idx])
        
        if self.history:
            pred = self.history[-1] @ self.predictors[idx]
            self.predictors[idx] += self.lr * np.outer(self.history[-1], next_state - pred)
        
        q = self.reps[idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * self.reps[idx]
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(state)


class RandomAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
    
    def act(self, state):
        return np.random.randint(self.action_dim), 0
    
    def update(self, *args):
        pass


def run_mountaincar(AgentClass, n_ep=500):
    np.random.seed(100)
    env = gym.make('MountainCar-v0')
    env.reset(seed=100)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = AgentClass(state_dim, action_dim)
    
    print(f"State: {state_dim}D, Action: {action_dim}")
    print(f"Training {n_ep} episodes...")
    
    successes = 0
    
    for ep in range(n_ep):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(1000):
            action, idx = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            total_reward += reward
            
            if done:
                if terminated:
                    successes += 1
                break
        
        if ep % 100 == 0:
            print(f"  Ep {ep}: reward={total_reward:.1f}, success={successes}")
    
    env.close()
    
    # 测试
    test_success = 0
    for _ in range(50):
        state, _ = env.reset()
        for _ in range(1000):
            action, _ = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            if terminated:
                test_success += 1
                break
            if truncated:
                break
    
    return test_success / 50 * 100


def main():
    print("="*60)
    print("FCRS vs Random - MountainCar (gymnasium)")
    print("="*60)
    
    print("\n[FCRS Prediction]")
    fcrs_success = run_mountaincar(FCRSMC, n_ep=500)
    print(f"  Test success: {fcrs_success:.1f}%")
    
    print("\n[Random]")
    random_success = run_mountaincar(RandomAgent, n_ep=500)
    print(f"  Test success: {random_success:.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"FCRS:   {fcrs_success:.1f}%")
    print(f"Random: {random_success:.1f}%")
    
    if fcrs_success > random_success:
        print(f"\nFCRS outperforms by {fcrs_success - random_success:+.1f}%")
    elif random_success > fcrs_success:
        print(f"\nRandom outperforms by {random_success - fcrs_success:+.1f}%")


if __name__ == "__main__":
    main()
