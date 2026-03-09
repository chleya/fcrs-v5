# -*- coding: utf-8 -*-
"""
FCRS-v5 高维输入测试 - 使用CartPole (连续状态)
"""

import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Warning: gymnasium not available")


class FCRSHighDim:
    """FCRS高维版本"""
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        
        # 表征池
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        
        # 策略
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


def run_cartpole(AgentClass, n_ep=500):
    if not HAS_GYM:
        print("gymnasium not available")
        return 0
    
    np.random.seed(100)
    env = gym.make('CartPole-v1')
    env.reset(seed=100)
    
    # 获取状态/动作维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = AgentClass(state_dim, action_dim)
    
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Training for {n_ep} episodes...")
    
    rewards = []
    
    for ep in range(n_ep):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action, idx = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        rewards.append(total_reward)
        
        if ep % 50 == 0:
            print(f"  Ep {ep}: reward={total_reward:.1f}")
    
    env.close()
    
    # 测试
    test_rewards = []
    for _ in range(20):
        state, _ = env.reset()
        total = 0
        for _ in range(500):
            action, _ = agent.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        test_rewards.append(total)
    
    return np.mean(test_rewards)


def main():
    print("="*60)
    print("FCRS High-Dim Input Test (CartPole)")
    print("="*60)
    
    if not HAS_GYM:
        print("Please install: pip install gymnasium")
        return
    
    print("\n[FCRS Prediction]")
    fcrs_score = run_cartpole(FCRSHighDim, n_ep=500)
    print(f"  Test score: {fcrs_score:.1f}")
    
    print("\n[Random]")
    random_score = run_cartpole(RandomAgent, n_ep=500)
    print(f"  Test score: {random_score:.1f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"FCRS:   {fcrs_score:.1f}")
    print(f"Random: {random_score:.1f}")
    
    if fcrs_score > random_score:
        print(f"\nFCRS outperforms Random by {fcrs_score - random_score:+.1f}")
    else:
        print("\nCartPole is simple; FCRS may not show advantage")


if __name__ == "__main__":
    main()
