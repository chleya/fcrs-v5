# -*- coding: utf-8 -*-
"""
FCRS vs RL基线 - 真正的连续状态网格世界

区别于one-hot离散版本：
- 状态是连续坐标 [0, 4] x [0, 4]
- 无法精确枚举每个状态
- 这才是FCRS的目标场景
"""

import numpy as np


class ContinuousGridWorld:
    """连续状态网格世界"""
    def __init__(self, size=5, n_obst=3, seed=42):
        self.size = size
        np.random.seed(seed)
        
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([4.0, 4.0])
        
        # 障碍物 (连续位置)
        self.obstacles = []
        while len(self.obstacles) < n_obst:
            pos = np.random.uniform(0.5, size-0.5, 2)
            if np.linalg.norm(pos - self.start) > 1.5 and np.linalg.norm(pos - self.goal) > 1.5:
                self.obstacles.append(pos)
        
        self.pos = self.start.copy()
        self.state_dim = 2
    
    def reset(self):
        self.pos = self.start.copy()
        return self.pos.copy()
    
    def step(self, action):
        moves = [(0, 0.4), (0, -0.4), (-0.4, 0), (0.4, 0)]
        dx, dy = moves[action]
        
        new_pos = self.pos + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0, self.size - 0.01)
        
        # 障碍物碰撞
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) < 0.6:
                return self.pos.copy(), -1.0, False
        
        self.pos = new_pos
        
        # 目标检测
        if np.linalg.norm(self.pos - self.goal) < 0.5:
            return self.pos.copy(), 10.0, True
        
        return self.pos.copy(), -0.1, False


class FCRSContinuous:
    """FCRS连续状态版"""
    def __init__(self, state_dim=2, action_dim=4, n_reps=5, lr=0.05):
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


class LinearQContinuous:
    """线性Q学习 (连续状态)"""
    def __init__(self, state_dim=2, action_dim=4, lr=0.1):
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
    env = ContinuousGridWorld(5, 3)
    agent = AgentClass(2, 4)
    
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
    print("FCRS vs RL Baselines - Continuous State Grid World")
    print("="*60)
    print("State: 2D continuous [0,4] x [0,4]")
    print("Task: Navigate to (4,4) avoiding obstacles\n")
    
    results = {}
    
    for name, cls in [("FCRS", FCRSContinuous), 
                      ("Linear Q", LinearQContinuous), 
                      ("Random", RandomAgent)]:
        print(f"[{name}]", end=" ")
        rate = run_exp(cls)
        results[name] = rate
        print(f"Success: {rate:.0f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        bar = "=" * int(rate / 5)
        print(f"{name:<12}: {rate:5.1f}% {bar}")
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    fcrs = results["FCRS"]
    linear = results["Linear Q"]
    rnd = results["Random"]
    
    if fcrs > linear:
        print(f"FCRS outperforms Linear Q by {fcrs - linear:+.1f}%")
    elif fcrs > rnd:
        print(f"FCRS outperforms Random by {fcrs - rnd:+.1f}%")
    else:
        print("Note: Linear Q may be more suitable for continuous state tasks")


if __name__ == "__main__":
    main()
