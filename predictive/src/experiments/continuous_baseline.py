# -*- coding: utf-8 -*-
"""
FCRS vs 深度RL基线 - 连续状态网格世界

使用之前验证成功的连续状态网格世界
"""

import numpy as np


class GridWorldContinuous:
    """连续状态网格世界 (之前56%成功的版本)"""
    def __init__(self, size=5, n_obst=3, seed=42):
        self.size = size
        np.random.seed(seed)
        
        # 起点/终点
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([4.0, 4.0])
        
        # 障碍物
        self.obstacles = []
        while len(self.obstacles) < n_obst:
            pos = np.random.uniform(0, size-1, 2)
            if np.linalg.norm(pos - self.start) > 1 and np.linalg.norm(pos - self.goal) > 1:
                self.obstacles.append(pos)
        
        self.pos = self.start.copy()
    
    def reset(self):
        self.pos = self.start.copy()
        return self.pos.copy()
    
    def step(self, action):
        moves = [(0, 0.5), (0, -0.5), (-0.5, 0), (0.5, 0)]
        dx, dy = moves[action]
        
        new_pos = self.pos + np.array([dx, dy])
        new_pos = np.clip(new_pos, 0, self.size - 1)
        
        # 检查障碍物
        for obs in self.obstacles:
            if np.linalg.norm(new_pos - obs) < 0.8:
                return self.pos.copy(), -1.0, False
        
        self.pos = new_pos
        
        # 检查目标
        if np.linalg.norm(self.pos - self.goal) < 0.5:
            return self.pos.copy(), 10.0, True
        
        return self.pos.copy(), -0.1, False


class FCRSAgent:
    """FCRS预测导向"""
    def __init__(self, state_dim=2, action_dim=4, n_reps=5, lr=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        
        # 表征池
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        # 预测器
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        # 策略
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        
        self.history = []
        self.explore = 0.3
    
    def act(self, state):
        # 预测导向选择表征
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - state) for p in self.predictors]
            idx = np.argmin(errors)
        else:
            idx = 0
        
        # 动作选择
        q = self.reps[idx] @ self.W + self.b
        if np.random.random() < self.explore:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(q)
        
        return action, idx
    
    def update(self, state, action, reward, next_state, idx):
        # 表征更新
        self.reps[idx] += self.lr * (state - self.reps[idx])
        
        # 预测器更新
        if self.history:
            pred = self.history[-1] @ self.predictors[idx]
            self.predictors[idx] += self.lr * np.outer(self.history[-1], next_state - pred)
        
        # 策略更新
        q = self.reps[idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * self.reps[idx]
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(state)


class LinearQLearning:
    """线性Q学习 (无神经网络)"""
    def __init__(self, state_dim=2, action_dim=4, lr=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        # 线性Q函数: Q(s,a) = s @ W[:,a] + b[a]
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
        
        td_error = reward + 0.99 * q_next - q[action]
        
        self.W[:, action] += self.lr * td_error * state
        self.b[action] += self.lr * td_error


class RandomAgent:
    def __init__(self, *args, **kwargs):
        pass
    
    def act(self, state):
        return np.random.randint(4), 0
    
    def update(self, *args):
        pass


def run_experiment(AgentClass, n_ep=200):
    """运行实验"""
    np.random.seed(100)
    env = GridWorldContinuous(5, 3)
    agent = AgentClass(2, 4)
    
    # 训练
    for ep in range(n_ep):
        state = env.reset()
        for step in range(50):  # 最多50步
            action, idx = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            if done:
                break
    
    # 测试
    successes = 0
    for _ in range(50):
        state = env.reset()
        for step in range(50):
            action, _ = agent.act(state)
            state, reward, done = env.step(action)
            if done:
                successes += 1
                break
    
    return successes / 50 * 100


def main():
    print("="*60)
    print("FCRS vs RL Baselines (Continuous Grid World)")
    print("="*60)
    print("\n任务: 连续状态网格世界 [0,4] x [0,4]")
    print("目标: 到达 (4,4), 障碍物避让\n")
    
    results = {}
    
    # 运行实验
    for name, cls in [("FCRS (预测)", FCRSAgent), 
                      ("Linear Q", LinearQLearning), 
                      ("Random", RandomAgent)]:
        print(f"[{name}] 运行中...", end=" ")
        rate = run_experiment(cls)
        results[name] = rate
        print(f"成功率: {rate:.0f}%")
    
    # 结果
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(rate / 5)
        print(f"{name:<15}: {rate:5.1f}% {bar}")
    
    # 分析
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    fcrs = results["FCRS (预测)"]
    linear = results["Linear Q"]
    random = results["Random"]
    
    print(f"\nFCRS vs Linear Q: {fcrs - linear:+.1f}%")
    print(f"FCRS vs Random:  {fcrs - random:+.1f}%")
    
    if fcrs > linear:
        print("\n✅ FCRS 在连续状态任务中优于线性Q学习")
    else:
        print("\n⚠️ FCRS 与线性Q学习相近")


if __name__ == "__main__":
    main()
