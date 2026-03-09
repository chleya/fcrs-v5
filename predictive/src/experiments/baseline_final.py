# -*- coding: utf-8 -*-
"""
FCRS vs 基线对比 (使用one-hot编码的网格世界)
"""

import numpy as np

class GridWorldOneHot:
    """One-hot编码的网格世界"""
    def __init__(self, size=5, n_obst=3, seed=42):
        self.size = size
        np.random.seed(seed)
        
        self.goal = size * size - 1
        self.obst = set()
        while len(self.obst) < n_obst:
            r = np.random.randint(0, size*size)
            if r != 0 and r != self.goal:
                self.obst.add(r)
        
        self.pos = 0
    
    def reset(self):
        self.pos = 0
        return self.get_state()
    
    def get_state(self):
        s = np.zeros(self.size * self.size)
        s[self.pos] = 1.0
        return s
    
    def step(self, a):
        moves = [(0,-1), (0,1), (-1,0), (1,0)]
        dx, dy = moves[a]
        nx = max(0, min(self.size-1, self.pos % self.size + dx))
        ny = max(0, min(self.size-1, self.pos // self.size + dy))
        new_pos = ny * self.size + nx
        
        if new_pos in self.obst:
            return self.get_state(), -1.0, False
        elif new_pos == self.goal:
            self.pos = new_pos
            return self.get_state(), 10.0, True
        else:
            self.pos = new_pos
            return self.get_state(), -0.1, False


class FCRSAgent:
    """FCRS预测导向方法"""
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.05):
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


class QTableAgent:
    """Q表方法"""
    def __init__(self, state_dim, action_dim, lr=0.1):
        self.Q = np.zeros((state_dim, action_dim))
        self.explore = 0.3
    
    def act(self, state):
        s = np.argmax(state)
        action = np.random.randint(self.Q.shape[1]) if np.random.random() < self.explore else np.argmax(self.Q[s])
        return action, 0
    
    def update(self, state, action, reward, next_state, idx):
        s = np.argmax(state)
        ns = np.argmax(next_state)
        self.Q[s, action] += 0.1 * (reward + 0.99 * np.max(self.Q[ns]) - self.Q[s, action])


class RandomAgent:
    def __init__(self, *args, **kwargs):
        pass
    
    def act(self, state):
        return np.random.randint(4), 0
    
    def update(self, *args):
        pass


def run(AgentClass, n_ep=200):
    np.random.seed(100)
    env = GridWorldOneHot(5, 3)
    agent = AgentClass(25, 4)
    
    # 训练
    for _ in range(n_ep):
        s = env.reset()
        for _ in range(50):
            a, idx = agent.act(s)
            ns, r, d = env.step(a)
            agent.update(s, a, r, ns, idx)
            s = ns
            if d: break
    
    # 测试
    success = 0
    for _ in range(50):
        s = env.reset()
        for _ in range(50):
            a, _ = agent.act(s)
            s, r, d = env.step(a)
            if d:
                success += 1
                break
    
    return success / 50 * 100


def main():
    print("="*50)
    print("FCRS vs Baselines (One-Hot Grid World)")
    print("="*50)
    
    for name, cls in [("FCRS", FCRSAgent), ("Q-Table", QTableAgent), ("Random", RandomAgent)]:
        print(f"\n[{name}]", end=" ")
        rate = run(cls)
        print(f"Success: {rate:.0f}%")


if __name__ == "__main__":
    main()
