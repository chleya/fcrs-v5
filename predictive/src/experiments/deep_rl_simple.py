# -*- coding: utf-8 -*-
"""
FCRS vs Deep RL Baselines (简化版)
"""

import numpy as np


class GridWorld:
    def __init__(self, size=5, n_obstacles=3, seed=42):
        self.size = size
        np.random.seed(seed)
        
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            pos = (np.random.randint(0, size), np.random.randint(0, size))
            if pos != self.start and pos != self.goal:
                self.obstacles.add(pos)
        
        self.pos = self.start
    
    def reset(self):
        self.pos = self.start
        return self.get_state()
    
    def get_state(self):
        return np.array([self.pos[0], self.pos[1]], dtype=np.float32)
    
    def step(self, action):
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = moves[action]
        
        new_x = max(0, min(self.size-1, self.pos[0] + dx))
        new_y = max(0, min(self.size-1, self.pos[1] + dy))
        
        if (new_x, new_y) in self.obstacles:
            reward = -1.0
            done = False
        elif (new_x, new_y) == self.goal:
            self.pos = (new_x, new_y)
            reward = 10.0
            done = True
        else:
            self.pos = (new_x, new_y)
            reward = -0.1
            done = False
        
        return self.get_state(), reward, done


class FCRS:
    def __init__(self, state_dim=2, action_dim=4, n_reps=5, lr=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.history = []
        self.explore = 0.3
    
    def select_action(self, state):
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - state) for p in self.predictors]
            rep_idx = np.argmin(errors)
        else:
            rep_idx = np.random.randint(len(self.reps))
        
        q = self.reps[rep_idx] @ self.W + self.b
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim), rep_idx
        return np.argmax(q), rep_idx
    
    def update(self, state, action, reward, next_state, rep_idx):
        self.reps[rep_idx] += 0.05 * (state - self.reps[rep_idx])
        if self.history:
            pred = self.history[-1] @ self.predictors[rep_idx]
            self.predictors[rep_idx] += 0.05 * np.outer(self.history[-1], next_state - pred)
        
        q = self.reps[rep_idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += 0.05 * (reward - q[a]) * self.reps[rep_idx]
                self.b[a] += 0.05 * (reward - q[a])
        
        self.history.append(state)


class QLearning:
    def __init__(self, state_dim=2, action_dim=4, lr=0.1):
        self.Q = np.zeros((5, 5, action_dim))
        self.explore = 0.3
    
    def select_action(self, state):
        x, y = int(state[0]), int(state[1])
        if np.random.random() < self.explore:
            return np.random.randint(self.Q.shape[2])
        return np.argmax(self.Q[x, y])
    
    def update(self, state, action, reward, next_state, done):
        x, y = int(state[0]), int(state[1])
        nx, ny = int(next_state[0]), int(next_state[1])
        
        target = reward + 0.99 * np.max(self.Q[nx, ny]) * (1 - done)
        self.Q[x, y, action] += 0.1 * (target - self.Q[x, y, action])


class RandomAgent:
    def __init__(self, **kwargs):
        pass
    
    def select_action(self, state):
        return np.random.randint(4)
    
    def update(self, *args):
        pass


def run_exp(agent_class, n_episodes=200):
    np.random.seed(100)
    env = GridWorld(5, 3)
    agent = agent_class(state_dim=2, action_dim=4)
    
    # 训练
    for ep in range(n_episodes):
        state = env.reset()
        for _ in range(50):
            action, rep_idx = agent.select_action(state) if agent_class == FCRS else agent.select_action(state)
            if agent_class == FCRS:
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, rep_idx)
            else:
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
    
    # 测试
    successes = 0
    for _ in range(50):
        state = env.reset()
        for _ in range(50):
            action = agent.select_action(state)[0] if agent_class == FCRS else agent.select_action(state)
            state, reward, done = env.step(action)
            if done:
                successes += 1
                break
    
    return successes / 50 * 100


def main():
    print("="*50)
    print("FCRS vs Baselines")
    print("="*50)
    
    agents = [
        ("FCRS (Prediction)", FCRS),
        ("Q-Learning", QLearning),
        ("Random", RandomAgent),
    ]
    
    results = {}
    
    for name, agent_class in agents:
        print(f"\n[{name}] Running...")
        rate = run_exp(agent_class)
        results[name] = rate
        print(f"  Success Rate: {rate:.1f}%")
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<20}: {rate:.1f}%")


if __name__ == "__main__":
    main()
