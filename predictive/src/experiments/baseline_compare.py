# -*- coding: utf-8 -*-
"""
FCRS vs 基线对比
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
        
        return np.array([self.pos[0], self.pos[1]], dtype=np.float32), reward, done


class FCRSAgent:
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.05):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        self.W = np.random.randn(state_dim, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        self.history = []
        self.explore = 0.3
    
    def act(self, state):
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - state) for p in self.predictors]
            rep_idx = np.argmin(errors)
        else:
            rep_idx = 0
        
        q = self.reps[rep_idx] @ self.W + self.b
        if np.random.random() < self.explore:
            action = np.random.randint(self.action_dim)
        else:
            action = np.argmax(q)
        
        return action, rep_idx
    
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


class QLearningAgent:
    def __init__(self, state_dim, action_dim, lr=0.1):
        self.action_dim = action_dim
        self.Q = np.zeros((5, 5, action_dim))
        self.explore = 0.3
    
    def act(self, state):
        x, y = int(state[0]), int(state[1])
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim), 0
        return np.argmax(self.Q[x, y]), 0
    
    def update(self, state, action, reward, next_state, rep_idx):
        x, y = int(state[0]), int(state[1])
        nx, ny = int(next_state[0]), int(next_state[1])
        
        target = reward + 0.99 * np.max(self.Q[nx, ny])
        self.Q[x, y, action] += 0.1 * (target - self.Q[x, y, action])


class RandomAgent:
    def __init__(self, state_dim, action_dim, **kwargs):
        self.action_dim = action_dim
    
    def act(self, state):
        return np.random.randint(self.action_dim), 0
    
    def update(self, *args):
        pass


def run_exp(AgentClass, n_ep=200):
    np.random.seed(100)
    env = GridWorld(5, 3)
    agent = AgentClass(2, 4)
    
    for _ in range(n_ep):
        s = env.reset()
        for _ in range(50):
            a, idx = agent.act(s)
            ns, r, d = env.step(a)
            agent.update(s, a, r, ns, idx)
            s = ns
            if d: break
    
    # Test
    successes = 0
    for _ in range(50):
        s = env.reset()
        for _ in range(50):
            a, _ = agent.act(s)
            s, r, d = env.step(a)
            if d:
                successes += 1
                break
    
    return successes / 50 * 100


def main():
    print("="*50)
    print("FCRS vs Baselines")
    print("="*50)
    
    for name, cls in [("FCRS", FCRSAgent), ("Q-Learning", QLearningAgent), ("Random", RandomAgent)]:
        print(f"\n[{name}]", end=" ")
        rate = run_exp(cls)
        print(f"Success Rate: {rate:.1f}%")


if __name__ == "__main__":
    main()
