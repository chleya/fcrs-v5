# -*- coding: utf-8 -*-
"""
FCRS-v5 多任务泛化性测试

测试更多经典控制任务
"""

import numpy as np


# ============ 任务1: 2D导航 ============

class Navigation2DEnv:
    """2D导航: 从起点到目标点"""
    
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.state = None
    
    def reset(self):
        self.state = np.array([0, 0])
        return self.state.copy()
    
    def step(self, action):
        # 动作: 0=上,1=下,2=左,3=右
        moves = [(0,1), (0,-1), (-1,0), (1,0)]
        dx, dy = moves[action]
        
        x = np.clip(self.state[0] + dx, 0, self.size-1)
        y = np.clip(self.state[1] + dy, 0, self.size-1)
        
        self.state = np.array([x, y])
        
        done = tuple(self.state) == self.goal
        reward = 10.0 if done else -0.1
        
        return self.state.copy(), reward, done
    
    @property
    def input_dim(self): return 2
    @property
    def action_dim(self): return 4


# ============ 任务2: 资源收集 ============

class ResourceCollectionEnv:
    """资源收集: 收集分散的资源点"""
    
    def __init__(self, n_resources=3):
        self.n_resources = n_resources
        self.grid_size = 5
        self.state = None
        self.resources = None
    
    def reset(self):
        self.state = np.array([2, 2])  # 起点
        # 随机资源点
        self.resources = set()
        while len(self.resources) < self.n_resources:
            r = (np.random.randint(0, self.grid_size), 
                 np.random.randint(0, self.grid_size))
            if r != (2, 2):
                self.resources.add(r)
        return self.state.copy()
    
    def step(self, action):
        moves = [(0,1), (0,-1), (-1,0), (1,0)]
        dx, dy = moves[action]
        
        x = np.clip(self.state[0] + dx, 0, self.grid_size-1)
        y = np.clip(self.state[1] + dy, 0, self.grid_size-1)
        
        self.state = np.array([x, y])
        
        reward = 0
        if tuple(self.state) in self.resources:
            reward = 10
            self.resources.remove(tuple(self.state))
        
        done = len(self.resources) == 0
        return self.state.copy(), reward, done
    
    @property
    def input_dim(self): return 2
    @property
    def action_dim(self): return 4


# ============ FCRS Agent ============

class SimplePredictor:
    def __init__(self, dim):
        self.W = np.eye(dim) * 0.5
    
    def predict(self, s):
        return s @ self.W
    
    def update(self, sc, sn):
        self.W += 0.01 * np.outer(sc, sn - sc @ self.W)


class FCRSAgent:
    def __init__(self, input_dim, action_dim, mode='prediction', 
                 pool=5, lr=0.01, explore=0.3):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.mode = mode
        self.pool = pool
        self.lr = lr
        self.explore = explore
        
        self.reps = [np.random.randn(input_dim)*0.1 for _ in range(pool)]
        self.predictors = [SimplePredictor(input_dim) for _ in range(pool)]
        self.W = np.random.randn(input_dim, action_dim)*0.1
        self.b = np.zeros(action_dim)
        self.hist = []
    
    def sel_rep(self, s):
        if self.mode == 'prediction':
            es = []
            for p in self.predictors:
                if self.hist:
                    pred = p.predict(self.hist[-1])
                    es.append(np.linalg.norm(pred - s))
                else:
                    es.append(1.0)
            return np.argmin(es)
        return np.argmin([np.linalg.norm(r-s) for r in self.reps])
    
    def sel_act(self, r):
        v = r @ self.W + self.b
        return np.random.randint(self.action_dim) if np.random.random() < self.explore else np.argmax(v)
    
    def update(self, s, a, r, ns, i):
        self.reps[i] += self.lr * (s - self.reps[i])
        if self.hist:
            self.predictors[i].update(self.hist[-1], s)
        v = self.reps[i] @ self.W + self.b
        for j in range(self.action_dim):
            if j == a:
                self.W[:,j] += self.lr * (r - v[j]) * self.reps[i]
                self.b[j] += self.lr * (r - v[j])
        self.hist.append(s)
        if len(self.hist) > 100: self.hist = self.hist[-100:]
    
    def run(self, env, episodes=100, max_steps=50):
        rewards = []
        for _ in range(episodes):
            s = env.reset()
            total = 0
            for _ in range(max_steps):
                i = self.sel_rep(s)
                a = self.sel_act(self.reps[i])
                ns, rw, d = env.step(a)
                self.update(s, a, rw, ns, i)
                s = ns
                total += rw
                if d: break
            rewards.append(total)
        return np.mean(rewards[-10:])


def test_all():
    print("="*60)
    print("FCRS-v5 Multi-Task Generalization Test")
    print("="*60)
    
    results = {}
    
    # 任务1: 2D导航
    print("\n=== Navigation 2D ===")
    env = Navigation2DEnv(5)
    for mode in ['prediction', 'reconstruction', 'random']:
        r = FCRSAgent(2, 4, mode).run(env, 100, 50)
        results[f'nav_{mode}'] = r
        print(f"{mode}: {r:.2f}")
    
    # 任务2: 资源收集
    print("\n=== Resource Collection ===")
    env = ResourceCollectionEnv(3)
    for mode in ['prediction', 'reconstruction', 'random']:
        r = FCRSAgent(2, 4, mode).run(env, 100, 50)
        results[f'res_{mode}'] = r
        print(f"{mode}: {r:.2f}")
    
    # 总结
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nNavigation:")
    pred = results['nav_prediction']
    recon = results['nav_reconstruction']
    rand = results['nav_random']
    print(f"  Prediction: {pred:.2f}, Reconstruction: {recon:.2f}, Random: {rand:.2f}")
    print(f"  Best: {'Prediction' if pred >= recon and pred >= rand else 'Other'}")
    
    print("\nResource Collection:")
    pred = results['res_prediction']
    recon = results['res_reconstruction']
    rand = results['res_random']
    print(f"  Prediction: {pred:.2f}, Reconstruction: {recon:.2f}, Random: {rand:.2f}")
    print(f"  Best: {'Prediction' if pred >= recon and pred >= rand else 'Other'}")


if __name__ == "__main__":
    test_all()
