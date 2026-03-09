# -*- coding: utf-8 -*-
"""
FCRS-v5 Atari with gymnasium (真实环境)
"""

import numpy as np

try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Warning: gymnasium not available, using simulation")


class AtariWrapper:
    """使用真实gymnasium的Atari封装"""
    def __init__(self, game='Pong', seed=42):
        if HAS_GYM:
            self.env = gym.make(game, render_mode='rgb_array')
            self.env.reset(seed=seed)
        else:
            self.env = None
        
        self.game = game
        self.score = 0
    
    def reset(self):
        if HAS_GYM:
            frame, info = self.env.reset()
            return self._preprocess(frame)
        return np.random.randint(0, 256, (84, 84), dtype=np.uint8)
    
    def step(self, action):
        if HAS_GYM:
            frame, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            return self._preprocess(frame), reward, done
        return np.random.randint(0, 256, (84, 84)), 0, False
    
    def _preprocess(self, frame):
        """预处理: 缩放到84x84灰度"""
        if len(frame.shape) == 3:
            # RGB -> 灰度
            gray = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
        else:
            gray = frame
        # 缩放
        from scipy.ndimage import zoom
        if gray.shape != (84, 84):
            zoom_factors = (84/gray.shape[0], 84/gray.shape[1])
            gray = zoom(gray, zoom_factors)
        return gray.astype(np.uint8)
    
    @property
    def input_dim(self):
        return 84 * 84
    
    @property
    def action_dim(self):
        if HAS_GYM:
            return self.env.action_space.n
        return 3


class FCRSAtari:
    """FCRS Atari版本"""
    def __init__(self, input_dim=7056, action_dim=3, n_reps=5, lr=0.01):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        
        # 降维: 7056 -> 64
        self.W_down = np.random.randn(input_dim, 64) * 0.01
        self.b_down = np.zeros(64)
        
        # 表征池
        self.reps = [np.random.randn(64) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(64) * 0.5 for _ in range(n_reps)]
        
        # 策略
        self.W = np.random.randn(64, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        
        self.history = []
        self.explore = 0.3
    
    def _preprocess(self, frame):
        x = frame.flatten().astype(np.float32) / 255.0
        x = x @ self.W_down + self.b_down
        x = np.tanh(x)
        return x
    
    def act(self, state):
        x = self._preprocess(state)
        
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - x) for p in self.predictors]
            idx = np.argmin(errors)
        else:
            idx = 0
        
        q = self.reps[idx] @ self.W + self.b
        action = np.random.randint(self.action_dim) if np.random.random() < self.explore else np.argmax(q)
        return action, idx
    
    def update(self, state, action, reward, next_state, idx):
        x = self._preprocess(state)
        nx = self._preprocess(next_state)
        
        self.reps[idx] += self.lr * (x - self.reps[idx])
        
        if self.history:
            pred = self.history[-1] @ self.predictors[idx]
            self.predictors[idx] += self.lr * np.outer(self.history[-1], nx - pred)
        
        q = self.reps[idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * self.reps[idx]
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(x)


class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim
    
    def act(self, state):
        return np.random.randint(self.action_dim), 0
    
    def update(self, *args):
        pass


def run_experiment(AgentClass, n_ep=200):
    np.random.seed(100)
    
    if not HAS_GYM:
        print("Using simulation mode")
        return 0
    
    env = AtariWrapper('PongNoFrameskip-v4')
    agent = AgentClass(7056, env.action_dim)
    
    print(f"Training for {n_ep} episodes...")
    
    for ep in range(n_ep):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while steps < 500:
            action, idx = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            total_reward += reward
            steps += 1
            if done:
                break
        
        if ep % 20 == 0:
            print(f"  Ep {ep}: reward={total_reward:.1f}, steps={steps}")
    
    # 测试
    test_rewards = []
    for _ in range(10):
        state = env.reset()
        total = 0
        for _ in range(500):
            action, _ = agent.act(state)
            state, reward, done = env.step(action)
            total += reward
            if done:
                break
        test_rewards.append(total)
    
    return np.mean(test_rewards)


def main():
    print("="*60)
    print("FCRS Atari - Real gymnasium Environment")
    print("="*60)
    
    if HAS_GYM:
        print("Using: PongNoFrameskip-v4\n")
        
        print("[FCRS] Training...")
        fcrs_score = run_experiment(FCRSAtari, n_ep=200)
        print(f"  Test score: {fcrs_score:.1f}")
        
        print("\n[Random] Training...")
        random_score = run_experiment(lambda *a, **k: RandomAgent(3), n_ep=200)
        print(f"  Test score: {random_score:.1f}")
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"FCRS:   {fcrs_score:.1f}")
        print(f"Random: {random_score:.1f}")
    else:
        print("gymnasium not installed")
        print("Install with: pip install gymnasium gymnasium[accept-rom-license]")


if __name__ == "__main__":
    main()
