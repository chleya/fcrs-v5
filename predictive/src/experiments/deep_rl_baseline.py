# -*- coding: utf-8 -*-
"""
FCRS-v5 深度RL基线对比

对比方法:
1. FCRS (预测导向) - 当前最优方法
2. DQN (Deep Q-Network)
3. A2C (Advantage Actor-Critic)
4. Q-Learning (表格型)
"""

import numpy as np


# ============ 网格世界环境 ============

class GridWorld:
    def __init__(self, size=5, n_obstacles=3, seed=42):
        self.size = size
        self.n_obstacles = n_obstacles
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
        state = np.zeros(self.size * self.size)
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action):
        actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        dx, dy = actions[action]
        
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            reward = -1.0
            done = False
        elif (new_x, new_y) in self.obstacles:
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


# ============ FCRS (预测导向) ============

class FCRS:
    """FCRS预测导向方法"""
    
    def __init__(self, state_dim, action_dim, n_reps=5, lr=0.05):
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
    
    def select_action(self, state):
        # 预测导向选择
        if self.history:
            errors = []
            for p in self.predictors:
                pred = self.history[-1] @ p
                errors.append(np.linalg.norm(pred - state))
            rep_idx = np.argmin(errors)
        else:
            rep_idx = np.random.randint(self.n_reps)
        
        # 动作选择
        q = self.reps[rep_idx] @ self.W + self.b
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim), rep_idx
        return np.argmax(q), rep_idx
    
    def update(self, state, action, reward, next_state, rep_idx):
        # 更新表征
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        
        # 更新预测器
        if self.history:
            pred = self.history[-1] @ self.predictors[rep_idx]
            error = next_state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.history[-1], error)
        
        # 更新策略
        q = self.reps[rep_idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * self.reps[rep_idx]
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(state)
        if len(self.history) > 100:
            self.history = self.history[-100:]


# ============ DQN (简化版) ============

class SimpleDQN:
    """简化DQN"""
    
    def __init__(self, state_dim, action_dim, hidden=32, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 简单网络
        self.W1 = np.random.randn(state_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
        self.explore = 0.3
    
    def forward(self, state):
        h = np.tanh(state @ self.W1 + self.b1)
        return h @ self.W2 + self.b2
    
    def select_action(self, state):
        q = self.forward(state)
        if np.random.random() < self.explore:
            return np.random.randint(self.action_dim)
        return np.argmax(q)
    
    def update(self, state, action, reward, next_state, done):
        # 简单DQN更新
        q = self.forward(state)
        target = reward + 0.99 * np.max(self.forward(next_state)) * (1 - done)
        
        # 梯度
        h = np.tanh(state @ self.W1 + self.b1)
        error = target - q[action]
        
        d_W2 = error * h
        d_b2 = error
        d_h = error * self.W2[action] * (1 - h**2)
        d_W1 = np.outer(state, d_h)
        d_b1 = d_h
        
        self.W2 += self.lr * d_W2
        self.b2 += self.lr * d_b2
        self.W1 += self.lr * d_W1
        self.b1 += self.lr * d_b1


# ============ A2C (简化版) ============

class SimpleA2C:
    """简化A2C"""
    
    def __init__(self, state_dim, action_dim, hidden=32, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 策略网络
        self.W_policy = np.random.randn(state_dim, hidden) * 0.1
        self.b_policy = np.zeros(hidden)
        self.W_out = np.random.randn(hidden, action_dim) * 0.1
        
        # 价值网络
        self.W_value = np.random.randn(state_dim, hidden) * 0.1
        self.b_value = np.zeros(hidden)
        self.W_vout = np.random.randn(hidden, 1) * 0.1
        
        self.explore = 0.3
    
    def forward(self, state):
        # 策略
        h_p = np.tanh(state @ self.W_policy + self.b_policy)
        policy = softmax(h_p @ self.W_out)
        
        # 价值
        h_v = np.tanh(state @ self.W_value + self.b_value)
        value = h_v @ self.W_vout
        
        return policy, value
    
    def select_action(self, state):
        policy, _ = self.forward(state)
        if np.random.random() < self.explore:
            return np.random.choice(self.action_dim, p=policy)
        return np.argmax(policy)
    
    def update(self, state, action, reward, next_state, done):
        policy, value = self.forward(state)
        _, next_value = self.forward(next_state)
        
        # 优势
        target = reward + 0.99 * next_value * (1 - done)
        advantage = target - value
        
        # 策略梯度
        for a in range(self.action_dim):
            if a == action:
                self.W_out[:, a] += self.lr * advantage * policy[a] * (1 - policy[a]) * np.tanh(state @ self.W_policy + self.b_policy)
        
        # 价值梯度
        self.W_vout += self.lr * advantage * np.tanh(state @ self.W_value + self.b_value)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ============ Q-Learning (表格型) ============

class QLearning:
    """表格型Q-Learning"""
    
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99):
        self.lr = lr
        self.gamma = gamma
        self.Q = np.zeros((state_dim, action_dim))
        self.explore = 0.3
    
    def select_action(self, state):
        state_idx = np.argmax(state)
        if np.random.random() < self.explore:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state_idx])
    
    def update(self, state, action, reward, next_state, done):
        state_idx = np.argmax(state)
        next_idx = np.argmax(next_state)
        
        target = reward + self.gamma * np.max(self.Q[next_idx]) * (1 - done)
        self.Q[state_idx, action] += self.lr * (target - self.Q[state_idx, action])


# ============ 对比实验 ============

def run_experiment(agent_class, agent_kwargs, name, n_episodes=200, seed=100):
    """运行实验"""
    np.random.seed(seed)
    
    env = GridWorld(5, 3, seed)
    
    if agent_class == FCRS:
        agent = agent_class(**agent_kwargs)
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 50:
                action, rep_idx = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, rep_idx)
                state = next_state
                steps += 1
    else:
        agent = agent_class(**agent_kwargs)
        for ep in range(n_episodes):
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 50:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1
    
    # 测试
    successes = 0
    for _ in range(50):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            if agent_class == FCRS:
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)
            state, reward, done = env.step(action)
            steps += 1
        if done:
            successes += 1
    
    return successes / 50 * 100


def main():
    print("="*60)
    print("FCRS vs Deep RL Baselines")
    print("="*60)
    
    agents = [
        ("FCRS (Prediction)", FCRS, {'state_dim': 25, 'action_dim': 4, 'n_reps': 5, 'lr': 0.05}),
        ("DQN", SimpleDQN, {'state_dim': 25, 'action_dim': 4, 'lr': 0.01}),
        ("A2C", SimpleA2C, {'state_dim': 25, 'action_dim': 4, 'lr': 0.01}),
        ("Q-Learning", QLearning, {'state_dim': 25, 'action_dim': 4, 'lr': 0.1}),
    ]
    
    results = {}
    
    for name, agent_class, kwargs in agents:
        print(f"\n[{name}] Running...")
        success_rate = run_experiment(agent_class, kwargs, name)
        results[name] = success_rate
        print(f"  Success Rate: {success_rate:.1f}%")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for name, rate in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{name:<20}: {rate:.1f}%")
    
    # 结论
    fcrs_rate = results["FCRS (Prediction)"]
    dqn_rate = results["DQN"]
    a2c_rate = results["A2C"]
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    
    if fcrs_rate > dqn_rate and fcrs_rate > a2c_rate:
        print(f"\nFCRS outperforms DQN by {fcrs_rate - dqn_rate:+.1f}%")
        print(f"FCRS outperforms A2C by {fcrs_rate - a2c_rate:+.1f}%")
    else:
        print("\nDeep RL methods perform comparably or better.")
        print("FCRS is designed for resource-constrained scenarios.")


if __name__ == "__main__":
    main()
