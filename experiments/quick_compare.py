"""
快速对比实验
"""

import numpy as np

print('='*60)
print('COMPLETE COMPARISON')
print('='*60)

# ==================== 1. Deep Learning ====================
print('\n1. Neural Network')

class NN:
    def __init__(self):
        self.W = np.random.randn(10, 3) * 0.1
    
    def forward(self, x):
        return np.dot(x, self.W)
    
    def train(self, X, y):
        for x, label in zip(X, y):
            out = self.forward(x)
            error = out[label] - 1
            self.W[:, label] += 0.01 * error * x

np.random.seed(42)
X = np.random.randn(500, 10)
y = np.random.randint(0, 3, 500)

nn = NN()
nn.train(X, y)

correct = sum(np.argmax(nn.forward(x)) == yi for x, yi in zip(X[:100], y[:100]))
print(f'   NN Accuracy: {correct/100:.1%}')

# ==================== 2. RL ====================
print('\n2. Reinforcement Learning')

class Env:
    def __init__(self):
        self.state = 0
    
    def step(self, a):
        self.state += a - 1
        if self.state >= 5:
            return self.state, 1, True
        return self.state, 0, False

class Agent:
    def __init__(self):
        self.Q = np.zeros((10, 3))
    
    def act(self, s):
        return np.argmax(self.Q[s])
    
    def learn(self, s, a, r, ns):
        self.Q[s, a] += 0.1 * (r + 0.9 * np.max(self.Q[ns]) - self.Q[s, a])

env = Env()
agent = Agent()

rewards = []
for _ in range(50):
    s = env.state = 0
    total = 0
    for _ in range(10):
        a = agent.act(s)
        ns, r, done = env.step(a)
        agent.learn(s, a, r, ns)
        s = ns
        total += r
        if done:
            break
    rewards.append(total)

print(f'   RL Avg Reward: {np.mean(rewards[-10:]):.2f}')

# ==================== 3. FCRS ====================
print('\n3. FCRS')

class FCRS:
    def __init__(self):
        self.reps = [{'v': np.random.randn(10)*0.1} for _ in range(3)]
    
    def step(self, x, c):
        best = min(range(3), key=lambda i: np.linalg.norm(x - self.reps[i]['v']))
        self.reps[best]['v'] += 0.5 * (x - self.reps[best]['v'])
        return np.linalg.norm(x - self.reps[best]['v'])

fcrs = FCRS()
for _ in range(500):
    x = np.random.randn(10)
    c = np.random.randint(0, 3)
    fcrs.step(x, c)

errs = [fcrs.step(np.random.randn(10), 0) for _ in range(100)]
print(f'   FCRS Error: {np.mean(errs):.4f}')

# ==================== Summary ====================
print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print('| Method    | Score  |')
print('|----------|--------|')
print(f'| NN       | {correct/100:.1%}   |')
print(f'| RL       | {np.mean(rewards[-10:]):.2f}    |')
print(f'| FCRS     | {np.mean(errs):.4f} |')
