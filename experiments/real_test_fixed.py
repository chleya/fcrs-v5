"""
修复版真实测试
"""

import numpy as np

# ==================== MNIST ====================
def test_mnist():
    print('='*60)
    print('1. MNIST Test')
    print('='*60)
    
    # 加载
    print('Loading...')
    data = np.load('F:/datasets/mnist.npz', allow_pickle=True)
    X = data['data']
    y = data['target'].astype(int)
    
    X = X[:10000] / 255.0
    y = y[:10000]
    
    n_train = 8000
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')
    
    # FCRS
    class FCRS:
        def __init__(self):
            self.reps = {}
            self.counts = {}
        
        def fit(self, X, y):
            for x, label in zip(X, y):
                if label not in self.reps:
                    self.reps[label] = np.zeros(len(x))
                    self.counts[label] = 0
                self.reps[label] += x
                self.counts[label] += 1
            for l in self.reps:
                self.reps[l] /= max(1, self.counts[l])
        
        def predict(self, X):
            preds = []
            for x in X:
                best = min(self.reps.keys(), key=lambda l: np.linalg.norm(x - self.reps[l]))
                preds.append(best)
            return np.array(preds)
    
    fcrs = FCRS()
    fcrs.fit(X_train, y_train)
    preds = fcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'FCRS Accuracy: {acc:.2%}')
    
    return acc


# ==================== Digits ====================
def test_digits():
    print('\n' + '='*60)
    print('2. Digits Dataset (NN)')
    print('='*60)
    
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # 简化NN
    class NN:
        def __init__(self):
            self.W1 = np.random.randn(64, 32) * 0.01
            self.W2 = np.random.randn(32, 10) * 0.01
        
        def forward(self, x):
            h = np.tanh(np.dot(x, self.W1))
            return np.dot(h, self.W2)
        
        def train(self, X, y, epochs=20):
            for e in range(epochs):
                for x, label in zip(X, y):
                    h = np.tanh(np.dot(x, self.W1))
                    out = np.dot(h, self.W2)
                    
                    # 梯度
                    d2 = out.copy()
                    d2[label] -= 1
                    
                    self.W2 += 0.001 * np.outer(h, d2)
                    d1 = np.dot(d2, self.W2.T) * (1 - h**2)
                    self.W1 += 0.001 * np.outer(x, d1)
    
    nn = NN()
    nn.train(X_train, y_train, epochs=20)
    
    correct = sum(np.argmax(nn.forward(x)) == y for x, y in zip(X_test, y_test))
    acc = correct / len(X_test)
    print(f'NN Accuracy: {acc:.2%}')
    
    return acc


# ==================== RL ====================
def test_rl():
    print('\n' + '='*60)
    print('3. Reinforcement Learning')
    print('='*60)
    
    class Env:
        def __init__(self):
            self.size = 4
            self.agent = [0, 0]
            self.goal = [self.size-1, self.size-1]
        
        def reset(self):
            self.agent = [0, 0]
            return self.agent[0] * self.size + self.agent[1]
        
        def step(self, a):
            if a == 0 and self.agent[1] > 0: self.agent[1] -= 1
            elif a == 1 and self.agent[1] < self.size-1: self.agent[1] += 1
            elif a == 2 and self.agent[0] > 0: self.agent[0] -= 1
            elif a == 3 and self.agent[0] < self.size-1: self.agent[0] += 1
            
            s = self.agent[0] * self.size + self.agent[1]
            if self.agent == self.goal:
                return s, 1, True
            return s, -0.01, False
    
    class Agent:
        def __init__(self):
            self.Q = np.zeros((16, 4))
        
        def act(self, s):
            return np.argmax(self.Q[s]) if np.random.random() > 0.1 else np.random.randint(4)
        
        def learn(self, s, a, r, ns):
            self.Q[s, a] += 0.1 * (r + 0.9 * max(self.Q[ns]) - self.Q[s, a])
    
    env = Env()
    agent = Agent()
    
    rewards = []
    for ep in range(300):
        s = env.reset()
        total = 0
        done = False
        while not done:
            a = agent.act(s)
            ns, r, done = env.step(a)
            agent.learn(s, a, r, ns)
            s = ns
            total += r
        rewards.append(total)
    
    avg = np.mean(rewards[-50:])
    print(f'RL Avg Reward: {avg:.3f}')
    print('[OK]' if avg > 0.5 else '[WARN]')
    
    return avg


# ==================== Main ====================
def main():
    print('='*60)
    print('REAL TESTS')
    print('='*60)
    
    results = {}
    
    try:
        results['MNIST-FCRS'] = test_mnist()
    except Exception as e:
        print(f'MNIST Error: {e}')
    
    try:
        results['Digits-NN'] = test_digits()
    except Exception as e:
        print(f'NN Error: {e}')
    
    try:
        results['RL'] = test_rl()
    except Exception as e:
        print(f'RL Error: {e}')
    
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    for k, v in results.items():
        print(f'{k}: {v:.2%}' if isinstance(v, float) and v < 2 else f'{k}: {v}')


if __name__ == "__main__":
    main()
