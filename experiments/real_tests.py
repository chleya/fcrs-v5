"""
完整真实测试: MNIST + 深度学习 + 强化学习
"""

import numpy as np

# ==================== MNIST 真实测试 ====================
def test_mnist():
    """MNIST测试"""
    print('='*60)
    print('1. MNIST Dataset Test')
    print('='*60)
    
    # 加载MNIST
    print('Loading MNIST from F:/datasets/mnist.npz...')
    data = np.load('F:/datasets/mnist.npz')
    X, y = data['data'], data['target']
    
    # 转为int
    y = y.astype(int)
    
    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features')
    print(f'Classes: {len(np.unique(y))}')
    
    # 归一化
    X = X / 255.0
    
    # 取前10000样本测试(太快)
    X = X[:10000]
    y = y[:10000]
    
    # 划分
    n_train = 8000
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')
    
    # FCRS-like分类器
    class FCRSClassifier:
        def __init__(self, dim=64):
            self.dim = dim
            self.representations = {}
            self.counts = {}
        
        def fit(self, X, y):
            for x, label in zip(X, y):
                if label not in self.representations:
                    self.representations[label] = np.zeros(len(x))
                    self.counts[label] = 0
                self.representations[label] += x
                self.counts[label] += 1
            
            for label in self.representations:
                self.representations[label] /= max(1, self.counts[label])
        
        def predict(self, X):
            preds = []
            for x in X:
                best_label = None
                best_dist = float('inf')
                for label, rep in self.representations.items():
                    d = np.linalg.norm(x - rep)
                    if d < best_dist:
                        best_dist = d
                        best_label = label
                preds.append(best_label)
            return np.array(preds)
    
    print('\nTraining FCRS...')
    fcrs = FCRSClassifier()
    fcrs.fit(X_train, y_train)
    
    print('Testing...')
    preds = fcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'FCRS Accuracy: {acc:.2%}')
    
    return acc


# ==================== 深度学习测试 ====================
def test_deep_learning():
    """深度学习测试"""
    print('\n' + '='*60)
    print('2. Deep Learning Test')
    print('='*60)
    
    # 使用digits数据集
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    
    # 归一化
    X = X / 16.0
    
    # 划分
    n_train = 1400
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # 简单神经网络
    class SimpleNN:
        def __init__(self, input_dim, hidden, output):
            self.W1 = np.random.randn(input_dim, hidden) * 0.1
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, output) * 0.1
            self.b2 = np.zeros(output)
        
        def forward(self, x):
            h = np.tanh(np.dot(x, self.W1) + self.b1)
            out = np.dot(h, self.W2) + self.b2
            return out
        
        def train(self, X, y, epochs=50):
            for epoch in range(epochs):
                for x, label in zip(X, y):
                    # 前向
                    h = np.tanh(np.dot(x, self.W1) + self.b1)
                    out = np.dot(h, self.W2) + self.b2
                    
                    # 简化反向传播
                    error = out.copy()
                    error[label] -= 1
                    
                    # 更新
                    self.W2 += 0.01 * np.outer(h, error)
                    self.b2 += 0.01 * error
                    
                    # 隐藏层
                    h_error = np.dot(error, self.W2.T) * (1 - h**2)
                    self.W1 += 0.01 * np.outer(x, h_error)
                    self.b1 += 0.01 * h_error
    
    print('Training Neural Network...')
    nn = SimpleNN(64, 32, 10)
    nn.train(X_train, y_train, epochs=50)
    
    # 测试
    correct = 0
    for x, y in zip(X_test, y_test):
        pred = np.argmax(nn.forward(x))
        if pred == y:
            correct += 1
    
    acc = correct / len(X_test)
    print(f'NN Accuracy: {acc:.2%}')
    
    return acc


# ==================== 强化学习测试 ====================
def test_rl():
    """强化学习测试"""
    print('\n' + '='*60)
    print('3. Reinforcement Learning Test')
    print('='*60)
    
    # Grid World环境
    class GridWorld:
        def __init__(self, size=5):
            self.size = size
            self.agent = [0, 0]
            self.goal = [size-1, size-1]
        
        def reset(self):
            self.agent = [0, 0]
            return tuple(self.agent)
        
        def step(self, action):
            # 0:上, 1:下, 2:左, 3:右
            if action == 0 and self.agent[1] > 0:
                self.agent[1] -= 1
            elif action == 1 and self.agent[1] < self.size-1:
                self.agent[1] += 1
            elif action == 2 and self.agent[0] > 0:
                self.agent[0] -= 1
            elif action == 3 and self.agent[0] < self.size-1:
                self.agent[0] += 1
            
            # 奖励
            if self.agent == self.goal:
                return tuple(self.agent), 1, True
            else:
                return tuple(self.agent), -0.01, False
    
    # Q-Learning
    class QAgent:
        def __init__(self, n_states, n_actions):
            self.Q = np.zeros((n_states, n_actions))
            self.n_actions = n_actions
        
        def act(self, state):
            if np.random.random() < 0.1:  # epsilon-greedy
                return np.random.randint(self.n_actions)
            return np.argmax(self.Q[state])
        
        def learn(self, state, action, reward, next_state):
            self.Q[state, action] += 0.1 * (reward + 0.9 * np.max(self.Q[next_state]) - self.Q[state, action])
    
    print('Training RL Agent...')
    env = GridWorld(size=5)
    agent = QAgent(25, 4)
    
    episodes = 500
    rewards = []
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward
        
        rewards.append(total_reward)
        
        if ep % 100 == 0:
            print(f'Episode {ep}: reward={total_reward:.2f}')
    
    avg_reward = np.mean(rewards[-50:])
    print(f'\nAvg Reward (last 50): {avg_reward:.3f}')
    
    if avg_reward > 0.5:
        print('[OK] RL Agent learned!')
    else:
        print('[WARN] RL learning slow')
    
    return avg_reward


# ==================== 主函数 ====================
def main():
    print('='*60)
    print('COMPLETE REAL TESTS')
    print('='*60)
    
    results = {}
    
    # 1. MNIST
    try:
        results['MNIST-FCRS'] = test_mnist()
    except Exception as e:
        print(f'MNIST Error: {e}')
    
    # 2. Deep Learning
    try:
        results['Digits-NN'] = test_deep_learning()
    except Exception as e:
        print(f'NN Error: {e}')
    
    # 3. RL
    try:
        results['RL'] = test_rl()
    except Exception as e:
        print(f'RL Error: {e}')
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    
    for name, value in results.items():
        if isinstance(value, float):
            if value < 1:
                print(f'{name}: {value:.3f}')
            else:
                print(f'{name}: {value:.2%}')
        else:
            print(f'{name}: {value}')


if __name__ == "__main__":
    main()
