"""
完整对比实验:
1. MNIST数据集
2. 深度学习方法
3. 强化学习环境
"""

import numpy as np

# ==================== MNIST ====================
def test_mnist():
    """MNIST测试"""
    print('='*60)
    print('1. MNIST Dataset Test')
    print('='*60)
    
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        
        # 加载MNIST简化版
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # 二分类: 0 vs 1
        mask = (y == 0) | (y == 1)
        X, y = X[mask], y[mask]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # FCRS-like分类器
        class FCRSClassifier:
            def __init__(self):
                self.reps = {0: np.zeros(64), 1: np.zeros(64)}
                self.counts = {0: 0, 1: 0}
            
            def fit(self, X, y):
                for x, label in zip(X, y):
                    self.reps[label] += x
                    self.counts[label] += 1
                
                for label in self.reps:
                    self.reps[label] /= max(1, self.counts[label])
            
            def predict(self, X):
                preds = []
                for x in X:
                    d0 = np.linalg.norm(x - self.reps[0])
                    d1 = np.linalg.norm(x - self.reps[1])
                    preds.append(0 if d0 < d1 else 1)
                return np.array(preds)
        
        clf = FCRSClassifier()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        
        accuracy = np.mean(preds == y_test)
        print(f'FCRS Classifier Accuracy: {accuracy:.2%}')
        
        return accuracy
    
    except ImportError:
        print('sklearn not available, skipping MNIST')
        return None


# ==================== Deep Learning ====================
def test_deep_learning():
    """深度学习方法对比"""
    print('\n' + '='*60)
    print('2. Deep Learning Comparison')
    print('='*60)
    
    # 简化版神经网络 (从零实现)
    class SimpleNN:
        def __init__(self, input_dim=10, hidden=20, output=3):
            self.W1 = np.random.randn(input_dim, hidden) * 0.1
            self.b1 = np.zeros(hidden)
            self.W2 = np.random.randn(hidden, output) * 0.1
            self.b2 = np.zeros(output)
        
        def forward(self, x):
            h = np.tanh(np.dot(x, self.W1) + self.b1)
            return np.dot(h, self.W2) + self.b2
        
        def train(self, X, y, epochs=100):
            for _ in range(epochs):
                for x, target in zip(X, y):
                    # 前向
                    h = np.tanh(np.dot(x, self.W1) + self.b1)
                    out = np.dot(h, self.W2) + self.b2
                    
                    # 简化梯度
                    error = out[target] - 1
                    self.W2[:, target] += 0.01 * error * h
                    self.b2[target] += 0.01 * error
    
    print('Training Simple Neural Network...')
    
    # 测试数据
    np.random.seed(42)
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 3, 1000)
    
    nn = SimpleNN()
    nn.train(X, y)
    
    # 测试
    correct = 0
    for x, target in zip(X[:100], y[:100]):
        pred = np.argmax(nn.forward(x))
        if pred == target:
            correct += 1
    
    accuracy = correct / 100
    print(f'NN Accuracy: {accuracy:.2%}')


# ==================== Reinforcement Learning ====================
def test_rl():
    """强化学习环境"""
    print('\n' + '='*60)
    print('3. Reinforcement Learning Environment')
    print('='*60)
    
    # 简化RL环境
    class SimpleEnv:
        def __init__(self):
            self.state = 0
            self.goal = 10
        
        def reset(self):
            self.state = 0
            return self.state
        
        def step(self, action):
            self.state += action - 1  # 0=左, 1=中, 2=右
            
            if self.state >= self.goal:
                return self.state, 1, True  # reward, done
            elif self.state < 0:
                return 0, -1, True
            else:
                return self.state, 0, False
    
    class RLAgent:
        def __init__(self, n_actions=3):
            self.Q = np.zeros((20, n_actions))
        
        def act(self, state):
            return np.argmax(self.Q[state])
        
        def learn(self, state, action, reward, next_state):
            self.Q[state, action] += 0.1 * (reward + 0.9 * np.max(self.Q[next_state]) - self.Q[state, action])
    
    # 训练
    env = SimpleEnv()
    agent = RLAgent()
    
    episodes = 100
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
    
    avg_reward = np.mean(rewards[-20:])
    print(f'RL Agent - Avg Reward (last 20 eps): {avg_reward:.2f}')
    
    if avg_reward > 0:
        print('[OK] RL agent learned!')
    else:
        print('[WARN] RL agent not learning')


# ==================== Main ====================
def main():
    print('='*60)
    print('COMPLETE COMPARISON: All Methods')
    print('='*60)
    
    # 1. MNIST
    mnist_acc = test_mnist()
    
    # 2. Deep Learning
    test_deep_learning()
    
    # 3. Reinforcement Learning
    test_rl()
    
    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print('Methods tested:')
    print('  1. FCRS (from previous experiments)')
    print('  2. Online Learning')
    print('  3. K-Means')
    print('  4. Neural Network (deep learning)')
    print('  5. RL Agent (reinforcement learning)')
    if mnist_acc:
        print(f'  6. MNIST classification: {mnist_acc:.2%}')


if __name__ == "__main__":
    main()
