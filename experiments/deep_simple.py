"""
简化的深度学习集成
"""

import numpy as np
from sklearn.datasets import load_digits


class FCRSNet:
    """FCRS + 神经网络"""
    
    def __init__(self):
        # 表征
        self.reps = np.random.randn(10, 64) * 0.1
        
        # 权重
        self.W1 = np.random.randn(64, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 10) * 0.1
        self.b2 = np.zeros(10)
    
    def forward(self, x):
        # 表征层
        sims = []
        for r in self.reps:
            sim = np.dot(x, r) / (np.linalg.norm(x) * np.linalg.norm(r) + 1e-8)
            sims.append(sim)
        
        # 表征注意力
        att = np.exp(sims)
        att = att / np.sum(att)
        
        # 加权
        fcrs_out = np.sum(att[:, None] * self.reps, axis=0)
        
        # 神经网络
        h = np.tanh(np.dot(fcrs_out, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        
        return out
    
    def train(self, x, y, lr=0.01):
        out = self.forward(x)
        
        # 输出梯度
        d_out = out.copy()
        d_out[y] -= 1
        
        # 隐藏层
        h = np.tanh(np.dot(self.forward(x - np.dot(x, self.reps[0]) * self.reps[0] / (np.linalg.norm(self.reps[0])**2 + 1e-8), self.W1) + self.b1)
        
        # 简化更新
        self.W2[:, y] += lr * h * d_out[y]
        
        # 表征更新
        sims = [np.dot(x, r) for r in self.reps]
        for i, r in enumerate(self.reps):
            self.reps[i] += lr * sims[i] * x
    
    def predict(self, x):
        return np.argmax(self.forward(x))


class SimpleNN:
    """简单NN"""
    
    def __init__(self):
        self.W1 = np.random.randn(64, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, 10) * 0.1
        self.b2 = np.zeros(10)
    
    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2
    
    def train(self, x, y, lr=0.01):
        out = self.forward(x)
        d = out.copy()
        d[y] -= 1
        
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        self.W2[:, y] += lr * h * d[y]
    
    def predict(self, x):
        return np.argmax(self.forward(x))


def test():
    print('='*60)
    print('Deep Learning Integration')
    print('='*60)
    
    # 数据
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # FCRS-Net
    print('\n1. FCRS-Net')
    net = FCRSNet()
    
    for i in range(20):
        for x, label in zip(X_train[:500], y_train[:500]):
            net.train(x, label, lr=0.01)
    
    correct = sum(net.predict(x) == y for x, y in zip(X_test, y_test))
    print(f'   Accuracy: {correct/len(X_test):.2%}')
    
    # Simple NN
    print('\n2. Simple NN')
    nn = SimpleNN()
    
    for i in range(20):
        for x, label in zip(X_train[:500], y_train[:500]):
            nn.train(x, label, lr=0.01)
    
    correct = sum(nn.predict(x) == y for x, y in zip(X_test, y_test))
    print(f'   Accuracy: {correct/len(X_test):.2%}')


if __name__ == "__main__":
    test()
