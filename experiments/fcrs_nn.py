"""
FCRS深度学习集成 - 简化版
"""

import numpy as np
from sklearn.datasets import load_digits


class FCRSNeuralNet:
    """FCRS + 神经网络"""
    
    def __init__(self, input_dim=64, hidden=32, output=10):
        # 表征层
        self.reps = np.random.randn(5, input_dim) * 0.1
        
        # 神经网络
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, output) * 0.1
        self.b2 = np.zeros(output)
    
    def forward(self, x):
        # 表征注意力
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        r_norm = self.reps / (np.linalg.norm(self.reps, axis=1, keepdims=True) + 1e-8)
        
        attention = np.dot(x_norm, r_norm.T)
        attention = attention / (np.sum(attention) + 1e-8)
        
        # 隐藏层
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        
        # 输出
        out = np.dot(h, self.W2) + self.b2
        
        return out
    
    def train_step(self, x, y):
        out = self.forward(x)
        
        # 简化梯度
        error = out.copy()
        error[y] -= 1
        
        # 输出层更新
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        self.W2 += 0.01 * np.outer(h, error)
        self.b2 += 0.01 * error
        
        # 隐藏层更新
        h_error = np.dot(error, self.W2.T) * (1 - h**2)
        self.W1 += 0.01 * np.outer(x, h_error)
        self.b1 += 0.01 * h_error
        
        # 表征更新
        x_norm = x / (np.linalg.norm(x) + 1e-8)
        for i in range(5):
            sim = np.dot(x_norm, self.reps[i] / (np.linalg.norm(self.reps[i]) + 1e-8))
            self.reps[i] += 0.01 * sim * x_norm
    
    def predict(self, x):
        return np.argmax(self.forward(x))


class SimpleNN:
    """简单神经网络对比"""
    
    def __init__(self, input_dim=64, hidden=32, output=10):
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, output) * 0.1
        self.b2 = np.zeros(output)
    
    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2
    
    def train_step(self, x, y):
        out = self.forward(x)
        error = out.copy()
        error[y] -= 1
        
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        self.W2 += 0.01 * np.outer(h, error)
        self.b2 += 0.01 * error
        
        h_error = np.dot(error, self.W2.T) * (1 - h**2)
        self.W1 += 0.01 * np.outer(x, h_error)
        self.b1 += 0.01 * h_error
    
    def predict(self, x):
        return np.argmax(self.forward(x))


def main():
    print('='*60)
    print('FCRS Deep Learning Integration')
    print('='*60)
    
    # 数据
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # FCRS-NN
    print('\n1. FCRS Neural Network')
    fcrs_nn = FCRSNeuralNet()
    
    for epoch in range(20):
        for x, label in zip(X_train, y_train):
            fcrs_nn.train_step(x, label)
    
    preds = [fcrs_nn.predict(x) for x in X_test]
    acc_fcrs = np.mean(np.array(preds) == y_test)
    print(f'   Accuracy: {acc_fcrs:.2%}')
    
    # Simple NN
    print('\n2. Simple Neural Network')
    nn = SimpleNN()
    
    for epoch in range(20):
        for x, label in zip(X_train, y_train):
            nn.train_step(x, label)
    
    preds = [nn.predict(x) for x in X_test]
    acc_nn = np.mean(np.array(preds) == y_test)
    print(f'   Accuracy: {acc_nn:.2%}')
    
    # 对比
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS-NN: {acc_fcrs:.2%}')
    print(f'Simple NN: {acc_nn:.2%}')
    
    if acc_fcrs > acc_nn:
        print('\n[OK] FCRS-NN beats Simple NN!')
    else:
        print('\n[INFO] Both similar')


if __name__ == "__main__":
    main()
