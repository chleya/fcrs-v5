"""
修复版FCRS神经网络
"""

import numpy as np
from sklearn.datasets import load_digits


class FCRSNetFixed:
    """修复版FCRS网络"""
    
    def __init__(self, n_reps=10):
        self.n_reps = n_reps
        # 表征中心
        self.centers = {}
        
        # 初始化每个类别的表征
        for i in range(10):
            self.centers[i] = np.random.randn(64) * 0.1
    
    def forward(self, x):
        """找最近表征"""
        distances = []
        
        for label, center in self.centers.items():
            d = np.linalg.norm(x - center)
            distances.append((label, d))
        
        # 排序
        distances.sort(key=lambda x: x[1])
        
        return distances[0][0], distances[0][1]
    
    def train(self, x, y, lr=0.1):
        """学习"""
        # 更新对应类别的表征
        self.centers[y] += lr * (x - self.centers[y])
    
    def predict(self, x):
        label, _ = self.forward(x)
        return label


class SimpleNN:
    """简单神经网络"""
    
    def __init__(self):
        self.W = np.random.randn(64, 10) * 0.1
    
    def forward(self, x):
        return np.dot(x, self.W)
    
    def train(self, x, y, lr=0.01):
        out = self.forward(x)
        
        # 简化更新
        error = out[y] - 1
        self.W[:, y] += lr * error * x
    
    def predict(self, x):
        return np.argmax(self.forward(x))


def test():
    print('='*60)
    print('Fixed Deep Learning Integration')
    print('='*60)
    
    # 数据
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # 1. FCRS
    print('\n1. FCRS Classifier')
    fcrs = FCRSNetFixed(n_reps=10)
    
    for x, label in zip(X_train, y_train):
        fcrs.train(x, label, lr=0.5)
    
    correct = sum(fcrs.predict(x) == y for x, y in zip(X_test, y_test))
    acc_fcrs = correct / len(X_test)
    print(f'   Accuracy: {acc_fcrs:.2%}')
    
    # 2. Simple NN
    print('\n2. Simple NN')
    nn = SimpleNN()
    
    for epoch in range(10):
        for x, label in zip(X_train, y_train):
            nn.train(x, label, lr=0.01)
    
    correct = sum(nn.predict(x) == y for x, y in zip(X_test, y_test))
    acc_nn = correct / len(X_test)
    print(f'   Accuracy: {acc_nn:.2%}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS: {acc_fcrs:.2%}')
    print(f'NN:   {acc_nn:.2%}')


if __name__ == "__main__":
    test()
