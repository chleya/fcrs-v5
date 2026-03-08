"""
深度学习集成 - 更深入的实现
完整的FCRS层 + 神经网络
"""

import numpy as np
from sklearn.datasets import load_digits


class FCRSLayer:
    """FCRS层 - 表征竞争层"""
    
    def __init__(self, input_dim, n_reps=5):
        self.input_dim = input_dim
        self.n_reps = n_reps
        
        # 表征向量
        self.representations = np.random.randn(n_reps, input_dim) * 0.1
        
        # 表征权重
        self.rep_weights = np.ones(n_reps) / n_reps
    
    def forward(self, x):
        """前向传播"""
        # 计算与每个表征的相似度
        similarities = []
        
        for rep in self.representations:
            norm_x = np.linalg.norm(x)
            norm_r = np.linalg.norm(rep)
            
            if norm_x > 0.01 and norm_r > 0.01:
                sim = np.dot(x, rep) / (norm_x * norm_r)
            else:
                sim = 0
            
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # 归一化权重 (softmax)
        exp_sim = np.exp(similarities - np.max(similarities))
        self.rep_weights = exp_sim / np.sum(exp_sim)
        
        # 加权表征
        weighted_rep = np.sum(self.rep_weights[:, None] * self.representations, axis=0)
        
        return weighted_rep
    
    def update(self, x, learning_rate=0.1):
        """更新表征"""
        # 找到最佳表征
        best_idx = np.argmax(self.rep_weights)
        
        # 更新最佳表征
        self.representations[best_idx] += learning_rate * x
        
        # 归一化
        self.representations[best_idx] /= (np.linalg.norm(self.representations[best_idx]) + 1e-8)


class FCRSNetwork:
    """FCRS神经网络"""
    
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10, n_reps=5):
        # FCRS层
        self.fcrs = FCRSLayer(input_dim, n_reps)
        
        # 隐藏层
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        
        # 输出层
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x):
        # FCRS层
        fcrs_out = self.fcrs.forward(x)
        
        # 隐藏层
        h = np.tanh(np.dot(fcrs_out, self.W1) + self.b1)
        
        # 输出层
        out = np.dot(h, self.W2) + self.b2
        
        return out
    
    def train(self, x, y, learning_rate=0.01):
        """训练一步"""
        # 前向
        out = self.forward(x)
        
        # 误差
        error = out.copy()
        error[y] -= 1
        
        # 输出层梯度
        h = np.tanh(np.dot(self.fcrs.forward(x), self.W1) + self.b1)
        
        self.W2 += learning_rate * np.outer(h, error)
        self.b2 += learning_rate * error
        
        # 隐藏层梯度
        h_error = np.dot(error, self.W2.T) * (1 - h**2)
        
        self.W1 += learning_rate * np.outer(self.fcrs.forward(x), h_error)
        self.b1 += learning_rate * h_error
        
        # FCRS层更新
        self.fcrs.update(x, learning_rate=0.1)
    
    def predict(self, x):
        out = self.forward(x)
        return np.argmax(out)


class StandardNN:
    """标准神经网络对比"""
    
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=10):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
    
    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2
    
    def train(self, x, y, lr=0.01):
        out = self.forward(x)
        error = out.copy()
        error[y] -= 1
        
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        
        self.W2 += lr * np.outer(h, error)
        self.b2 += lr * error
        
        h_error = np.dot(error, self.W2.T) * (1 - h**2)
        
        self.W1 += lr * np.outer(x, h_error)
        self.b1 += lr * h_error
    
    def predict(self, x):
        return np.argmax(self.forward(x))


def test():
    """测试"""
    print('='*60)
    print('Deep Learning Integration - Deep Version')
    print('='*60)
    
    # 数据
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # 1. FCRS Network
    print('\n1. FCRS Network')
    fcrs_net = FCRSNetwork(input_dim=64, hidden_dim=32, output_dim=10, n_reps=5)
    
    for epoch in range(30):
        for x, label in zip(X_train, y_train):
            fcrs_net.train(x, label, learning_rate=0.01)
    
    correct = sum(fcrs_net.predict(x) == y for x, y in zip(X_test, y_test))
    acc_fcrs = correct / len(X_test)
    print(f'   Accuracy: {acc_fcrs:.2%}')
    
    # 2. Standard NN
    print('\n2. Standard NN')
    nn = StandardNN(input_dim=64, hidden_dim=32, output_dim=10)
    
    for epoch in range(30):
        for x, label in zip(X_train, y_train):
            nn.train(x, label, lr=0.01)
    
    correct = sum(nn.predict(x) == y for x, y in zip(X_test, y_test))
    acc_nn = correct / len(X_test)
    print(f'   Accuracy: {acc_nn:.2%}')
    
    # 3. FCRS with more reps
    print('\n3. FCRS Network (10 reps)')
    fcrs_net10 = FCRSNetwork(input_dim=64, hidden_dim=32, output_dim=10, n_reps=10)
    
    for epoch in range(30):
        for x, label in zip(X_train, y_train):
            fcrs_net10.train(x, label, learning_rate=0.01)
    
    correct = sum(fcrs_net10.predict(x) == y for x, y in zip(X_test, y_test))
    acc_fcrs10 = correct / len(X_test)
    print(f'   Accuracy: {acc_fcrs10:.2%}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS-NN (5 reps):  {acc_fcrs:.2%}')
    print(f'FCRS-NN (10 reps): {acc_fcrs10:.2%}')
    print(f'Standard NN:      {acc_nn:.2%}')
    
    best = max(acc_fcrs, acc_fcrs10, acc_nn)
    if best == acc_fcrs or best == acc_fcrs10:
        print('\n[OK] FCRS Network wins!')
    else:
        print('\n[INFO] Standard NN competitive')


if __name__ == "__main__":
    test()
