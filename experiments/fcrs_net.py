"""
简化的FCRS神经网络
"""

import numpy as np
from sklearn.datasets import load_digits


class FCRSNet:
    """FCRS表征 + 简单分类器"""
    
    def __init__(self, n_reps=10):
        # 表征
        self.n_reps = n_reps
        self.reps = np.random.randn(n_reps, 64) * 0.1
    
    def forward(self, x):
        # 计算与每个表征的相似度
        similarities = []
        
        for rep in self.reps:
            norm_x = np.linalg.norm(x)
            norm_r = np.linalg.norm(rep)
            
            if norm_x > 0.01 and norm_r > 0.01:
                sim = np.dot(x, rep) / (norm_x * norm_r)
            else:
                sim = 0
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # softmax
        exp_sim = np.exp(similarities - np.max(similarities))
        attention = exp_sim / np.sum(exp_sim)
        
        # 加权表征
        weighted = np.sum(attention[:, None] * self.reps, axis=0)
        
        return weighted, attention
    
    def train(self, x, y, lr=0.1):
        """训练"""
        weighted, attention = self.forward(x)
        
        # 找到最佳表征
        best_idx = np.argmax(attention)
        
        # 更新最佳表征
        self.reps[best_idx] += lr * x
        
        # 归一化
        self.reps[best_idx] /= (np.linalg.norm(self.reps[best_idx]) + 1e-8)
    
    def predict(self, x):
        weighted, _ = self.forward(x)
        
        # 最近邻分类
        best = 0
        best_dist = float('inf')
        
        for i in range(self.n_reps):
            d = np.linalg.norm(weighted - self.reps[i])
            if d < best_dist:
                best_dist = d
                best = i
        
        return best


class SimpleClassifier:
    """简单分类器"""
    
    def __init__(self):
        self.reps = np.random.randn(10, 64) * 0.1
    
    def train(self, x, y, lr=0.1):
        self.reps[y] += lr * x
        self.reps[y] /= (np.linalg.norm(self.reps[y]) + 1e-8)
    
    def predict(self, x):
        dists = [np.linalg.norm(x - r) for r in self.reps]
        return np.argmin(dists)


def test():
    print('='*60)
    print('FCRS Neural Network Integration')
    print('='*60)
    
    # 数据
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    n = 1400
    X_train, X_test = X[:n], X[n:]
    y_train, y_test = y[:n], y[n:]
    
    # 1. FCRS-Net
    print('\n1. FCRS-Net (10表征)')
    fcrs = FCRSNet(n_reps=10)
    
    for x, label in zip(X_train, y_train):
        fcrs.train(x, label, lr=0.1)
    
    correct = sum(fcrs.predict(x) == y for x, y in zip(X_test, y_test))
    acc_fcrs = correct / len(X_test)
    print(f'   Accuracy: {acc_fcrs:.2%}')
    
    # 2. FCRS-Net (5表征)
    print('\n2. FCRS-Net (5表征)')
    fcrs5 = FCRSNet(n_reps=5)
    
    for x, label in zip(X_train, y_train):
        fcrs5.train(x, label, lr=0.1)
    
    correct = sum(fcrs5.predict(x) == y for x, y in zip(X_test, y_test))
    acc_fcrs5 = correct / len(X_test)
    print(f'   Accuracy: {acc_fcrs5:.2%}')
    
    # 3. 简单分类器
    print('\n3. Simple Classifier')
    simple = SimpleClassifier()
    
    for x, label in zip(X_train, y_train):
        simple.train(x, label, lr=0.1)
    
    correct = sum(simple.predict(x) == y for x, y in zip(X_test, y_test))
    acc_simple = correct / len(X_test)
    print(f'   Accuracy: {acc_simple:.2%}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS-Net (10): {acc_fcrs:.2%}')
    print(f'FCRS-Net (5):  {acc_fcrs5:.2%}')
    print(f'Simple:       {acc_simple:.2%}')


if __name__ == "__main__":
    test()
