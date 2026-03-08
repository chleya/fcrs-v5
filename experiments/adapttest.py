"""
自适应维度测试
"""

import numpy as np
from sklearn.datasets import load_digits, load_iris


class AdaptiveFCRS:
    """自适应维度"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reps = {i: np.zeros(64) for i in range(n_classes)}
        self.counts = {i: 0 for i in range(n_classes)}
        self.dims = {i: 10 for i in range(n_classes)}
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            dim = self.dims[label]
            
            # 学习
            self.reps[label][:dim] += 0.5 * (x[:dim] - self.reps[label][:dim])
            self.counts[label] += 1
            
            # 自适应维度
            error = np.linalg.norm(x[:dim] - self.reps[label][:dim])
            
            if error < 0.3 and dim < 64:
                self.dims[label] += 1
            elif error > 1.0 and dim > 3:
                self.dims[label] -= 1
    
    def predict(self, X):
        preds = []
        for x in X:
            best = min(range(self.n_classes),
                     key=lambda l: np.linalg.norm(x[:self.dims[l]] - self.reps[l][:self.dims[l]]))
            preds.append(best)
        return np.array(preds)


class FixedFCRS:
    """固定维度"""
    
    def __init__(self, n_classes, dim):
        self.n_classes = n_classes
        self.dim = dim
        self.reps = {i: np.zeros(64) for i in range(n_classes)}
        self.counts = {i: 0 for i in range(n_classes)}
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            self.reps[label][:self.dim] += 0.5 * (x[:self.dim] - self.reps[label][:self.dim])
            self.counts[label] += 1
    
    def predict(self, X):
        preds = []
        for x in X:
            best = min(range(self.n_classes),
                     key=lambda l: np.linalg.norm(x[:self.dim] - self.reps[l][:self.dim]))
            preds.append(best)
        return np.array(preds)


def test():
    print('='*60)
    print('Adaptive vs Fixed Dimension Test')
    print('='*60)
    
    # Digits
    print('\n=== Digits ===')
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_test = len(X) // 5
    X_train, X_test = X[idx[n_test:]], X[idx[:n_test]]
    y_train, y_test = y[idx[n_test:]], y[idx[:n_test]]
    
    # Adaptive
    afcrs = AdaptiveFCRS(10)
    afcrs.fit(X_train, y_train)
    preds = afcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'Adaptive: {acc:.2%}')
    
    # Fixed
    ffcrs = FixedFCRS(10, 20)
    ffcrs.fit(X_train, y_train)
    preds = ffcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'Fixed(20): {acc:.2%}')
    
    # Iris
    print('\n=== Iris ===')
    iris = load_iris()
    X, y = iris.data, iris.target
    
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_test = len(X) // 5
    X_train, X_test = X[idx[n_test:]], X[idx[:n_test]]
    y_train, y_test = y[idx[n_test:]], y[idx[:n_test]]
    
    # Adaptive
    afcrs = AdaptiveFCRS(3)
    afcrs.fit(X_train, y_train)
    preds = afcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'Adaptive: {acc:.2%}')
    
    # Fixed
    ffcrs = FixedFCRS(3, 4)
    ffcrs.fit(X_train, y_train)
    preds = ffcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'Fixed(4): {acc:.2%}')


if __name__ == "__main__":
    test()
