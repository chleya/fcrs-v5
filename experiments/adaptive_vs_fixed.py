"""
更多测试：自适应维度
"""

import numpy as np
from sklearn.datasets import load_digits, load_iris


class AdaptiveFCRS:
    """自适应维度FCRS"""
    
    def __init__(self):
        self.reps = {i: np.random.randn(64) * 0.1 for i in range(10)}
        self.dims = {i: 10 for i in range(10)}
        self.counts = {i: 0 for i in range(10)}
    
    def predict(self, x):
        best_label = None
        best_dist = float('inf')
        
        for label, rep in self.reps.items():
            dim = self.dims[label]
            d = np.linalg.norm(x[:dim] - rep[:dim])
            if d < best_dist:
                best_dist = d
                best_label = label
        
        return best_label
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            dim = self.dims[label]
            
            # 学习
            self.reps[label][:dim] += 0.5 * (x[:dim] - self.reps[label][:dim])
            self.counts[label] += 1
            
            # 动态调整维度
            error = np.linalg.norm(x[:dim] - self.reps[label][:dim])
            
            if error < 0.3 and self.dims[label] < 64:
                self.dims[label] += 1
            elif error > 1.0 and self.dims[label] > 5:
                self.dims[label] -= 1


class FixedFCRS:
    """固定维度FCRS"""
    
    def __init__(self, dim=20):
        self.dim = dim
        self.reps = {i: np.random.randn(64) * 0.1 for i in range(10)}
        self.counts = {i: 0 for i in range(10)}
    
    def predict(self, x):
        best_label = None
        best_dist = float('inf')
        
        for label, rep in self.reps.items():
            d = np.linalg.norm(x[:self.dim] - rep[:self.dim])
            if d < best_dist:
                best_dist = d
                best_label = label
        
        return best_label
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            self.reps[label][:self.dim] += 0.5 * (x[:self.dim] - self.reps[label][:self.dim])
            self.counts[label] += 1


def test():
    print('='*60)
    print('Adaptive vs Fixed Dimension')
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
    afcrs = AdaptiveFCRS()
    afcrs.fit(X_train, y_train)
    preds = [afcrs.predict(x) for x in X_test]
    acc_adaptive = np.mean(np.array(preds) == y_test)
    print(f'Adaptive: {acc_adaptive:.2%}')
    
    # Fixed
    ffcrs = FixedFCRS(dim=20)
    ffcrs.fit(X_train, y_train)
    preds = [ffcrs.predict(x) for x in X_test]
    acc_fixed = np.mean(np.array(preds) == y_test)
    print(f'Fixed: {acc_fixed:.2%}')
    
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
    afcrs = AdaptiveFCRS()
    afcrs.fit(X_train, y_train)
    preds = [afcrs.predict(x) for x in X_test]
    acc_adaptive = np.mean(np.array(preds) == y_test)
    print(f'Adaptive: {acc_adaptive:.2%}')
    
    # Fixed
    ffcrs = FixedFCRS(dim=4)
    ffcrs.fit(X_train, y_train)
    preds = [ffcrs.predict(x) for x in X_test]
    acc_fixed = np.mean(np.array(preds) == y_test)
    print(f'Fixed: {acc_fixed:.2%}')


if __name__ == "__main__":
    test()
