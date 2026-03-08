"""
真实MNIST数据集测试
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class FCRSClassifier:
    """FCRS分类器"""
    def __init__(self, dim=8):
        self.dim = dim
        self.representations = {}
        self.counts = {}
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            if label not in self.representations:
                self.representations[label] = np.zeros(len(x))
                self.counts[label] = 0
            
            # 累加
            self.representations[label] += x
            self.counts[label] += 1
        
        # 平均
        for label in self.representations:
            self.representations[label] /= max(1, self.counts[label])
    
    def predict(self, X):
        predictions = []
        for x in X:
            best_label = None
            best_dist = float('inf')
            
            for label, rep in self.representations.items():
                d = np.linalg.norm(x - rep)
                if d < best_dist:
                    best_dist = d
                    best_label = label
            
            predictions.append(best_label)
        
        return np.array(predictions)


class NearestNeighbor:
    """最近邻"""
    def __init__(self):
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest = np.argmin(distances)
            predictions.append(self.y_train[nearest])
        return np.array(predictions)


def main():
    print('='*60)
    print('REAL MNIST-like Dataset Test')
    print('='*60)
    
    # 加载真实数据
    print('\nLoading digits dataset...')
    digits = load_digits()
    X, y = digits.data, digits.target
    
    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features')
    print(f'Classes: {len(np.unique(y))}')
    
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')
    
    # 1. FCRS
    print('\n1. FCRS Classifier')
    fcrs = FCRSClassifier()
    fcrs.fit(X_train, y_train)
    preds = fcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f'   Accuracy: {acc:.2%}')
    
    # 2. Nearest Neighbor
    print('\n2. Nearest Neighbor')
    nn = NearestNeighbor()
    nn.fit(X_train, y_train)
    preds = nn.predict(X_test)
    acc_nn = np.mean(preds == y_test)
    print(f'   Accuracy: {acc_nn:.2%}')
    
    # 3. Random Baseline
    print('\n3. Random Baseline')
    random_preds = np.random.randint(0, 10, len(X_test))
    acc_rand = np.mean(random_preds == y_test)
    print(f'   Accuracy: {acc_rand:.2%}')
    
    # Summary
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS:          {acc:.2%}')
    print(f'Nearest Neigh: {acc_nn:.2%}')
    print(f'Random:        {acc_rand:.2%}')


if __name__ == "__main__":
    main()
