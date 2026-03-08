"""
FCRS 基准测试
在标准数据集上测试
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine


class FCRSClassifier:
    """FCRS分类器"""
    
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.centers = {}
        self.counts = {}
    
    def fit(self, X, y):
        for x, label in zip(X, y):
            if label not in self.centers:
                self.centers[label] = np.zeros(len(x))
                self.counts[label] = 0
            
            self.centers[label] += x
            self.counts[label] += 1
        
        for label in self.centers:
            self.centers[label] /= max(1, self.counts[label])
    
    def predict(self, X):
        preds = []
        for x in X:
            best_label = min(self.centers.keys(),
                          key=lambda l: np.linalg.norm(x - self.centers[l]))
            preds.append(best_label)
        return np.array(preds)


class KNN:
    """K近邻"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        preds = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest = np.argsort(distances)[:self.k]
            labels = self.y_train[nearest]
            preds.append(np.bincount(labels).argmax())
        return np.array(preds)


class RandomForest:
    """简化随机森林"""
    
    def __init__(self, n_trees=5):
        self.n_trees = n_trees
        self.trees = []
    
    def fit(self, X, y):
        for _ in range(self.n_trees):
            # 随机采样
            idx = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[idx]
            y_sample = y[idx]
            
            # 训练决策树(简化)
            tree = {}
            for x, label in zip(X_sample, y_sample):
                key = tuple(x[:5] > np.mean(x[:5]))
                tree[key] = label
            
            self.trees.append(tree)
    
    def predict(self, X):
        preds = []
        for x in X:
            votes = []
            for tree in self.trees:
                key = tuple(x[:5] > np.mean(x[:5]))
                if key in tree:
                    votes.append(tree[key])
            
            if votes:
                preds.append(np.bincount(votes).argmax())
            else:
                preds.append(0)
        return np.array(preds)


def benchmark(dataset_name, load_fn):
    """基准测试"""
    print(f'\n=== {dataset_name} ===')
    
    X, y = load_fn(return_X_y=True)
    
    # 划分
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    n_test = len(X) // 5
    X_train, X_test = X[idx[n_test:]], X[idx[:n_test]]
    y_train, y_test = y[idx[n_test:]], y[idx[:n_test]]
    
    # 归一化
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    results = {}
    
    # FCRS
    fcrs = FCRSClassifier(n_classes=len(np.unique(y)))
    fcrs.fit(X_train, y_train)
    preds = fcrs.predict(X_test)
    acc = np.mean(preds == y_test)
    results['FCRS'] = acc
    print(f'FCRS: {acc:.2%}')
    
    # KNN
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    acc = np.mean(preds == y_test)
    results['KNN'] = acc
    print(f'KNN: {acc:.2%}')
    
    # Random Forest
    rf = RandomForest(n_trees=5)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = np.mean(preds == y_test)
    results['RF'] = acc
    print(f'RF: {acc:.2%}')
    
    return results


def main():
    print('='*60)
    print('FCRS Benchmark Tests')
    print('='*60)
    
    all_results = {}
    
    # Iris
    all_results['Iris'] = benchmark('Iris', load_iris)
    
    # Wine
    all_results['Wine'] = benchmark('Wine', load_wine)
    
    # Breast Cancer
    all_results['Cancer'] = benchmark('Breast Cancer', load_breast_cancer)
    
    # Summary
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    print('\nDataset   | FCRS   | KNN    | RF')
    print('-' * 40)
    for name, results in all_results.items():
        print(f'{name:10} | {results["FCRS"]:.2%} | {results["KNN"]:.2%} | {results["RF"]:.2%}')


if __name__ == "__main__":
    main()
