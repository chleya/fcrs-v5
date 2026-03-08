"""
更多对比实验: Online Learning, Perceptron, K-Means
"""

import numpy as np


class Environment:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.centers = [np.random.randn(50) * 2 for _ in range(n_classes)]
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.centers[cls] + np.random.randn(50) * 0.1, cls


class OnlineLearning:
    """在线学习 - 每次更新"""
    def __init__(self, dim=10):
        self.dim = dim
        self.weights = np.random.randn(dim) * 0.1
        self.bias = 0
    
    def predict(self, x):
        return np.dot(x[:self.dim], self.weights) + self.bias
    
    def learn(self, x, target):
        pred = self.predict(x)
        error = target - pred
        
        # 在线更新
        self.weights += 0.1 * error * x[:self.dim]
        self.bias += 0.1 * error


class Perceptron:
    """感知机"""
    def __init__(self, dim=10):
        self.dim = dim
        self.weights = np.random.randn(dim) * 0.1
    
    def predict(self, x):
        return np.dot(x[:self.dim], self.weights)
    
    def predict_class(self, x):
        return 1 if self.predict(x) > 0 else 0
    
    def learn(self, x, target_cls):
        pred_cls = self.predict_class(x)
        if pred_cls != target_cls:
            self.weights += 0.1 * (target_cls - pred_cls) * x[:self.dim]


class KMeans:
    """K-Means聚类"""
    def __init__(self, n_clusters=3, dim=10):
        self.n_clusters = n_clusters
        self.dim = dim
        self.centers = [np.random.randn(dim) * 0.1 for _ in range(n_clusters)]
    
    def predict(self, x):
        distances = [np.linalg.norm(x[:self.dim] - c) for c in self.centers]
        return min(distances)
    
    def fit(self, X, epochs=10):
        for _ in range(epochs):
            # 分配
            labels = []
            for x in X:
                distances = [np.linalg.norm(x[:self.dim] - c) for c in self.centers]
                labels.append(np.argmin(distances))
            
            # 更新
            for i in range(self.n_clusters):
                points = [X[j] for j in range(len(X)) if labels[j] == i]
                if points:
                    self.centers[i] = np.mean(points, axis=0)


class ImprovedFCRS:
    """改进的FCRS"""
    def __init__(self, lambda_val=0.5):
        self.lambda_val = lambda_val
        self.dimension = 10
        self.representations = []
        for _ in range(5):
            self.representations.append({
                'vector': np.random.randn(50) * 0.1,
                'count': 0
            })
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            
            norm_v = np.linalg.norm(v)
            norm_x = np.linalg.norm(x_sub)
            
            if norm_v > 0.01 and norm_x > 0.01:
                score = np.dot(v, x_sub) / (norm_v * norm_x)
            else:
                score = -1
            
            if score > best_score:
                best_score = score
                best = i
        
        return best
    
    def step(self, x, cls):
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            
            error = np.linalg.norm(x_sub - v)
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x_sub - v)
            self.representations[best]['count'] += 1
            
            return error
        
        return None


def run_comparisons():
    """运行对比"""
    print('='*60)
    print('Extended Comparison')
    print('='*60)
    
    env = Environment(n_classes=3)
    
    results = {}
    
    # 1. Improved FCRS
    print('\n1. Improved FCRS')
    fcrs = ImprovedFCRS()
    
    for _ in range(1000):
        x, cls = env.generate()
        fcrs.step(x, cls)
    
    errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = fcrs.step(x, cls)
        if err:
            errors.append(err)
    
    results['FCRS'] = np.mean(errors)
    print(f'   Error: {results["FCRS"]:.4f}')
    
    # 2. Online Learning
    print('\n2. Online Learning')
    ol = OnlineLearning(dim=10)
    
    for _ in range(1000):
        x, cls = env.generate()
        ol.learn(x, cls)
    
    errors = []
    for _ in range(100):
        x, cls = env.generate()
        pred = ol.predict(x)
        # 近似误差
        errors.append(abs(pred - cls))
    
    results['Online Learning'] = np.mean(errors)
    print(f'   Error: {results["Online Learning"]:.4f}')
    
    # 3. K-Means
    print('\n3. K-Means')
    km = KMeans(n_clusters=3, dim=10)
    
    # 训练数据
    X_train = []
    for _ in range(500):
        x, cls = env.generate()
        X_train.append(x)
    
    km.fit(np.array(X_train))
    
    errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = km.predict(x)
        errors.append(err)
    
    results['K-Means'] = np.mean(errors)
    print(f'   Error: {results["K-Means"]:.4f}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    for name, error in sorted(results.items(), key=lambda x: x[1]):
        print(f'{name}: {error:.4f}')
    
    best = min(results, key=results.get)
    print(f'\nBest: {best}')


if __name__ == "__main__":
    run_comparisons()
