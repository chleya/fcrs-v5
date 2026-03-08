"""
修复版对比实验
"""

import numpy as np


class Environment:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.centers = [np.random.randn(10) * 2 for _ in range(n_classes)]
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.centers[cls] + np.random.randn(10) * 0.1, cls


class OnlineLearning:
    """在线学习"""
    def __init__(self, dim=10):
        self.dim = dim
        self.weights = [np.random.randn(dim) * 0.1 for _ in range(3)]
    
    def predict(self, x):
        # 找最近的类中心
        distances = [np.linalg.norm(x[:self.dim] - w) for w in self.weights]
        return np.argmin(distances)
    
    def learn(self, x, cls):
        # 更新最近中心
        self.weights[cls] += 0.1 * (x[:self.dim] - self.weights[cls])


class KMeans:
    """K-Means"""
    def __init__(self, n_clusters=3, dim=10):
        self.n_clusters = n_clusters
        self.dim = dim
        self.centers = [np.random.randn(dim) * 0.1 for _ in range(n_clusters)]
    
    def predict(self, x):
        distances = [np.linalg.norm(x[:self.dim] - c) for c in self.centers]
        return min(distances)
    
    def fit(self, X, epochs=10):
        for _ in range(epochs):
            labels = []
            for x in X:
                distances = [np.linalg.norm(x[:self.dim] - c) for c in self.centers]
                labels.append(np.argmin(distances))
            
            for i in range(self.n_clusters):
                points = [X[j] for j in range(len(X)) if labels[j] == i]
                if points:
                    self.centers[i] = np.mean(points, axis=0)


class FCRS:
    """FCRS"""
    def __init__(self):
        self.dimension = 10
        self.reps = [{'v': np.random.randn(10)*0.1} for _ in range(3)]
    
    def step(self, x, cls):
        # 找最近表征
        best = 0
        best_err = float('inf')
        
        for i, r in enumerate(self.reps):
            e = np.linalg.norm(x[:self.dimension] - r['v'][:self.dimension])
            if e < best_err:
                best_err = e
                best = i
        
        # 学习
        self.reps[best]['v'][:self.dimension] += 0.5 * (x[:self.dimension] - self.reps[best]['v'][:self.dimension])
        
        return best_err


def main():
    print('='*60)
    print('Comparison')
    print('='*60)
    
    env = Environment(n_classes=3)
    
    # 1. FCRS
    fcrs = FCRS()
    for _ in range(500):
        x, c = env.generate()
        fcrs.step(x, c)
    
    errs = [fcrs.step(*env.generate()) for _ in range(100)]
    print(f'FCRS: {np.mean(errs):.4f}')
    
    # 2. Online Learning
    ol = OnlineLearning(dim=10)
    for _ in range(500):
        x, c = env.generate()
        ol.learn(x, c)
    
    preds = [ol.predict(x) for x, c in [env.generate() for _ in range(100)]]
    # 用分类准确率
    correct = sum(1 for i, (x, c) in enumerate([env.generate() for _ in range(100)]) if preds[i] == c)
    print(f'Online Learning: {correct/100:.2%}')
    
    # 3. K-Means
    X = [x for x, c in [env.generate() for _ in range(500)]]
    km = KMeans(dim=10)
    km.fit(X)
    
    errs = [km.predict(x) for x, c in [env.generate() for _ in range(100)]]
    print(f'K-Means: {np.mean(errs):.4f}')


if __name__ == "__main__":
    main()
