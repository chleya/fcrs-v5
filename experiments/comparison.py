"""
对比实验: FCRS vs 现有方法
按照审查建议添加
"""

import numpy as np


class Environment:
    """测试环境"""
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.centers = [np.random.randn(10) * 2 for _ in range(n_classes)]
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.centers[cls] + np.random.randn(10) * 0.1, cls


class FCRS:
    """FCRS系统"""
    def __init__(self, lambda_val=0.5):
        self.lambda_val = lambda_val
        self.representations = []
        self.dimension = 10
        for _ in range(3):
            self.representations.append({'vector': np.random.randn(10) * 0.1})
    
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
    
    def step(self, x):
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            # 学习
            self.representations[best]['vector'][:self.dimension] += 0.1 * (x[:self.dimension] - v)
            
            return error
        
        return None


class FixedDim:
    """固定维度方法"""
    def __init__(self, dim=10):
        self.dim = dim
        self.centers = [np.random.randn(10) * 0.1 for _ in range(3)]
    
    def predict(self, x):
        best = 0
        best_dist = float('inf')
        
        for i, c in enumerate(self.centers):
            d = np.linalg.norm(x[:self.dim] - c[:self.dim])
            if d < best_dist:
                best_dist = d
                best = i
        
        return best_dist
    
    def learn(self, x, cls):
        self.centers[cls] += 0.1 * (x[:self.dim] - self.centers[cls])


class PCA:
    """PCA降维方法"""
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.mean = None
        self.components = None
    
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idx[:self.n_components]]
    
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def predict(self, X):
        X_transformed = self.transform(X)
        # 简单分类器
        return np.mean(X_transformed, axis=1)


def run_comparison():
    """运行对比实验"""
    print('='*60)
    print('FCRS vs Existing Methods Comparison')
    print('='*60)
    
    env = Environment(n_classes=3)
    
    # 1. FCRS
    print('\n1. FCRS (lambda=0.5)')
    fcrs = FCRS(lambda_val=0.5)
    
    for _ in range(500):
        x, cls = env.generate()
        fcrs.step(x)
    
    # 测试
    fcrs_errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = fcrs.step(x)
        if err:
            fcrs_errors.append(err)
    
    print(f'   Error: {np.mean(fcrs_errors):.4f}')
    print(f'   Dimension: {fcrs.dimension}')
    
    # 2. Fixed Dimension
    print('\n2. Fixed Dimension (dim=10)')
    fixed = FixedDim(dim=10)
    
    for _ in range(500):
        x, cls = env.generate()
        fixed.learn(x, cls)
    
    fixed_errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = fixed.predict(x)
        fixed_errors.append(err)
    
    print(f'   Error: {np.mean(fixed_errors):.4f}')
    
    # 3. PCA
    print('\n3. PCA (n_components=10)')
    # 训练PCA
    X_train = []
    for _ in range(500):
        x, cls = env.generate()
        X_train.append(x)
    
    pca = PCA(n_components=10)
    pca.fit(np.array(X_train))
    
    pca_errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = np.abs(pca.predict(x.reshape(1, -1))[0])
        pca_errors.append(err)
    
    print(f'   Error: {np.mean(pca_errors):.4f}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'FCRS:       {np.mean(fcrs_errors):.4f}')
    print(f'Fixed Dim:  {np.mean(fixed_errors):.4f}')
    print(f'PCA:        {np.mean(pca_errors):.4f}')
    
    # 判断
    results = {
        'FCRS': np.mean(fcrs_errors),
        'Fixed': np.mean(fixed_errors),
        'PCA': np.mean(pca_errors)
    }
    
    best = min(results, key=results.get)
    print(f'\nBest method: {best}')


if __name__ == "__main__":
    run_comparison()
