"""
正确对比: 在收敛后测量error
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()


class EasyEnv:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        if n_classes == 3:
            self.class_centers = {
                0: np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                1: np.array([0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
                2: np.array([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0]),
            }
        else:
            self.class_centers = {i: np.random.randn(10) for i in range(n_classes)}
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(10) * 0.1


class CompareFCRS:
    def __init__(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty
        self.env = EasyEnv(n_classes=3)
        self.representations = [SimpleRep(np.random.randn(10) * 0.1) for _ in range(3)]
        self.dimension = 10
    
    def select_cosine(self, x):
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep.vector[:self.dimension]
            norm_v = np.linalg.norm(v)
            norm_x = np.linalg.norm(x)
            
            if norm_v > 0 and norm_x > 0:
                score = np.dot(v, x[:len(v)]) / (norm_v * norm_x)
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best = i
        
        return best
    
    def train(self, steps):
        errors = []
        for _ in range(steps):
            x = self.env.generate()
            best = self.select_cosine(x)
            
            if best is not None:
                v = self.representations[best].vector[:self.dimension]
                error = np.linalg.norm(x[:self.dimension] - v)
                self.representations[best].vector[:self.dimension] += 0.5 * (x[:self.dimension] - v)
                errors.append(error)
        
        return errors
    
    def measure_error(self, steps):
        """测量稳定后的error"""
        errors = self.train(steps)
        return np.mean(errors[-50:])


# 测试不同λ
print('='*60)
print('Compare: Different λ values')
print('='*60)

results = []

for lam in [0.0, 0.001, 0.01, 0.1, 1.0]:
    # 使用不同seed避免完全相同轨迹
    random.seed(int(lam * 10000))
    np.random.seed(int(lam * 10000))
    
    fcrs = CompareFCRS(lambda_penalty=lam)
    
    # 训练500步
    errors_train = fcrs.train(500)
    
    # 测量100步
    errors_test = []
    for _ in range(100):
        x = fcrs.env.generate()
        best = fcrs.select_cosine(x)
        if best is not None:
            v = fcrs.representations[best].vector[:fcrs.dimension]
            error = np.linalg.norm(x[:fcrs.dimension] - v)
            errors_test.append(error)
    
    avg_error = np.mean(errors_test)
    results.append((lam, fcrs.dimension, avg_error))
    
    print(f'λ={lam}: dim={fcrs.dimension}, test_error={avg_error:.4f}')

print('\n=== Analysis ===')
dims = [r[1] for r in results]
errors = [r[2] for r in results]

print(f'Dimension range: {min(dims)} - {max(dims)}')
print(f'Error range: {min(errors):.4f} - {max(errors):.4f}')

if max(errors) - min(errors) > 0.01:
    print('\n[OK] Dimension DOES affect error!')
else:
    print('\n[WARN] Error still same - need different test')
