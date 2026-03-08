"""
最终验证: 用更大的gain概率
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()


class EasyEnv:
    def __init__(self):
        self.class_centers = {
            0: np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1: np.array([0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            2: np.array([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0]),
        }
    
    def generate(self):
        cls = np.random.randint(0, 3)
        return self.class_centers[cls] + np.random.randn(10) * 0.1


class FinalFCRS:
    def __init__(self, lambda_penalty, gain_prob=0.8):
        self.lambda_penalty = lambda_penalty
        self.gain_prob = gain_prob
        self.env = EasyEnv()
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
    
    def train_and_measure(self, steps=500):
        errors = []
        for _ in range(steps):
            x = self.env.generate()
            best = self.select_cosine(x)
            
            if best is not None:
                v = self.representations[best].vector[:self.dimension]
                error = np.linalg.norm(x[:self.dimension] - v)
                self.representations[best].vector[:self.dimension] += 0.5 * (x[:self.dimension] - v)
                errors.append(error)
                
                # Spawn - 用更大的概率
                gain = random.random()
                if gain < self.gain_prob:  # 80%概率尝试扩张
                    if len(self.representations) < 5 and gain > self.lambda_penalty:
                        self.dimension += 1
        
        return np.mean(errors[-50:]), self.dimension


print('='*60)
print('FINAL TEST: Dimension vs Error')
print('='*60)

# 固定gain_prob，变化λ
results = []

for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    random.seed(42)
    np.random.seed(42)
    
    fcrs = FinalFCRS(lambda_penalty=lam, gain_prob=0.8)
    error, dim = fcrs.train_and_measure(500)
    
    results.append((lam, dim, error))
    print(f'λ={lam}: dim={dim}, error={error:.4f}')

print('\n=== CORRELATION ===')
dims = [r[1] for r in results]
errors = [r[2] for r in results]

print(f'Dims: {dims}')
print(f'Errors: {[f"{e:.3f}" for e in errors]}')

# 相关性
if dims != [dims[0]]*len(dims):
    print('\n[SUCCESS] Dimensions ARE different!')
else:
    print('\n[FAIL] All same dimensions')

# 找最优
best = min(results, key=lambda x: x[2])
print(f'\nBest: λ={best[0]}, dim={best[1]}, error={best[2]:.4f}')
