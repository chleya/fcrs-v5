"""
Debug: 为什么Error还是一样？
测试: 在不同训练阶段记录Error
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


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


class TestFCRS:
    def __init__(self, lambda_penalty):
        self.lambda_penalty = lambda_penalty
        self.env = EasyEnv()
        self.representations = [SimpleRep(np.random.randn(10) * 0.1) for _ in range(3)]
        self.dimension = 10
        self.errors = []
    
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
    
    def step(self):
        x = self.env.generate()
        
        best = self.select_cosine(x)
        
        if best is not None:
            v = self.representations[best].vector[:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            self.representations[best].vector[:self.dimension] += 0.5 * (x[:self.dimension] - v)
            self.errors.append(error)
            
            # 结构决策
            gain = random.random() * 0.3
            if len(self.representations) < 5 and gain > self.lambda_penalty:
                self.dimension += 1


# 测试两个极端λ值
print('='*60)
print('Compare: λ=0 (max dim) vs λ=1 (min dim)')
print('='*60)

# λ=0, max dimension
random.seed(0)
np.random.seed(0)

fcrs1 = TestFCRS(lambda_penalty=0)
for _ in range(500):
    fcrs1.step()

print(f'\nλ=0:')
print(f'  Final dimension: {fcrs1.dimension}')
print(f'  Error at step 100: {fcrs1.errors[100]:.4f}')
print(f'  Error at step 200: {fcrs1.errors[200]:.4f}')
print(f'  Error at step 300: {fcrs1.errors[300]:.4f}')
print(f'  Error at step 400: {fcrs1.errors[400]:.4f}')
print(f'  Error at step 499: {fcrs1.errors[499]:.4f}')

# λ=1, min dimension  
random.seed(0)
np.random.seed(0)

fcrs2 = TestFCRS(lambda_penalty=1.0)
for _ in range(500):
    fcrs2.step()

print(f'\nλ=1:')
print(f'  Final dimension: {fcrs2.dimension}')
print(f'  Error at step 100: {fcrs2.errors[100]:.4f}')
print(f'  Error at step 200: {fcrs2.errors[200]:.4f}')
print(f'  Error at step 300: {fcrs2.errors[300]:.4f}')
print(f'  Error at step 400: {fcrs2.errors[400]:.4f}')
print(f'  Error at step 499: {fcrs2.errors[499]:.4f}')

# 检查表征学习
print(f'\n=== Representation Analysis ===')
print(f'\nλ=0 final rep[0][:3]: {fcrs1.representations[0].vector[:3]}')
print(f'λ=1 final rep[0][:3]: {fcrs2.representations[0].vector[:3]}')
