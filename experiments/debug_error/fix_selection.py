"""
Debug 3: 修复选择机制
问题: dot product 不区分表征
修复: 使用cosine similarity或euclidean distance
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class EasyEnv:
    def __init__(self):
        self.input_dim = 10
        self.class_centers = {
            0: np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1: np.array([0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            2: np.array([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0]),
        }
    
    def generate_input(self):
        cls = np.random.randint(0, 3)
        return self.class_centers[cls] + np.random.randn(10) * 0.01


class FixedFCRS:
    """修复后的FCRS - 使用cosine similarity"""
    
    def __init__(self, lr=0.5):
        self.lr = lr
        self.representations = []
        self.env = EasyEnv()
        
        for _ in range(3):
            self.representations.append(SimpleRep(np.random.randn(10) * 0.1))
        
        self.errors = []
    
    def select_cosine(self, x):
        """修复: 使用cosine similarity"""
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            # Cosine similarity
            norm_rep = np.linalg.norm(rep.vector)
            norm_x = np.linalg.norm(x)
            
            if norm_rep > 0 and norm_x > 0:
                score = np.dot(rep.vector, x) / (norm_rep * norm_x)
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best = i
        
        return best
    
    def step(self):
        x = self.env.generate_input()
        
        # 使用修复后的选择
        best = self.select_cosine(x)
        
        if best is not None:
            error = np.linalg.norm(x - self.representations[best].vector)
            
            # 学习
            self.representations[best].vector += self.lr * (x - self.representations[best].vector)
            
            self.representations[best].fitness_history.append(-error)
            self.errors.append(error)
            
            return error
        
        return None


# 测试修复后的系统
print('='*60)
print('Test: Fixed Selection Mechanism')
print('='*60)

np.random.seed(0)
random.seed(0)

fcrs = FixedFCRS(lr=0.5)

print('\nFirst 5 steps:')
for i in range(5):
    err = fcrs.step()
    print(f'Step {i}: error = {err:.4f}')

print('\nLast 5 steps:')
for i in range(495, 500):
    err = fcrs.step()
    print(f'Step {i}: error = {err:.4f}')

print(f'\nAverage error (first 10): {np.mean(fcrs.errors[:10]):.4f}')
print(f'Average error (last 10): {np.mean(fcrs.errors[-10:]):.4f}')

# 检查表征
print('\nFinal representations:')
for i, rep in enumerate(fcrs.representations):
    print(f'Rep {i}: norm = {np.linalg.norm(rep.vector):.4f}, first 3 = {rep.vector[:3]}')

# 测试区分度
print('\nDiscrimination test:')
for cls in range(3):
    x = fcrs.env.class_centers[cls] + np.random.randn(10) * 0.01
    
    scores = []
    for rep in fcrs.representations:
        norm_rep = np.linalg.norm(rep.vector)
        norm_x = np.linalg.norm(x)
        score = np.dot(rep.vector, x) / (norm_rep * norm_x + 1e-8)
        scores.append(score)
    
    print(f'Class {cls}: scores = {[f"{s:.3f}" for s in scores]}, selected = {np.argmax(scores)}')
