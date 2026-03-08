"""
Debug: 检查是否只有1个表征在工作
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []
        self.activation_count = 0


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


class TrackFCRS:
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
            self.representations[best].activation_count += 1
            self.errors.append(error)


# 运行并跟踪
random.seed(42)
np.random.seed(42)

fcrs = TrackFCRS(lambda_penalty=0)

for _ in range(100):
    fcrs.step()

print('Activation counts:')
for i, rep in enumerate(fcrs.representations):
    print(f'  Rep {i}: {rep.activation_count} times')

print(f'\nTotal activations: {sum(r.activation_count for r in fcrs.representations)}')

# 检查每个表征是否被更新
print('\nRepresentation changes (first 3 dims):')
for i, rep in enumerate(fcrs.representations):
    print(f'  Rep {i}: {rep.vector[:3]}')
