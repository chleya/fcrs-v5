"""
Debug 2: 验证学习是否真的在工作
"""

import numpy as np
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class EasyEnv:
    """简单可分离任务"""
    def __init__(self):
        self.input_dim = 10
        # 3个类，中心相距很远
        self.class_centers = {
            0: np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1: np.array([0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            2: np.array([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0]),
        }
    
    def generate_input(self):
        cls = np.random.randint(0, 3)
        return self.class_centers[cls] + np.random.randn(10) * 0.01  # 很低噪声!


class TestFCRS:
    def __init__(self, lr=0.5):
        self.lr = lr
        self.representations = []
        self.env = EasyEnv()
        
        # 初始化 - 用随机向量
        for _ in range(3):
            self.representations.append(SimpleRep(np.random.randn(10) * 0.1))
        
        self.errors = []
    
    def step(self):
        x = self.env.generate_input()
        
        # 选择最佳表征
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            score = np.dot(rep.vector, x)
            if score > best_score:
                best_score = score
                best = i
        
        # 计算误差
        error = np.linalg.norm(x - self.representations[best].vector)
        
        # 学习 - 直接复制输入
        self.representations[best].vector = self.representations[best].vector + self.lr * (x - self.representations[best].vector)
        
        self.representations[best].fitness_history.append(-error)
        self.errors.append(error)
        
        return error


# 运行测试
print('='*60)
print('Test: Does learning work?')
print('='*60)

np.random.seed(0)
random.seed(0)

fcrs = TestFCRS(lr=0.5)

print('\nFirst 10 steps:')
for i in range(10):
    err = fcrs.step()
    print(f'Step {i}: error = {err:.4f}')

print('\nLast 10 steps:')
for i in range(490, 500):
    err = fcrs.step()
    print(f'Step {i}: error = {err:.4f}')

print(f'\nAverage error (first 10): {np.mean(fcrs.errors[:10]):.4f}')
print(f'Average error (last 10): {np.mean(fcrs.errors[-10:]):.4f}')

# 检查表征
print('\nFinal representations:')
for i, rep in enumerate(fcrs.representations):
    print(f'Rep {i}: norm = {np.linalg.norm(rep.vector):.4f}')
    print(f'  vector[:3] = {rep.vector[:3]}')

# 测试分类
print('\nClassification test:')
correct = 0
total = 100

for _ in range(total):
    x = fcrs.env.generate_input()
    true_cls = fcrs.env.class_centers.keys()
    
    best = None
    best_score = -float('inf')
    
    for i, rep in enumerate(fcrs.representations):
        score = np.dot(rep.vector, x)
        if score > best_score:
            best_score = score
            best = i
    
    # 简单判断
    print(f'Input class: ?, Selected rep: {best}, Score: {best_score:.2f}')
