"""
验证: 更难任务下，维度是否影响误差？
目标: 找到"维度有价值"的场景
"""

import numpy as np
import random


class HardEnv:
    """更难的任务: 高维输入 + 复杂模式"""
    
    def __init__(self, input_dim=50, n_classes=10):
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        # 类中心更靠近，需要更多维度来区分
        self.class_centers = {}
        for i in range(n_classes):
            # 低范数中心 - 更难区分
            self.class_centers[i] = np.random.randn(input_dim) * 0.3
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        # 高噪声
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.5


class HardFCRS:
    def __init__(self, input_dim=50, n_classes=10, lr=0.1, lambda_penalty=0.5):
        self.input_dim = input_dim
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.env = HardEnv(input_dim, n_classes)
        self.representations = []
        self.dimension = 10  # 从10维开始
        
        # 初始化表征
        for _ in range(min(3, n_classes)):
            self.representations.append({
                'vector': np.random.randn(input_dim) * 0.1,
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
    
    def step(self):
        x = self.env.generate()
        
        best = self.select(x)
        
        if best is not None:
            # 计算误差
            v = self.representations[best]['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            error = np.linalg.norm(x_sub - v)
            
            # 学习
            self.representations[best]['vector'][:self.dimension] += self.lr * (x_sub - v)
            self.representations[best]['count'] += 1
            
            # 维度扩张
            gain = random.random() * 0.5
            if gain > self.lambda_penalty and self.dimension < self.input_dim:
                self.dimension += 1
            
            return error
        
        return None


def test_harder_tasks():
    """测试不同难度任务"""
    print('='*60)
    print('Test: Does dimension matter in HARD tasks?')
    print('='*60)
    
    configs = [
        ('Easy', 10, 3, 0.1),   # 简单
        ('Medium', 20, 5, 0.3),  # 中等
        ('Hard', 50, 10, 0.5),  # 难
    ]
    
    results = []
    
    for name, dim, n_cls, noise in configs:
        print(f'\n=== {name} (dim={dim}, classes={n_cls}) ===')
        
        # 测试不同λ
        for lam in [0.0, 0.3, 0.6]:
            random.seed(42)
            np.random.seed(42)
            
            fcrs = HardFCRS(input_dim=dim, n_classes=n_cls, lr=0.1, lambda_penalty=lam)
            
            # 训练
            errors = []
            for _ in range(500):
                err = fcrs.step()
                if err:
                    errors.append(err)
            
            # 测量
            test_errors = []
            for _ in range(100):
                err = fcrs.step()
                if err:
                    test_errors.append(err)
            
            avg_dim = fcrs.dimension
            avg_error = np.mean(test_errors)
            
            results.append({
                'task': name,
                'lambda': lam,
                'dimension': avg_dim,
                'error': avg_error
            })
            
            print(f'λ={lam}: dim={avg_dim}, error={avg_error:.4f}')
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    # 分析
    print('\nTask vs Dimension correlation:')
    for name, _, _, _ in configs:
        task_results = [r for r in results if r['task'] == name]
        
        dims = [r['dimension'] for r in task_results]
        errs = [r['error'] for r in task_results]
        
        # 检查是否相关
        if len(set(dims)) > 1:
            dim_range = max(dims) - min(dims)
            err_range = max(errs) - min(errs)
            
            print(f'{name}: dim_range={dim_range}, err_range={err_range:.4f}')
            
            if err_range > 0.01:
                print(f'  -> YES! Error changes with dimension')
            else:
                print(f'  -> NO! Error does not change')
        else:
            print(f'{name}: dimension fixed')


if __name__ == "__main__":
    test_harder_tasks()
