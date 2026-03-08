"""
Debug: 为什么Error不随维度变化？
测试不同任务难度下λ对error的影响
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class TaskEnv:
    """不同难度的任务"""
    def __init__(self, n_classes=3, noise=0.3):
        self.n_classes = n_classes
        self.noise = noise
        self.input_dim = 10
        # 类别中心
        self.class_centers = {}
        for i in range(n_classes):
            # 简单任务：类中心距离远；难任务：距离近
            if n_classes <= 3:
                # 简单：中心在正交方向
                center = np.zeros(self.input_dim)
                center[i * 3] = 3.0
                self.class_centers[i] = center
            else:
                # 难：中心靠近原点
                self.class_centers[i] = np.random.randn(self.input_dim) * 0.5
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * self.noise


class DebugFCRS:
    def __init__(self, n_classes=3, noise=0.3, lr=0.1, lambda_penalty=0.1):
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.representations = []
        self.env = TaskEnv(n_classes, noise)
        
        self.errors = []
        self.dimension = 10
        
        for _ in range(3):
            x = self.env.generate_input()
            self.representations.append(SimpleRep(x))
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            v = rep.vector[:self.dimension]
            score = np.dot(v, x) / (np.linalg.norm(v) + 1e-8)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def step(self):
        x = self.env.generate_input()
        
        active = self.select(x)
        
        if active is not None:
            error_vec = x - active.vector[:self.dimension]
            error = np.linalg.norm(error_vec)
            
            active.vector[:self.dimension] += self.lr * error_vec
            active.fitness_history.append(-error)
            
            self.errors.append(error)
            
            # 结构决策
            gain = random.random() * 0.3
            if len(self.representations) < 5 and gain > self.lambda_penalty:
                self.dimension += 1


def run_debug(n_classes, noise, lambda_penalty, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = DebugFCRS(n_classes=n_classes, noise=noise, lr=0.1, lambda_penalty=lambda_penalty)
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'n_classes': n_classes,
        'noise': noise,
        'lambda': lambda_penalty,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
    }


def main():
    print('='*60)
    print('Debug: Error vs Dimension')
    print('='*60)
    
    # 测试不同任务难度
    tasks = [
        ('Easy', 3, 0.1),
        ('Medium', 5, 0.3),
        ('Hard', 10, 0.5),
    ]
    
    lambdas = [0.001, 0.1, 1.0]
    
    results = []
    
    for task_name, n_classes, noise in tasks:
        print(f'\n=== {task_name} (classes={n_classes}, noise={noise}) ===')
        
        for lam in lambdas:
            for seed in [0, 1, 2]:
                r = run_debug(n_classes, noise, lam, seed)
                results.append(r)
                print(f"λ={lam}: dim={r['final_dimension']}, error={r['avg_error']:.4f}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    for task_name, n_classes, noise in tasks:
        print(f'\n{task_name}:')
        print(f'{"λ":<10} {"dimension":<12} {"error":<12}')
        
        task_results = [r for r in results if r['n_classes'] == n_classes]
        
        for lam in lambdas:
            lam_results = [r for r in task_results if r['lambda'] == lam]
            avg_dim = np.mean([r['final_dimension'] for r in lam_results])
            avg_err = np.mean([r['avg_error'] for r in lam_results])
            print(f'{lam:<10} {avg_dim:<12.1f} {avg_err:<12.4f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/debug_error/results.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print('\nSaved to results.json')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    for task_name, n_classes, noise in tasks:
        task_results = [r for r in results if r['n_classes'] == n_classes]
        
        dims = []
        errors = []
        
        for lam in lambdas:
            lam_results = [r for r in task_results if r['lambda'] == lam]
            dims.append(np.mean([r['final_dimension'] for r in lam_results]))
            errors.append(np.mean([r['avg_error'] for r in lam_results]))
        
        print(f'\n{task_name}:')
        print(f'  Dimension range: {min(dims):.1f} - {max(dims):.1f}')
        print(f'  Error range: {min(errors):.4f} - {max(errors):.4f}')
        
        if max(errors) - min(errors) > 0.1:
            print(f'  [OK] λ DOES affect error!')
        else:
            print(f'  [WARN] λ does NOT affect error')


if __name__ == "__main__":
    main()
