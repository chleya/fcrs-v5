"""
Experiment 1-5: Fixed Selection Mechanism
使用cosine similarity修复选择机制
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class TaskEnv:
    def __init__(self, n_classes=3, noise=0.3):
        self.n_classes = n_classes
        self.noise = noise
        self.input_dim = 10
        self.class_centers = {i: np.random.randn(10) * 0.5 for i in range(n_classes)}
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(10) * self.noise


class FixedFCRS:
    """修复后的FCRS - 使用cosine similarity"""
    
    def __init__(self, n_classes=3, noise=0.3, lr=0.5, lambda_penalty=0.1):
        self.n_classes = n_classes
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.env = TaskEnv(n_classes, noise)
        self.representations = []
        self.dimension = 10
        self.spawn_count = 0
        
        self.errors = []
        
        # 初始化
        for _ in range(3):
            x = self.env.generate()
            self.representations.append(SimpleRep(x))
    
    def select_cosine(self, x):
        """使用cosine similarity选择"""
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
            
            # 学习
            self.representations[best].vector[:self.dimension] += self.lr * (x[:self.dimension] - v)
            
            self.representations[best].fitness_history.append(-error)
            self.errors.append(error)
            
            # 结构决策
            gain = random.random() * 0.3
            if len(self.representations) < 5 and gain > self.lambda_penalty:
                self.dimension += 1
                self.spawn_count += 1


def run_exp(n_classes, noise, lambda_penalty, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = FixedFCRS(n_classes=n_classes, noise=noise, lr=0.5, lambda_penalty=lambda_penalty)
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'n_classes': n_classes,
        'noise': noise,
        'lambda': lambda_penalty,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
        'expansion_rate': fcrs.spawn_count / steps,
    }


def main():
    print('='*60)
    print('Fixed FCRS Experiments')
    print('='*60)
    
    results = []
    
    # ===== Exp1: 原始维度扩张 =====
    print('\n=== Exp1: Dimension Expansion ===')
    for seed in [0, 1, 2]:
        r = run_exp(n_classes=3, noise=0.3, lambda_penalty=0, seed=seed)
        results.append(('Exp1', r))
        print(f"seed={seed}: dim={r['final_dimension']}, error={r['avg_error']:.4f}")
    
    # ===== Exp2: λ sweep =====
    print('\n=== Exp2: λ Sweep ===')
    for lam in [0.001, 0.01, 0.1, 1.0]:
        for seed in [0, 1, 2]:
            r = run_exp(n_classes=3, noise=0.3, lambda_penalty=lam, seed=seed)
            results.append(('Exp2', r))
            print(f"λ={lam}, seed={seed}: dim={r['final_dimension']}, error={r['avg_error']:.4f}")
    
    # ===== Exp3: 不同任务难度 =====
    print('\n=== Exp3: Task Complexity ===')
    for n_cls in [3, 10, 50]:
        for seed in [0, 1, 2]:
            r = run_exp(n_classes=n_cls, noise=0.3, lambda_penalty=0.1, seed=seed)
            results.append(('Exp3', r))
            print(f"n_classes={n_cls}, seed={seed}: dim={r['final_dimension']}, error={r['avg_error']:.4f}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    # Exp1结果
    exp1_results = [r[1] for r in results if r[0] == 'Exp1']
    print(f'\nExp1 (λ=0):')
    print(f"  Dimension: {np.mean([r['final_dimension'] for r in exp1_results]):.1f}")
    print(f"  Error: {np.mean([r['avg_error'] for r in exp1_results]):.4f}")
    
    # Exp2结果
    print(f'\nExp2 (λ sweep):')
    for lam in [0.001, 0.01, 0.1, 1.0]:
        lam_results = [r[1] for r in results if r[0] == 'Exp2' and r[1]['lambda'] == lam]
        print(f"  λ={lam}: dim={np.mean([r['final_dimension'] for r in lam_results]):.1f}, error={np.mean([r['avg_error'] for r in lam_results]):.4f}")
    
    # Exp3结果
    print(f'\nExp3 (task complexity):')
    for n_cls in [3, 10, 50]:
        cls_results = [r[1] for r in results if r[0] == 'Exp3' and r[1]['n_classes'] == n_cls]
        print(f"  n_classes={n_cls}: dim={np.mean([r['final_dimension'] for r in cls_results]):.1f}, error={np.mean([r['avg_error'] for r in cls_results]):.4f}")
    
    # 保存
    with open('F:/fcrs-v5/experiments/fixed_results.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print('\nSaved to fixed_results.json')


if __name__ == "__main__":
    main()
