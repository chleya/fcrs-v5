"""
Experiment 4: Task Complexity × Capacity
Question: Does task complexity drive representation capacity?
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class TaskEnv:
    """任务环境 - 复杂度可调"""
    def __init__(self, input_dim=10, n_classes=3):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class TaskFCRS:
    """FCRS with fixed λ"""
    
    def __init__(self, input_dim=10, n_classes=3, lr=0.01, lambda_penalty=0.1):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.representations = []
        self.env = TaskEnv(input_dim, n_classes)
        
        self.errors = []
        self.dimension = input_dim
        self.spawn_count = 0
        
        # 初始化
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
            self._structural_decision()
        else:
            self.errors.append(float('inf'))
    
    def _structural_decision(self):
        # Spawn: gain > λ
        gain = random.random() * 0.3
        if len(self.representations) < 5 and gain > self.lambda_penalty:
            self.dimension += 1
            self.spawn_count += 1


def run_exp4(n_classes, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = TaskFCRS(
        input_dim=10,
        n_classes=n_classes,
        lr=0.01,
        lambda_penalty=0.1  # fixed at regulated regime
    )
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'n_classes': n_classes,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
        'expansion_rate': fcrs.spawn_count / steps,
    }


def main():
    print('='*60)
    print('Experiment 4: Task Complexity × Capacity')
    print('='*60)
    
    n_classes_list = [3, 5, 10, 20, 50]
    seeds = [0, 1, 2]
    
    results = []
    
    for n_classes in n_classes_list:
        print(f'\n--- n_classes = {n_classes} ---')
        
        for seed in seeds:
            result = run_exp4(n_classes, seed)
            results.append(result)
            print(f"seed={seed}: dim={result['final_dimension']}, loss={result['avg_error']:.4f}, exp_rate={result['expansion_rate']:.2f}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    summary = {}
    for n_classes in n_classes_list:
        cls_results = [r for r in results if r['n_classes'] == n_classes]
        
        summary[n_classes] = {
            'dimension': np.mean([r['final_dimension'] for r in cls_results]),
            'error': np.mean([r['avg_error'] for r in cls_results]),
            'expansion_rate': np.mean([r['expansion_rate'] for r in cls_results]),
        }
    
    print(f'{"n_classes":<12} {"dimension":<12} {"error":<12} {"exp_rate":<12}')
    print('-'*60)
    
    for n_classes in n_classes_list:
        s = summary[n_classes]
        print(f'{n_classes:<12} {s["dimension"]:<12.1f} {s["error"]:<12.4f} {s["expansion_rate"]:<12.3f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/exp4_complexity/results.json', 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    
    print('\nSaved to results.json')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    dims = [summary[n]['dimension'] for n in n_classes_list]
    errors = [summary[n]['error'] for n in n_classes_list]
    
    # 检查单调性
    dim_diff = dims[-1] - dims[0]
    error_diff = errors[-1] - errors[0]
    
    print(f'Dimension range: {dims[0]:.1f} → {dims[-1]:.1f} (Δ={dim_diff:.1f})')
    print(f'Error range: {errors[0]:.4f} → {errors[-1]:.4f} (Δ={error_diff:.4f})')
    
    if dim_diff > 50:
        print('[OK] Dimension increases with complexity')
    else:
        print('[WARN] Dimension does not increase')
    
    if error_diff < 0:
        print('[OK] Error decreases with complexity')
    else:
        print('[INFO] Error increases with complexity')


if __name__ == "__main__":
    main()
