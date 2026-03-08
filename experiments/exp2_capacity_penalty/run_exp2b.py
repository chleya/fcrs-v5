"""
Experiment 2b: Structural Penalty Test (Simplified)
H1: capacity cost in emergence rules
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []
        self.activation_count = 0


class SimpleEnv:
    def __init__(self, input_dim=10, n_classes=3):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class SimpleFCRS:
    """简化FCRS - 固定维数，跟踪活跃表征数"""
    
    def __init__(self, pool_capacity=5, input_dim=10, n_classes=3, lr=0.01, lambda_penalty=0.0):
        self.input_dim = input_dim
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.representations = []
        self.env = SimpleEnv(input_dim, n_classes)
        
        self.errors = []
        self.spawn_count = 0
        self.prune_count = 0
        self.dimension = input_dim
        
        # 初始化
        for _ in range(3):
            x = self.env.generate_input()
            self.representations.append(SimpleRep(x))
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            # 只比较前dimension维
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
            # 只更新前dimension维
            error_vec = x - active.vector[:self.dimension]
            error = np.linalg.norm(error_vec)
            
            active.vector[:self.dimension] += self.lr * error_vec
            
            active.activation_count += 1
            active.fitness_history.append(-error)
            
            self.errors.append(error)
            
            self._structural_decision(error)
        else:
            self.errors.append(float('inf'))
    
    def _structural_decision(self, error):
        """结构决策"""
        
        # 计算边际收益 (简化)
        import random
        gain = random.random() * 0.3
        
        # Spawn: if gain > λ
        if len(self.representations) < 5 and gain > self.lambda_penalty:
            self.dimension += 1
            self.spawn_count += 1
        
        # Prune: if fitness < λ
        if len(self.representations) > 1:
            fitnesses = [np.mean(r.fitness_history[-10:]) if r.fitness_history else 0 
                        for r in self.representations]
            min_fitness = min(fitnesses)
            
            if min_fitness < self.lambda_penalty:
                # 移除最差
                min_idx = fitnesses.index(min_fitness)
                self.representations.pop(min_idx)
                self.prune_count += 1


def run_exp2b(lambda_penalty, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = SimpleFCRS(
        pool_capacity=5,
        input_dim=10,
        n_classes=3,
        lr=0.01,
        lambda_penalty=lambda_penalty
    )
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'lambda': lambda_penalty,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
        'spawn': fcrs.spawn_count,
        'prune': fcrs.prune_count,
    }


def main():
    print('='*60)
    print('Experiment 2b: Structural Penalty')
    print('='*60)
    
    lambdas = [0, 0.01, 0.1, 1.0]
    seeds = [0, 1, 2]
    
    results = []
    
    for lam in lambdas:
        print(f'\n--- λ = {lam} ---')
        
        for seed in seeds:
            result = run_exp2b(lam, seed)
            results.append(result)
            print(f"seed={seed}: error={result['avg_error']:.4f}, dim={result['final_dimension']}, spawn={result['spawn']}, prune={result['prune']}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    summary = {}
    for lam in lambdas:
        lam_results = [r for r in results if r['lambda'] == lam]
        
        summary[lam] = {
            'error': np.mean([r['avg_error'] for r in lam_results]),
            'dimension': np.mean([r['final_dimension'] for r in lam_results]),
            'spawn': np.mean([r['spawn'] for r in lam_results]),
            'prune': np.mean([r['prune'] for r in lam_results]),
        }
    
    print(f'{"λ":<10} {"error":<12} {"dimension":<12} {"spawn":<10} {"prune":<10}')
    print('-'*60)
    
    for lam in lambdas:
        s = summary[lam]
        print(f'{lam:<10} {s["error"]:<12.4f} {s["dimension"]:<12.1f} {s["spawn"]:<10.1f} {s["prune"]:<10.1f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/exp2_capacity_penalty/exp2b_results.json', 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    
    print('\nSaved to exp2b_results.json')


if __name__ == "__main__":
    main()
