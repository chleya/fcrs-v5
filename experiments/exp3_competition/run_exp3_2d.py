"""
Experiment 3: Competition × Capacity Cost (2D Grid)
H3: 两个机制的交互作用
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class SimpleEnv:
    def __init__(self, input_dim=10, n_classes=3):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class FullFCRS:
    """完整FCRS - 带容量惩罚和竞争"""
    
    def __init__(self, input_dim=10, n_classes=3, lr=0.01, 
                 lambda_penalty=0.1, elimination_rate=0.3):
        self.input_dim = input_dim
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        self.elimination_rate = elimination_rate
        
        self.representations = []
        self.env = SimpleEnv(input_dim, n_classes)
        
        self.errors = []
        self.dimension = input_dim
        self.spawn_count = 0
        self.elim_count = 0
        
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
        
        # Competition: 淘汰最低适应度
        if len(self.representations) > 1:
            fitnesses = []
            for r in self.representations:
                if r.fitness_history:
                    f = np.mean(r.fitness_history[-10:])
                else:
                    f = 0
                fitnesses.append(f)
            
            indices = np.argsort(fitnesses)
            n_elim = max(1, int(len(self.representations) * self.elimination_rate))
            
            keep_indices = indices[n_elim:]
            
            new_reps = []
            for i, rep in enumerate(self.representations):
                if i in keep_indices:
                    new_reps.append(rep)
                else:
                    self.elim_count += 1
            
            self.representations = new_reps
            
            # 补充
            while len(self.representations) < 3:
                x = self.env.generate_input()
                self.representations.append(SimpleRep(x))


def run_exp3(lambda_penalty, elim_rate, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = FullFCRS(
        input_dim=10,
        n_classes=3,
        lr=0.01,
        lambda_penalty=lambda_penalty,
        elimination_rate=elim_rate
    )
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'lambda': lambda_penalty,
        'elimination_rate': elim_rate,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
        'spawn': fcrs.spawn_count,
        'eliminate': fcrs.elim_count,
        'expansion_rate': fcrs.spawn_count / steps,
        'pruning_rate': fcrs.elim_count / steps,
    }


def main():
    print('='*60)
    print('Experiment 3: Competition × Capacity Cost')
    print('='*60)
    
    lambdas = [0.01, 0.1, 1.0]
    elim_rates = [0.1, 0.3, 0.5, 0.9]
    seeds = [0, 1, 2]
    
    results = []
    
    total = len(lambdas) * len(elim_rates) * len(seeds)
    run_id = 0
    
    for lam in lambdas:
        for rate in elim_rates:
            print(f'\n--- λ={lam}, elim={rate} ---')
            
            for seed in seeds:
                run_id += 1
                result = run_exp3(lam, rate, seed)
                results.append(result)
                print(f"[{run_id}/{total}] dim={result['final_dimension']}, exp={result['expansion_rate']:.2f}")
    
    # 汇总 - 2D grid
    print('\n' + '='*60)
    print('Results Grid: λ × elimination_rate')
    print('='*60)
    
    # 按λ分组打印
    for lam in lambdas:
        print(f'\n=== λ = {lam} ===')
        print(f'{"elim_rate":<12} {"dim":<8} {"exp_rate":<12} {"prune_rate":<12}')
        
        for rate in elim_rates:
            rate_results = [r for r in results if r['lambda'] == lam and r['elimination_rate'] == rate]
            
            avg_dim = np.mean([r['final_dimension'] for r in rate_results])
            avg_exp = np.mean([r['expansion_rate'] for r in rate_results])
            avg_prune = np.mean([r['pruning_rate'] for r in rate_results])
            
            print(f'{rate:<12} {avg_dim:<8.1f} {avg_exp:<12.3f} {avg_prune:<12.3f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/exp3_competition/results_2d.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    
    print('\nSaved to results_2d.json')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    # 检查是否是two-regime
    print('\n[Two-Regime Analysis]')
    for lam in lambdas:
        dims = []
        for rate in elim_rates:
            rate_results = [r for r in results if r['lambda'] == lam and r['elimination_rate'] == rate]
            dims.append(np.mean([r['final_dimension'] for r in rate_results]))
        
        variation = np.max(dims) - np.min(dims)
        print(f'λ={lam}: dimension range = {variation:.1f}')


if __name__ == "__main__":
    main()
