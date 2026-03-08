"""
Experiment 5: λ Phase Transition
Question: Does capacity show phase transition as λ changes?
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []


class TaskEnv:
    def __init__(self, input_dim=10, n_classes=50):  # hardest task
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class PhaseFCRS:
    def __init__(self, input_dim=10, n_classes=50, lr=0.01, lambda_penalty=0.1):
        self.input_dim = input_dim
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.representations = []
        self.env = TaskEnv(input_dim, n_classes)
        
        self.errors = []
        self.dimension = input_dim
        self.spawn_count = 0
        
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
        gain = random.random() * 0.3
        if len(self.representations) < 5 and gain > self.lambda_penalty:
            self.dimension += 1
            self.spawn_count += 1


def run_exp5(lambda_penalty, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = PhaseFCRS(
        input_dim=10,
        n_classes=50,
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
        'expansion_rate': fcrs.spawn_count / steps,
    }


def main():
    print('='*60)
    print('Experiment 5: λ Phase Transition')
    print('='*60)
    
    lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    seeds = [0, 1, 2]
    
    results = []
    
    for lam in lambdas:
        print(f'\n--- λ = {lam} ---')
        
        for seed in seeds:
            result = run_exp5(lam, seed)
            results.append(result)
            print(f"seed={seed}: dim={result['final_dimension']}, loss={result['avg_error']:.4f}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary: λ Phase Transition')
    print('='*60)
    
    summary = {}
    for lam in lambdas:
        lam_results = [r for r in results if r['lambda'] == lam]
        
        summary[lam] = {
            'dimension': np.mean([r['final_dimension'] for r in lam_results]),
            'error': np.mean([r['avg_error'] for r in lam_results]),
            'expansion_rate': np.mean([r['expansion_rate'] for r in lam_results]),
        }
    
    print(f'{"λ":<12} {"dimension":<12} {"error":<12} {"exp_rate":<12}')
    print('-'*60)
    
    for lam in lambdas:
        s = summary[lam]
        print(f'{lam:<12} {s["dimension"]:<12.1f} {s["error"]:<12.4f} {s["expansion_rate"]:<12.3f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/exp5_lambda_sweep/results.json', 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    
    print('\nSaved to results.json')
    
    # 分析phase transition
    print('\n' + '='*60)
    print('Phase Transition Analysis')
    print('='*60)
    
    dims = [summary[lam]['dimension'] for lam in lambdas]
    errors = [summary[lam]['error'] for lam in lambdas]
    
    # 检查是否存在jump
    for i in range(1, len(dims)):
        diff = dims[i] - dims[i-1]
        if abs(diff) > 50:
            print(f'[PHASE TRANSITION] λ: {lambdas[i-1]} → {lambdas[i]}: Δdim = {diff:.1f}')
    
    print(f'\nDimension range: {min(dims):.1f} → {max(dims):.1f}')
    print(f'Error range: {min(errors):.4f} → {max(errors):.4f}')


if __name__ == "__main__":
    main()
