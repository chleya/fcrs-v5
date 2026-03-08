"""
严格对照实验 - 改进版FCRS
10次运行 + 统计检验
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from scipy import stats


class ImprovedFCRS:
    def __init__(self, pool_capacity=5, input_dim=10, lr=0.01):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.lr = lr
        
        self.representations = []
        for _ in range(3):
            self.representations.append({
                'vector': np.random.randn(input_dim),
                'count': 0,
                'fitness': []
            })
        
        self.errors = []
        
    def select(self, x):
        best_idx = 0
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            score = np.dot(rep['vector'], x) / (np.linalg.norm(rep['vector']) + 1e-8)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def step(self, x):
        idx = self.select(x)
        rep = self.representations[idx]
        
        error = x - rep['vector']
        error_norm = np.linalg.norm(error)
        
        # 在线学习
        rep['vector'] += self.lr * error
        
        rep['count'] += 1
        rep['fitness'].append(-error_norm)
        
        self.errors.append(error_norm)
        
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class OnlineLearning:
    def __init__(self, input_dim=10, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr
        self.vector = np.random.randn(input_dim)
        self.errors = []
    
    def step(self, x):
        error = x - self.vector
        self.vector += self.lr * error
        self.errors.append(np.linalg.norm(error))
        return np.linalg.norm(error)
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class CompetitionOnly:
    def __init__(self, pool_capacity=5, input_dim=10):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        
        self.representations = []
        for _ in range(3):
            self.representations.append({
                'vector': np.random.randn(input_dim),
                'count': 0,
                'fitness': []
            })
        
        self.errors = []
    
    def select(self, x):
        best_idx = 0
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            score = np.dot(rep['vector'], x) / (np.linalg.norm(rep['vector']) + 1e-8)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def step(self, x):
        idx = self.select(x)
        rep = self.representations[idx]
        
        error_norm = np.linalg.norm(x - rep['vector'])
        
        rep['count'] += 1
        rep['fitness'].append(-error_norm)
        
        self.errors.append(error_norm)
        
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class RandomBaseline:
    def __init__(self, pool_capacity=5, input_dim=10):
        self.representations = []
        for _ in range(pool_capacity):
            self.representations.append(np.random.randn(input_dim))
        self.errors = []
    
    def step(self, x):
        idx = np.random.randint(0, len(self.representations))
        error_norm = np.linalg.norm(x - self.representations[idx])
        self.errors.append(error_norm)
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class FixedBaseline:
    def __init__(self, pool_capacity=5, input_dim=10):
        self.representations = []
        for _ in range(pool_capacity):
            self.representations.append(np.random.randn(input_dim))
        self.errors = []
    
    def select(self, x):
        best_idx = 0
        best_error = float('inf')
        
        for i, rep in enumerate(self.representations):
            error = np.linalg.norm(x - rep)
            if error < best_error:
                best_error = error
                best_idx = i
        
        return best_idx
    
    def step(self, x):
        idx = self.select(x)
        error_norm = np.linalg.norm(x - self.representations[idx])
        self.errors.append(error_norm)
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


def run_rigorous_test(n_runs=10, steps=500):
    """严格测试"""
    print('='*60)
    print('Rigorous Comparison - Improved FCRS')
    print('='*60)
    print(f'Runs: {n_runs}, Steps: {steps}')
    print('')
    
    from core import EnvironmentLoop
    env = EnvironmentLoop(input_dim=10)
    
    results = {
        'Random': [],
        'Fixed': [],
        'Online': [],
        'Competition': [],
        'Improved FCRS': []
    }
    
    for run in range(n_runs):
        print(f'Run {run+1}/{n_runs}')
        
        np.random.seed(run)
        
        random_sys = RandomBaseline(pool_capacity=5, input_dim=10)
        
        np.random.seed(run)
        fixed_sys = FixedBaseline(pool_capacity=5, input_dim=10)
        
        np.random.seed(run)
        online_sys = OnlineLearning(input_dim=10, lr=0.01)
        
        np.random.seed(run)
        competition_sys = CompetitionOnly(pool_capacity=5, input_dim=10)
        
        np.random.seed(run)
        improved_sys = ImprovedFCRS(pool_capacity=5, input_dim=10, lr=0.01)
        
        for _ in range(steps):
            x = env.generate_input()
            
            random_sys.step(x)
            fixed_sys.step(x)
            online_sys.step(x)
            competition_sys.step(x)
            improved_sys.step(x)
        
        results['Random'].append(random_sys.get_avg_error())
        results['Fixed'].append(fixed_sys.get_avg_error())
        results['Online'].append(online_sys.get_avg_error())
        results['Competition'].append(competition_sys.get_avg_error())
        results['Improved FCRS'].append(improved_sys.get_avg_error())
    
    # 统计
    print('')
    print('='*60)
    print('Statistical Results')
    print('='*60)
    
    print('\nMean +/- Std:')
    for name, errors in results.items():
        mean = np.mean(errors)
        std = np.std(errors)
        print(f'{name}: {mean:.4f} +/- {std:.4f}')
    
    # t检验
    print('\nt-test (Improved FCRS vs others):')
    fcrs = np.array(results['Improved FCRS'])
    
    for name in ['Random', 'Fixed', 'Online', 'Competition']:
        other = np.array(results[name])
        t, p = stats.ttest_ind(fcrs, other)
        
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        
        better = 'BETTER' if np.mean(fcrs) < np.mean(other) else 'WORSE'
        
        print(f'{name}: t={t:.2f}, p={p:.4f} {sig} ({better})')
    
    return results


if __name__ == "__main__":
    results = run_rigorous_test(n_runs=10, steps=500)
