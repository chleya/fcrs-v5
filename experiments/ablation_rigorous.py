"""
消融实验
按审查意见：移除各组件测试系统表现变化
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem, RepresentationPool
import numpy as np
from scipy import stats


class AblationNoNewDim:
    """消融1: 移除新维度诞生机制"""
    
    def __init__(self, pool_capacity=5, input_dim=10):
        self.pool = RepresentationPool(pool_capacity, input_dim)
        
        # 初始化表征
        for _ in range(3):
            x = np.random.randn(input_dim)
            self.pool.add(x)
        
        self.errors = []
        
    def step(self, x):
        # 激活表征
        active = self.pool.select(x)
        
        if active:
            active.activation_count += 1
            active.reuse += 1
            
            # 预测
            pred = active.vector
            error = np.linalg.norm(x - pred)
            
            # 记录适应度
            active.fitness_history.append(-error)
        else:
            error = float('inf')
        
        self.errors.append(error)
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class AblationNoReuse:
    """消融2: 移除复用机制"""
    
    def __init__(self, pool_capacity=5, input_dim=10):
        self.pool = RepresentationPool(pool_capacity, input_dim)
        
        # 初始化表征
        for _ in range(3):
            x = np.random.randn(input_dim)
            self.pool.add(x)
        
        self.errors = []
        
    def step(self, x):
        # 每次随机选择，不复用
        active = self.pool.select(x)
        
        if active:
            # 重置复用计数
            active.reuse = 0
            active.activation_count += 1
            
            pred = active.vector
            error = np.linalg.norm(x - pred)
            active.fitness_history.append(-error)
        else:
            error = float('inf')
        
        self.errors.append(error)
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class AblationNoCompetition:
    """消融3: 移除竞争机制"""
    
    def __init__(self, pool_capacity=5, input_dim=10):
        self.pool = RepresentationPool(pool_capacity, input_dim)
        
        # 初始化表征
        for _ in range(3):
            x = np.random.randn(input_dim)
            self.pool.add(x)
        
        self.errors = []
        
    def step(self, x):
        # 随机选择，不竞争
        if self.pool.representations:
            idx = np.random.randint(0, len(self.pool.representations))
            active = self.pool.representations[idx]
            
            active.activation_count += 1
            
            pred = active.vector
            error = np.linalg.norm(x - pred)
            active.fitness_history.append(-error)
        else:
            error = float('inf')
        
        self.errors.append(error)
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


def run_ablation(n_runs=10, steps=500):
    """运行消融实验"""
    print('='*60)
    print('Ablation Experiments')
    print('='*60)
    print(f'Runs: {n_runs}, Steps: {steps}')
    print('')
    
    results = {
        'Full System': [],
        'No New Dim': [],
        'No Reuse': [],
        'No Competition': []
    }
    
    from core import EnvironmentLoop
    env = EnvironmentLoop(input_dim=10)
    
    for run in range(n_runs):
        print(f'Run {run+1}/{n_runs}')
        
        np.random.seed(run)
        
        # Full System
        np.random.seed(run)
        full = FCRSystem(pool_capacity=5, vector_dim=10)
        
        np.random.seed(run)
        no_new_dim = AblationNoNewDim(pool_capacity=5, input_dim=10)
        
        np.random.seed(run)
        no_reuse = AblationNoReuse(pool_capacity=5, input_dim=10)
        
        np.random.seed(run)
        no_competition = AblationNoCompetition(pool_capacity=5, input_dim=10)
        
        for _ in range(steps):
            x = env.generate_input()
            
            full.step()
            active = full.pool.select(x)
            if active and len(active.vector) == len(x):
                e = np.linalg.norm(x - active.vector)
            else:
                e = float('inf')
            
            no_new_dim.step(x)
            no_reuse.step(x)
            no_competition.step(x)
        
        results['Full System'].append(full.pool.get_total_dims())
        results['No New Dim'].append(no_new_dim.get_avg_error())
        results['No Reuse'].append(no_reuse.get_avg_error())
        results['No Competition'].append(no_competition.get_avg_error())
    
    # 统计
    print('')
    print('='*60)
    print('Results')
    print('='*60)
    
    for name, values in results.items():
        mean = np.mean(values)
        std = np.std(values)
        print(f'{name}: {mean:.4f} +/- {std:.4f}')
    
    return results


if __name__ == "__main__":
    results = run_ablation(n_runs=10, steps=500)
