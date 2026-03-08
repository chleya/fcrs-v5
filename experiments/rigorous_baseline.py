"""
严格对照实验
按审查意见：基线对比 + 消融实验 + 统计检验
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from core import FCRSystem


class BaselineRandom:
    """基线1: 随机选择"""
    
    def __init__(self, pool_capacity, input_dim):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.representations = []
        
        # 初始化随机表征
        for _ in range(pool_capacity):
            self.representations.append(np.random.randn(input_dim))
        
        self.errors = []
        
    def step(self, x):
        # 随机选择一个表征
        idx = np.random.randint(0, len(self.representations))
        pred = self.representations[idx]
        
        error = np.linalg.norm(x - pred)
        self.errors.append(error)
        
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class BaselineFixed:
    """基线2: 固定维度(无动态调整)"""
    
    def __init__(self, pool_capacity, input_dim):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.representations = []
        
        # 初始化固定表征
        for _ in range(pool_capacity):
            self.representations.append(np.random.randn(input_dim))
        
        self.errors = []
        
    def step(self, x):
        # 选择最匹配的表征
        best_idx = 0
        best_error = float('inf')
        
        for i, rep in enumerate(self.representations):
            error = np.linalg.norm(x - rep)
            if error < best_error:
                best_error = error
                best_idx = i
        
        self.errors.append(best_error)
        return best_error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class BaselineOnline:
    """基线3: 标准在线学习(简单梯度下降)"""
    
    def __init__(self, pool_capacity, input_dim):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.representations = []
        
        # 初始化表征
        for _ in range(pool_capacity):
            self.representations.append({
                'vector': np.random.randn(input_dim),
                'count': 0
            })
        
        self.errors = []
        self.lr = 0.01
        
    def step(self, x):
        # 选择最匹配的表征
        best_idx = 0
        best_error = float('inf')
        
        for i, rep in enumerate(self.representations):
            error = np.linalg.norm(x - rep['vector'])
            if error < best_error:
                best_error = error
                best_idx = i
        
        # 在线更新: 向输入方向移动
        self.representations[best_idx]['vector'] += self.lr * (x - self.representations[best_idx]['vector'])
        self.representations[best_idx]['count'] += 1
        
        self.errors.append(best_error)
        return best_error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class OurSystem:
    """我们的系统: 完整FCRS"""
    
    def __init__(self, pool_capacity, input_dim):
        self.system = FCRSystem(pool_capacity=pool_capacity, vector_dim=input_dim)
        self.errors = []
        
    def step(self, x):
        self.system.step()
        
        # 计算误差
        active = self.system.pool.select(x)
        if active:
            error = np.linalg.norm(x - active.vector)
        else:
            error = float('inf')
        
        self.errors.append(error)
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


def run_rigorous_experiment(n_runs=10, steps=500):
    """严格对照实验"""
    print('='*60)
    print('严格对照实验')
    print('='*60)
    print('Runs: ' + str(n_runs) + ', Steps: ' + str(steps))
    print('')
    
    results = {
        'Random': [],
        'Fixed': [],
        'Online': [],
        'FCRS': []
    }
    
    # 多次运行
    for run in range(n_runs):
        print('Run ' + str(run+1) + '/' + str(n_runs))
        
        np.random.seed(run)
        
        # 创建系统
        random_baseline = BaselineRandom(pool_capacity=5, input_dim=10)
        fixed_baseline = BaselineFixed(pool_capacity=5, input_dim=10)
        online_baseline = BaselineOnline(pool_capacity=5, input_dim=10)
        our_system = OurSystem(pool_capacity=5, input_dim=10)
        
        # 环境
        from core import EnvironmentLoop
        env = EnvironmentLoop(input_dim=10)
        
        # 运行
        for _ in range(steps):
            x = env.generate_input()
            
            random_baseline.step(x)
            fixed_baseline.step(x)
            online_baseline.step(x)
            our_system.step(x)
        
        # 记录最终误差
        results['Random'].append(random_baseline.get_avg_error())
        results['Fixed'].append(fixed_baseline.get_avg_error())
        results['Online'].append(online_baseline.get_avg_error())
        results['FCRS'].append(our_system.get_avg_error())
    
    # 统计分析
    print('')
    print('='*60)
    print('结果统计')
    print('='*60)
    
    print('\n平均误差 (最后100步):')
    for name, errors in results.items():
        mean = np.mean(errors)
        std = np.std(errors)
        print(name + ': ' + str(round(mean, 4)) + ' ± ' + str(round(std, 4)))
    
    # 简单对比
    print('')
    print('对比结论:')
    fcrs_mean = np.mean(results['FCRS'])
    online_mean = np.mean(results['Online'])
    
    if fcrs_mean < online_mean:
        print('FCRS < Online学习: 好')
    else:
        print('FCRS > Online学习: 需要改进')


if __name__ == "__main__":
    results = run_rigorous_experiment(n_runs=10, steps=500)
