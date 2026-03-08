"""
FCRS-v5 改进版 - 加入在线学习
核心改进: 表征竞争 + 在线学习
"""

import numpy as np


class ImprovedFCRS:
    """改进版FCRS"""
    
    def __init__(self, pool_capacity=5, input_dim=10, lr=0.01):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.lr = lr  # 学习率
        
        # 表征池
        self.representations = []
        
        # 初始化
        for _ in range(3):
            self.representations.append({
                'vector': np.random.randn(input_dim),
                'count': 0,
                'fitness': []
            })
        
        self.errors = []
        
    def select(self, x):
        """选择最匹配表征"""
        best_idx = 0
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            score = np.dot(rep['vector'], x) / (np.linalg.norm(rep['vector']) + 1e-8)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def step(self, x):
        """一步"""
        # 1. 选择
        idx = self.select(x)
        rep = self.representations[idx]
        
        # 2. 计算误差
        pred = rep['vector']
        error = x - pred
        error_norm = np.linalg.norm(error)
        
        # 3. 在线学习(关键改进!)
        rep['vector'] += self.lr * error  # 更新表征!
        
        # 4. 记录
        rep['count'] += 1
        rep['fitness'].append(-error_norm)
        
        self.errors.append(error_norm)
        
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class CompetitionOnly:
    """对照组: 只有竞争,无学习"""
    
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
        
        # 只选择,不更新!
        error = np.linalg.norm(x - rep['vector'])
        
        rep['count'] += 1
        rep['fitness'].append(-error)
        
        self.errors.append(error)
        
        return error
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


class OnlineLearning:
    """基线: 纯在线学习"""
    
    def __init__(self, input_dim=10, lr=0.01):
        self.input_dim = input_dim
        self.lr = lr
        
        # 单一表征
        self.vector = np.random.randn(input_dim)
        
        self.errors = []
    
    def step(self, x):
        error = x - self.vector
        self.vector += self.lr * error
        
        self.errors.append(np.linalg.norm(error))
        
        return np.linalg.norm(error)
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0


def test_improved():
    """测试改进版"""
    print('='*60)
    print('Improved FCRS Test')
    print('='*60)
    
    from core import EnvironmentLoop
    env = EnvironmentLoop(input_dim=10)
    
    # 测试各系统
    systems = {
        'Online Learning': OnlineLearning(input_dim=10, lr=0.01),
        'Competition Only': CompetitionOnly(pool_capacity=5, input_dim=10),
        'Improved FCRS': ImprovedFCRS(pool_capacity=5, input_dim=10, lr=0.01),
    }
    
    results = {}
    
    for name, system in systems.items():
        print(f'\n{name}:')
        
        for i in range(500):
            x = env.generate_input()
            system.step(x)
        
        avg_error = system.get_avg_error()
        results[name] = avg_error
        print(f'  Avg Error: {avg_error:.4f}')
    
    print('\n' + '='*60)
    print('Results:')
    print('='*60)
    
    for name, error in sorted(results.items(), key=lambda x: x[1]):
        print(f'{name}: {error:.4f}')
    
    # 对比
    improved = results['Improved FCRS']
    online = results['Online Learning']
    competition = results['Competition Only']
    
    print('')
    if improved < competition:
        print('✓ Improved FCRS beats Competition Only!')
    else:
        print('✗ Improved FCRS still worse than Competition Only')
    
    if improved < online * 1.5:  # 允许50%损失
        print('✓ Improved FCRS competitive with Online Learning')
    else:
        print('✗ Need more work')


if __name__ == "__main__":
    test_improved()
