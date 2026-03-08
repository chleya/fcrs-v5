# FCRS-v5 核心代码 (最终版)

import numpy as np


class FCRS:
    """FCRS核心类"""
    
    def __init__(self, pool_capacity=5, input_dim=10, lr=0.01):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.lr = lr
        
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
        idx = self.select(x)
        rep = self.representations[idx]
        
        # 预测误差
        error = x - rep['vector']
        error_norm = np.linalg.norm(error)
        
        # 核心: 在线学习
        rep['vector'] += self.lr * error
        
        # 记录
        rep['count'] += 1
        rep['fitness'].append(-error_norm)
        
        self.errors.append(error_norm)
        
        return error_norm
    
    def get_avg_error(self):
        return np.mean(self.errors[-100:]) if self.errors else 0
    
    def get_stats(self):
        return {
            'n_reps': len(self.representations),
            'avg_error': self.get_avg_error(),
            'total_count': sum(r['count'] for r in self.representations)
        }


def demo():
    """演示"""
    from core import EnvironmentLoop
    env = EnvironmentLoop(input_dim=10)
    
    fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)
    
    for _ in range(500):
        x = env.generate_input()
        fcrs.step(x)
    
    print('FCRS Demo:')
    print('  Avg Error:', round(fcrs.get_avg_error(), 4))
    print('  Stats:', fcrs.get_stats())


if __name__ == "__main__":
    demo()
