"""
改进版自适应截断
"""

import numpy as np


class AdaptiveFCRS:
    """自适应FCRS - 根据环境难度调整"""
    
    def __init__(self):
        self.reps = [{'v': np.random.randn(20)*0.1, 'errors': []} for _ in range(5)]
        
        # 关键参数
        self.dimension = 20  # 动态维度
        self.learning_rate = 0.5
    
    def estimate_difficulty(self, x):
        """估计难度: 输入与中心的距离"""
        # 简单方法: 随机选取一个表征的距离
        dists = [np.linalg.norm(x - r['v']) for r in self.reps]
        return np.mean(dists)
    
    def adapt_dimension(self, difficulty):
        """根据难度调整维度"""
        # 难 → 增加维度
        # 易 → 减少维度
        
        if difficulty > 1.0:
            self.dimension = min(20, self.dimension + 1)
        elif difficulty < 0.5:
            self.dimension = max(5, self.dimension - 1)
    
    def step(self, x):
        """一步"""
        # 估计难度
        difficulty = self.estimate_difficulty(x)
        
        # 调整维度
        self.adapt_dimension(difficulty)
        
        # 选择最佳表征
        best = min(range(5), 
                   key=lambda i: np.linalg.norm(x[:self.dimension] - self.reps[i]['v'][:self.dimension]))
        
        # 学习
        self.reps[best]['v'][:self.dimension] += self.learning_rate * (
            x[:self.dimension] - self.reps[best]['v'][:self.dimension])
        
        # 误差
        error = np.linalg.norm(x[:self.dimension] - self.reps[best]['v'][:self.dimension])
        
        # 记录
        self.reps[best]['errors'].append(error)
        
        return error, self.dimension


class FixedFCRS:
    """固定维度FCRS"""
    def __init__(self, dim=10):
        self.dim = dim
        self.reps = [{'v': np.random.randn(20)*0.1} for _ in range(5)]
    
    def step(self, x):
        best = min(range(5), key=lambda i: np.linalg.norm(x[:self.dim] - self.reps[i]['v'][:self.dim]))
        self.reps[best]['v'][:self.dim] += 0.5 * (x[:self.dim] - self.reps[best]['v'][:self.dim])
        return np.linalg.norm(x[:self.dim] - self.reps[best]['v'][:self.dim])


def test():
    print('='*60)
    print('Adaptive Dimension Test')
    print('='*60)
    
    # 不同难度环境
    class Env:
        def __init__(self, spread):
            self.centers = [np.random.randn(20) * spread for _ in range(3)]
        def generate(self):
            c = self.centers[np.random.randint(3)]
            return c + np.random.randn(20) * 0.1
    
    results = []
    
    # 自适应
    print('\nAdaptive:')
    for spread in [0.5, 1.0, 2.0]:
        env = Env(spread)
        fcrs = AdaptiveFCRS()
        
        dims = []
        for _ in range(500):
            err, dim = fcrs.step(env.generate())
            dims.append(dim)
        
        # 测试
        test_errs = [fcrs.step(env.generate())[0] for _ in range(100)]
        
        print(f'Spread={spread}: dim={np.mean(dims[-100:]):.0f}, error={np.mean(test_errs):.4f}')
        results.append(('adaptive', spread, np.mean(test_errs)))
    
    # 固定
    print('\nFixed:')
    for spread in [0.5, 1.0, 2.0]:
        env = Env(spread)
        fcrs = FixedFCRS(dim=10)
        
        for _ in range(500):
            fcrs.step(env.generate())
        
        test_errs = [fcrs.step(env.generate()) for _ in range(100)]
        
        print(f'Spread={spread}: error={np.mean(test_errs):.4f}')
        results.append(('fixed', spread, np.mean(test_errs)))
    
    print('\nSummary:')
    print('Method      | Spread | Error')
    print('-' * 30)
    for method, spread, err in results:
        print(f'{method:11} | {spread:6} | {err:.4f}')


if __name__ == "__main__":
    test()
