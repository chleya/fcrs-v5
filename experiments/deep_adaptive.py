"""
深入自适应截断策略
真正的自适应 - 根据任务难度自动调整
"""

import numpy as np


class AdaptiveTruncationSystem:
    """真正的自适应截断系统"""
    
    def __init__(self):
        # 表征池
        self.representations = []
        for _ in range(10):
            self.representations.append({
                'vector': np.random.randn(50) * 0.1,
                'fitness': 0,
                'age': 0
            })
        
        # 截断策略参数
        self.truncation_threshold = 0.5  # 初始截断阈值
        self.adaptation_rate = 0.01
        
        # 历史
        self.fitness_history = []
        self.dimension_history = []
    
    def calculate_fitness(self, x, rep):
        """计算适应度"""
        v = rep['vector']
        
        # 使用前20维
        v_sub = v[:20]
        x_sub = x[:20]
        
        # 余弦相似度作为适应度
        norm_v = np.linalg.norm(v_sub)
        norm_x = np.linalg.norm(x_sub)
        
        if norm_v > 0.01 and norm_x > 0.01:
            return np.dot(v_sub, x_sub) / (norm_v * norm_x)
        return -1
    
    def select(self, x):
        """选择最佳表征"""
        best = None
        best_fitness = -float('inf')
        
        for i, rep in enumerate(self.representations):
            fitness = self.calculate_fitness(x, rep)
            if fitness > best_fitness:
                best_fitness = fitness
                best = i
        
        return best, best_fitness
    
    def learn(self, x, best):
        """学习"""
        if best is not None:
            v = self.representations[best]['vector'][:20]
            x_sub = x[:20]
            
            # 学习
            self.representations[best]['vector'][:20] += 0.5 * (x_sub - v)
            
            # 更新适应度
            self.representations[best]['fitness'] += 1
            self.representations[best]['age'] += 1
    
    def truncate(self):
        """截断 - 移除最差的表征"""
        # 按适应度排序
        sorted_reps = sorted(range(len(self.representations)),
                          key=lambda i: self.representations[i]['fitness'],
                          reverse=True)
        
        # 保留前50%
        n_keep = max(1, len(self.representations) // 2)
        keep_indices = sorted_reps[:n_keep]
        
        # 更新
        new_reps = [self.representations[i] for i in keep_indices]
        
        # 补充新的随机表征
        while len(new_reps) < 5:
            new_reps.append({
                'vector': np.random.randn(50) * 0.1,
                'fitness': 0,
                'age': 0
            })
        
        self.representations = new_reps
    
    def adapt_truncation(self):
        """自适应调整截断策略"""
        if not self.fitness_history:
            return
        
        recent_fitness = np.mean(self.fitness_history[-20:])
        
        # 如果适应度下降，增加截断频率
        if recent_fitness < 0.3:
            self.truncation_threshold *= 1.1
        # 如果适应度高，减少截断频率
        elif recent_fitness > 0.7:
            self.truncation_threshold *= 0.9
        
        # 限制范围
        self.truncation_threshold = max(0.1, min(0.9, self.truncation_threshold))
    
    def step(self, x):
        """一步"""
        # 选择
        best, fitness = self.select(x)
        
        if best is not None:
            # 学习
            self.learn(x, best)
            
            # 记录
            self.fitness_history.append(fitness)
            
            # 自适应截断
            if len(self.fitness_history) > 50:
                self.adapt_truncation()
                
                # 定期截断
                if len(self.fitness_history) % 20 == 0:
                    self.truncate()
            
            return fitness
        
        return None


class FixedTruncate:
    """固定截断策略"""
    
    def __init__(self, truncate_interval=20):
        self.truncate_interval = truncate_interval
        self.reps = [{'v': np.random.randn(50)*0.1, 'f': 0} for _ in range(10)]
        self.step_count = 0
    
    def step(self, x):
        self.step_count += 1
        
        # 选择
        best = max(range(10), key=lambda i: np.dot(x[:20], self.reps[i]['v'][:20]))
        
        # 学习
        self.reps[best]['v'][:20] += 0.5 * (x[:20] - self.reps[best]['v'][:20])
        self.reps[best]['f'] += 1
        
        # 固定截断
        if self.step_count % self.truncate_interval == 0:
            # 保留最好的5个
            sorted_reps = sorted(range(10), key=lambda i: self.reps[i]['f'], reverse=True)[:5]
            new_reps = [self.reps[i] for i in sorted_reps]
            while len(new_reps) < 10:
                new_reps.append({'v': np.random.randn(50)*0.1, 'f': 0})
            self.reps = new_reps
        
        return 0


def test():
    """测试"""
    print('='*60)
    print('Deep Adaptive Truncation Test')
    print('='*60)
    
    # 环境
    class Env:
        def __init__(self, spread):
            self.centers = [np.random.randn(50) * spread for _ in range(3)]
        def generate(self):
            c = self.centers[np.random.randint(3)]
            return c + np.random.randn(50) * 0.1
    
    results = []
    
    # 1. 自适应截断
    print('\n1. Adaptive Truncation')
    for spread in [0.5, 1.0, 2.0]:
        env = Env(spread)
        system = AdaptiveTruncationSystem()
        
        for _ in range(500):
            x = env.generate()
            system.step(x)
        
        # 测试
        fitness = np.mean(system.fitness_history[-50:])
        print(f'Spread={spread}: fitness={fitness:.4f}')
        results.append(('adaptive', spread, fitness))
    
    # 2. 固定截断
    print('\n2. Fixed Truncation')
    for spread in [0.5, 1.0, 2.0]:
        env = Env(spread)
        system = FixedTruncate(truncate_interval=20)
        
        for _ in range(500):
            x = env.generate()
            system.step(x)
        
        # 测试
        avg_f = np.mean([r['f'] for r in system.reps])
        print(f'Spread={spread}: fitness={avg_f:.4f}')
        results.append(('fixed', spread, avg_f))
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('Method      | Spread | Fitness')
    print('-' * 35)
    for method, spread, fitness in results:
        print(f'{method:11} | {spread:6} | {fitness:.4f}')


if __name__ == "__main__":
    test()
