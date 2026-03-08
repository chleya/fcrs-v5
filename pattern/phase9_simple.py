"""
Phase 9 Final: The Emergent Agent
最简单的版本: 随机权重 → 演化 → 找到正确权重
"""

import numpy as np


class SimpleTask:
    """简单任务: y = sin(x)"""
    def __init__(self):
        self.x = 1.0
    
    def generate_target(self):
        """目标: x * 0.5"""
        self.x += 0.1
        return self.x * 0.5


class Evolution:
    """演化"""
    
    def __init__(self, pop_size=50):
        # 随机初始
        self.population = []
        for _ in range(pop_size):
            self.population.append({
                'weight': np.random.randn() * 2,
                'bias': np.random.randn() * 2,
                'fitness': 0,
            })
        
        self.task = SimpleTask()
    
    def step(self):
        """一步"""
        # 获取目标
        x = self.task.x
        target = self.task.generate_target()
        
        # 评估
        for ind in self.population:
            pred = ind['weight'] * x + ind['bias']
            error = abs(pred - target)
            ind['fitness'] = 1.0 / (error + 0.01)
        
        # 选择
        sorted_pop = sorted(self.population, key=lambda p: p['fitness'], reverse=True)
        survivors = sorted_pop[:10]
        
        # 复制
        new_pop = []
        for parent in survivors:
            for _ in range(5):
                child = {
                    'weight': parent['weight'] + np.random.randn() * 0.1,
                    'bias': parent['bias'] + np.random.randn() * 0.1,
                    'fitness': 0,
                }
                new_pop.append(child)
        
        self.population = new_pop[:50]
        
        # 最佳
        best = survivors[0]
        pred = best['weight'] * x + best['bias']
        return abs(pred - target), best['weight'], best['bias']


def main():
    print('='*60)
    print('Phase 9: Can Evolution Find the Solution?')
    print('='*60)
    print('\nTask: y = 0.5 * x')
    print('Starting with random weights...\n')
    
    evo = Evolution(pop_size=50)
    
    for gen in range(100):
        error, w, b = evo.step()
        
        if gen % 20 == 0:
            print(f'Gen {gen}: error={error:.4f}, weight={w:.4f}, bias={b:.4f}')
    
    print('\n=== Result ===')
    print(f'Expected: weight=0.5, bias=0')
    print(f'Found: weight={w:.4f}, bias={b:.4f}')
    
    if abs(w - 0.5) < 0.1:
        print('\n[OK] Evolution found the solution!')
        print('Structure emerged from random soup!')
    else:
        print('\n[WARN] Still searching...')


if __name__ == "__main__":
    main()
