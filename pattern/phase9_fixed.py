"""
Phase 9 Final: Fixed Pattern Evolution
固定模式，看演化能否找到
"""

import numpy as np


class FixedPattern:
    """固定模式"""
    def __init__(self):
        # 固定的 x-y 对
        self.training_data = [
            (1.0, 0.5),
            (2.0, 1.0),
            (3.0, 1.5),
            (4.0, 2.0),
            (5.0, 2.5),
        ]
    
    def evaluate(self, weight, bias):
        """评估一个解"""
        total_error = 0
        for x, y in self.training_data:
            pred = weight * x + bias
            total_error += abs(pred - y)
        return total_error / len(self.training_data)


class Evolution:
    def __init__(self, pop_size=100):
        self.pattern = FixedPattern()
        
        self.population = []
        for _ in range(pop_size):
            self.population.append({
                'weight': np.random.randn() * 5,
                'bias': np.random.randn() * 5,
                'fitness': 0,
            })
    
    def step(self):
        # 评估
        for ind in self.population:
            error = self.pattern.evaluate(ind['weight'], ind['bias'])
            ind['fitness'] = 1.0 / (error + 0.001)
        
        # 选择
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        survivors = sorted_pop[:20]
        
        # 复制 + 变异
        new_pop = []
        for parent in survivors:
            for _ in range(5):
                child = {
                    'weight': parent['weight'] + np.random.randn() * 0.1,
                    'bias': parent['bias'] + np.random.randn() * 0.1,
                    'fitness': 0,
                }
                new_pop.append(child)
        
        self.population = new_pop[:100]
        
        # 最佳
        best = survivors[0]
        error = self.pattern.evaluate(best['weight'], best['bias'])
        
        return error, best['weight'], best['bias']


def main():
    print('='*60)
    print('Phase 9: Fixed Pattern Evolution')
    print('='*60)
    print('\nPattern: y = 0.5 * x')
    print('(1,0.5), (2,1.0), (3,1.5), (4,2.0), (5,2.5)\n')
    
    evo = Evolution(pop_size=100)
    
    for gen in range(200):
        error, w, b = evo.step()
        
        if gen % 20 == 0:
            print(f'Gen {gen}: error={error:.4f}, w={w:.4f}, b={b:.4f}')
    
    print('\n=== Result ===')
    print(f'Expected: w=0.5, b=0')
    print(f'Found: w={w:.4f}, b={b:.4f}')
    
    if error < 0.01:
        print('\n[SUCCESS] Evolution found the pattern!')
        print('Structure emerged from random soup!')
    else:
        print('\n[Searching...]')


if __name__ == "__main__":
    main()
