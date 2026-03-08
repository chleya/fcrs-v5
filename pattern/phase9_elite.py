"""
Phase 9 Final: Evolution with Elitism
精英保留: 确保最佳解不丢失
"""

import numpy as np


class FixedPattern:
    def __init__(self):
        self.training_data = [
            (1.0, 0.5),
            (2.0, 1.0),
            (3.0, 1.5),
            (4.0, 2.0),
            (5.0, 2.5),
        ]
    
    def evaluate(self, weight, bias):
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
        
        self.best_ever = {'weight': 0, 'bias': 0, 'error': float('inf')}
    
    def step(self):
        # 评估
        for ind in self.population:
            error = self.pattern.evaluate(ind['weight'], ind['bias'])
            ind['fitness'] = 1.0 / (error + 0.001)
            
            if error < self.best_ever['error']:
                self.best_ever = {
                    'weight': ind['weight'],
                    'bias': ind['bias'],
                    'error': error
                }
        
        # 选择
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        survivors = sorted_pop[:20]
        
        # 精英保留
        new_pop = [self.best_ever.copy()]
        new_pop[0]['fitness'] = 0
        
        # 复制 + 变异
        for parent in survivors:
            for _ in range(5):
                child = {
                    'weight': parent['weight'] + np.random.randn() * 0.1,
                    'bias': parent['bias'] + np.random.randn() * 0.1,
                    'fitness': 0,
                }
                new_pop.append(child)
        
        self.population = new_pop[:100]
        
        return self.best_ever['error'], self.best_ever['weight'], self.best_ever['bias']


def main():
    print('='*60)
    print('Phase 9: Evolution with Elitism')
    print('='*60)
    print('\nPattern: y = 0.5 * x\n')
    
    evo = Evolution(pop_size=100)
    
    for gen in range(200):
        error, w, b = evo.step()
        
        if gen % 50 == 0:
            print(f'Gen {gen}: error={error:.6f}, w={w:.4f}, b={b:.4f}')
    
    print('\n=== Result ===')
    print(f'Expected: w=0.5, b=0')
    print(f'Found: w={w:.6f}, b={b:.6f}')
    
    if error < 0.001:
        print('\n[SUCCESS] Evolution found the exact pattern!')
        print('Structure emerged from random soup!')
    else:
        print(f'\n[Close] Error: {error:.6f}')


if __name__ == "__main__":
    main()
