"""
Phase 9 v3: The Emergent Agent - Pure Evolution
纯演化: 长期稳定环境，看结构是否涌现
"""

import numpy as np


class PeriodicEnv:
    """稳定周期性环境"""
    def __init__(self):
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        # 纯周期信号
        return np.sin(2 * np.pi * self.phase / 10)


class EvolvingNetwork:
    """演化网络"""
    
    def __init__(self, pop_size=100):
        # 初始种群
        self.population = []
        for _ in range(pop_size):
            self.population.append({
                'weights': np.random.randn(10) * 0.5,
                'bias': np.random.randn() * 0.1,
                'fitness': 0,
            })
        
        self.generation = 0
        self.history = []
    
    def evaluate(self, env):
        """评估整个种群"""
        total_error = 0
        
        for _ in range(10):  # 每个个体测试10次
            x = np.random.randn(10)
            actual = env.generate()
            
            best_error = float('inf')
            for ind in self.population:
                pred = np.dot(ind['weights'], x) + ind['bias']
                error = abs(pred - actual)
                if error < best_error:
                    best_error = error
            
            total_error += best_error
        
        return total_error / 10
    
    def evolve(self):
        """演化一代"""
        # 评估
        errors = []
        for ind in self.population:
            x = np.random.randn(10)
            actual = np.sin(2 * np.pi * (self.generation + 1) / 10)
            
            pred = np.dot(ind['weights'], x) + ind['bias']
            error = abs(pred - actual)
            errors.append(error)
            
            # 适应度 = 1/误差
            ind['fitness'] = 1.0 / (error + 0.1)
        
        # 选择 (top 20%)
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        survivors = sorted_pop[:20]
        
        # 复制 + 变异
        new_pop = []
        for parent in survivors:
            for _ in range(5):  # 每个父母产生5个孩子
                child = {
                    'weights': parent['weights'] + np.random.randn(10) * 0.1,
                    'bias': parent['bias'] + np.random.randn() * 0.05,
                    'fitness': 0,
                }
                new_pop.append(child)
        
        self.population = new_pop[:100]  # 保持种群大小
        self.generation += 1
        
        return np.mean(errors)


def main():
    print('='*60)
    print('Phase 9 v3: Pure Evolution')
    print('='*60)
    print('\nGoal: Watch structure emerge from random soup')
    print('Environment: Stable periodic signal\n')
    
    env = PeriodicEnv()
    network = EvolvingNetwork(pop_size=100)
    
    # 运行
    for gen in range(50):
        error = network.evolve()
        
        if gen % 10 == 0:
            # 检查最佳个体
            best = max(network.population, key=lambda x: x['fitness'])
            print(f'Gen {gen}: error={error:.4f}, best_fitness={best["fitness"]:.2f}')
            print(f'  Best weights: {best["weights"][:3]}')
    
    # 最终分析
    print('\n=== Final Analysis ===')
    
    # 按适应度排序
    sorted_pop = sorted(network.population, key=lambda x: x['fitness'], reverse=True)
    best = sorted_pop[0]
    
    print(f'Best fitness: {best["fitness"]:.4f}')
    print(f'Best weights: {best["weights"]}')
    
    # 检查是否学到了周期结构
    # 如果学到了，应该有一组权重对应周期性
    weight_pattern = np.dot(best['weights'], best['weights'])
    print(f'\nWeight pattern (self-correlation): {weight_pattern:.4f}')
    
    if best['fitness'] > 5:
        print('\n[OK] Evolution found a good solution!')
    else:
        print('\n[WARN] Still searching...')


if __name__ == "__main__":
    main()
