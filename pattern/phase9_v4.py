"""
Phase 9 v4: The Emergent Agent - Fixed Input
修正: 输入是固定的模式，预测输出
"""

import numpy as np


class PatternEnv:
    """模式环境: 输入-输出配对"""
    def __init__(self):
        self.phase = 0
        # 固定的输入模式
        self.input_pattern = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0])
    
    def generate(self):
        self.phase += 1
        # 输出 = 周期性
        return np.sin(2 * np.pi * self.phase / 10)


class EvolvingAgent:
    """演化智能体"""
    
    def __init__(self, pop_size=100):
        self.population = []
        for _ in range(pop_size):
            self.population.append({
                'weights': np.random.randn(10) * 0.5,
                'bias': 0,
                'fitness': 0,
            })
        
        self.env = PatternEnv()
        self.generation = 0
    
    def evaluate(self):
        """评估"""
        total_fit = 0
        
        # 固定输入
        x = self.env.input_pattern
        
        for ind in self.population:
            # 预测
            pred = np.dot(ind['weights'], x) + ind['bias']
            
            # 实际
            actual = self.env.generate()
            
            # 误差
            error = abs(pred - actual)
            
            # 适应度
            ind['fitness'] = 1.0 / (error + 0.01)
            total_fit += ind['fitness']
        
        return total_fit / len(self.population)
    
    def evolve(self):
        """演化"""
        # 评估
        self.evaluate()
        
        # 选择 top 20%
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        survivors = sorted_pop[:20]
        
        # 复制 + 变异
        new_pop = []
        for parent in survivors:
            for _ in range(5):
                child = {
                    'weights': parent['weights'] + np.random.randn(10) * 0.2,
                    'bias': parent['bias'] + np.random.randn() * 0.1,
                    'fitness': 0,
                }
                new_pop.append(child)
        
        self.population = new_pop[:100]
        self.generation += 1
        
        # 返回最佳误差
        best = sorted_pop[0]
        x = self.env.input_pattern
        pred = np.dot(best['weights'], x) + best['bias']
        actual = self.env.generate()
        
        return abs(pred - actual), best


def main():
    print('='*60)
    print('Phase 9 v4: Emergent Agent - Fixed Pattern')
    print('='*60)
    
    agent = EvolvingAgent(pop_size=100)
    
    for gen in range(50):
        error, best = agent.evolve()
        
        if gen % 10 == 0:
            print(f'Gen {gen}: error={error:.4f}')
            print(f'  Weights: {best["weights"][:5]}')
            print(f'  Bias: {best["bias"]:.4f}')
    
    print('\n=== Final ===')
    best = max(agent.population, key=lambda x: x['fitness'])
    x = agent.env.input_pattern
    pred = np.dot(best['weights'], x) + best['bias']
    actual = agent.env.generate()
    
    print(f'Prediction: {pred:.4f}')
    print(f'Actual: {actual:.4f}')
    print(f'Error: {abs(pred-actual):.4f}')
    
    if abs(pred - actual) < 0.5:
        print('\n[OK] System learned the pattern!')


if __name__ == "__main__":
    main()
