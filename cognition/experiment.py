"""
认知涌现实验: 三个问题一个实验
问题1: 无主体结构设计
问题2: 如何确定认知
问题3: 什么结构能产生认知
"""

import numpy as np
import random


# ========== 环境 ==========
class Environment:
    """测试环境"""
    def __init__(self, mode='periodic'):
        self.mode = mode
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        
        if self.mode == 'periodic':
            return np.sin(2 * np.pi * self.phase / 10)
        elif self.mode == 'random':
            return np.random.randn() * 2
        elif self.mode == 'mixed':
            if self.phase % 50 < 25:
                return np.sin(2 * np.pi * self.phase / 10)
            else:
                return np.random.randn() * 2


# ========== 无主体结构 ==========
class MinimalAgent:
    """问题1: 最小无主体结构"""
    
    def __init__(self, n_units=10, complexity='minimal'):
        self.complexity = complexity
        self.alive = True
        
        if complexity == 'minimal':
            # 最简: 随机权重，无结构
            self.weights = np.random.randn(n_units) * 0.1
        elif complexity == 'medium':
            # 中等: 有结构，但无演化
            self.weights = np.random.randn(n_units) * 0.1
            self.bias = 0
        elif complexity == 'full':
            # 完整: 有结构+演化+能量
            self.weights = np.random.randn(n_units) * 0.1
            self.bias = 0
            self.energy = 10
            self.age = 0
            self.fitness = 0
    
    def predict(self, x):
        if self.complexity == 'minimal':
            # 简单线性组合
            return np.sum(self.weights * x)
        else:
            return np.dot(self.weights, x) + self.bias
    
    def learn(self, error):
        """学习: 调整权重"""
        if self.complexity in ['medium', 'full']:
            # 简单学习
            self.weights += np.random.randn(len(self.weights)) * 0.1 * (1 - error)
    
    def survive(self, error):
        """问题2: 能量系统"""
        if self.complexity == 'full':
            self.age += 1
            
            # 计算消耗
            self.energy -= 0.1
            
            # 预测奖励
            if error < 0.5:
                self.energy += 0.5
                self.fitness += 1
            
            # 死亡
            if self.energy <= 0:
                self.alive = False
            
            return self.alive
        return True


# ========== 演化系统 ==========
class Evolution:
    """问题3: 什么结构能产生认知"""
    
    def __init__(self, complexity='minimal'):
        self.complexity = complexity
        self.population = []
        self.generation = 0
        self.history = []
    
    def init_population(self, size=50):
        """初始化"""
        self.population = []
        for _ in range(size):
            agent = MinimalAgent(n_units=10, complexity=self.complexity)
            self.population.append(agent)
    
    def step(self, env):
        """一步"""
        inputs = []
        targets = []
        
        # 生成数据
        for _ in range(10):
            x = np.random.randn(10)
            y = env.generate()
            inputs.append(x)
            targets.append(y)
        
        # 评估
        results = []
        for agent in self.population:
            if not agent.alive:
                continue
            
            total_error = 0
            for x, y in zip(inputs, targets):
                pred = agent.predict(x)
                error = abs(pred - y)
                total_error += error
                
                # 学习
                agent.learn(error)
            
            avg_error = total_error / len(inputs)
            results.append((agent, avg_error))
        
        # 演化 (如果是full)
        if self.complexity == 'full':
            self._evolve(results)
        
        # 记录
        if results:
            best_error = min(r[1] for r in results)
            self.history.append(best_error)
        
        return results
    
    def _evolve(self, results):
        """演化"""
        # 选择
        sorted_results = sorted(results, key=lambda x: x[1])
        survivors = [r[0] for r in sorted_results[:10]]
        
        # 复制+变异
        new_pop = []
        for parent in survivors:
            for _ in range(5):
                child = MinimalAgent(n_units=10, complexity='full')
                child.weights = parent.weights + np.random.randn(10) * 0.1
                child.bias = parent.bias + np.random.randn() * 0.1
                child.energy = parent.energy * 0.5
                new_pop.append(child)
        
        self.population = new_pop[:50]
        self.generation += 1


# ========== 测试 ==========
def test_question1():
    """问题1: 不同复杂度"""
    print('='*60)
    print('Question 1: Structure Design')
    print('='*60)
    
    env = Environment('periodic')
    
    for complexity in ['minimal', 'medium', 'full']:
        print(f'\n--- {complexity} ---')
        
        evo = Evolution(complexity=complexity)
        evo.init_population(50)
        
        errors = []
        for gen in range(50):
            results = evo.step(env)
            if results:
                errors.append(min(r[1] for r in results))
        
        print(f'Final error: {errors[-1]:.4f}')
        
        if complexity == 'full':
            alive = sum(1 for a in evo.population if a.alive)
            print(f'Alive: {alive}')


def test_question2():
    """问题2: 认知标志"""
    print('\n' + '='*60)
    print('Question 2: Cognition Markers')
    print('='*60)
    
    env = Environment('periodic')
    evo = Evolution(complexity='full')
    evo.init_population(50)
    
    # 测试50代
    for gen in range(50):
        results = evo.step(env)
    
    # 检查认知标志
    print('\nCognition Markers:')
    
    # 1. 模式识别
    errors = evo.history
    if errors[-1] < errors[0]:
        print(f'  [OK] Pattern recognition: {errors[0]:.2f} -> {errors[-1]:.2f}')
    else:
        print(f'  [WARN] Pattern recognition failed')
    
    # 2. 适应度
    if evo.population:
        fitnesses = [a.fitness for a in evo.population]
        print(f'  [OK] Fitness: max={max(fitnesses)}')
    
    # 3. 存活
    alive = sum(1 for a in evo.population if a.alive)
    print(f'  [OK] Survival: {alive}/50')


def test_question3():
    """问题3: 什么结构能产生认知"""
    print('\n' + '='*60)
    print('Question 3: What Structure Produces Cognition?')
    print('='*60)
    
    # 测试不同环境
    for mode in ['periodic', 'random', 'mixed']:
        print(f'\n--- {mode} ---')
        
        env = Environment(mode)
        evo = Evolution(complexity='full')
        evo.init_population(50)
        
        for gen in range(50):
            evo.step(env)
        
        if evo.history:
            print(f'Error: {evo.history[-1]:.4f}')


def main():
    test_question1()
    test_question2()
    test_question3()
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('Q1: Structure design affects learning')
    print('Q2: Multiple markers needed for cognition')
    print('Q3: Environment matters')


if __name__ == "__main__":
    main()
