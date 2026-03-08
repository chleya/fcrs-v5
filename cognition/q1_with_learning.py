"""
问题1: 无主体结构 + 学习机制
"""

import numpy as np


class Agent:
    """无主体 + 学习"""
    
    def __init__(self, capacity=10):
        self.dna = np.random.randn(capacity)  # 随机结构
        self.energy = 10.0
        self.meta_rate = 0.01
        self.age = 0
        self.alive = True
        self.fitness = 0  # 适应度
    
    def process(self, env_input):
        return np.dot(self.dna, env_input)
    
    def step(self, env_input, target, threshold=0.5, reward_gain=1.0):
        if not self.alive:
            return None
        
        self.age += 1
        
        # 1. 预测
        output = self.process(env_input)
        error = abs(output - target)
        
        # 2. 消耗
        energy_cost = self.meta_rate * len(self.dna)
        self.energy -= energy_cost
        
        # 3. 学习 + 奖励 (关键！)
        if error < threshold:
            self.energy += reward_gain
            self.fitness += 1
            
            # 学习: 误差小 → 强化当前方向
            self.dna += (target - output) * env_input * 0.1
        else:
            # 没预测对 → 变异尝试新方向
            if self.energy < 5.0:
                mutation = np.random.normal(0, 0.3, size=len(self.dna))
                self.dna += mutation
        
        # 4. 死亡
        if self.energy <= 0:
            self.alive = False
        
        return error


class Environment:
    def __init__(self):
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        return np.sin(self.phase * 0.5)


def run():
    print('='*60)
    print('Question 1: Agent with Learning')
    print('='*60)
    
    agents = [Agent(capacity=10) for _ in range(20)]
    env = Environment()
    
    for step in range(200):
        x = np.random.randn(10)
        target = env.generate()
        
        errors = []
        for agent in agents:
            if not agent.alive:
                continue
            err = agent.step(x, target)
            errors.append(err)
        
        # 补充新agent
        alive = sum(1 for a in agents if a.alive)
        if alive < 10:
            agents.append(Agent(capacity=10))
        
        if step % 50 == 0 and errors:
            print(f'Step {step}: error={np.mean(errors):.3f}, alive={alive}')
    
    # 分析
    print('\n=== Result ===')
    
    # 检查DNA变化
    living = [a for a in agents if a.alive]
    if living:
        avg_dna = np.mean([len(a.dna) for a in living])
        print(f'DNA size: {avg_dna:.1f}')
        
        # 检查适应度
        fitness = [a.fitness for a in living]
        print(f'Fitness: max={max(fitness)}, avg={np.mean(fitness):.1f}')
        
        # 误差变化
        errors = [abs(a.process(np.random.randn(10)) - env.generate()) for a in living]
        print(f'Error: {np.mean(errors):.3f}')


if __name__ == "__main__":
    run()
