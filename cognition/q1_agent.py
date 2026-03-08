"""
问题1: 无主体结构设计
实现您的优雅设计
"""

import numpy as np


class Agent:
    """您的优雅设计"""
    
    def __init__(self, capacity=10):
        self.dna = np.random.randn(capacity)  # 原始随机结构
        self.energy = 10.0  # 初始能量储备
        self.meta_rate = 0.01  # 基础代谢率
        self.age = 0
        self.alive = True
    
    def process(self, env_input):
        """基于DNA产生响应"""
        # 简单: DNA作为权重
        return np.dot(self.dna, env_input)
    
    def step(self, env_input, last_error, threshold=0.5, reward_gain=1.0):
        """一步"""
        if not self.alive:
            return None
        
        self.age += 1
        
        # 1. 动作
        output = self.process(env_input)
        
        # 2. 消耗: 结构越大，消耗越高
        energy_cost = self.meta_rate * len(self.dna)
        self.energy -= energy_cost
        
        # 3. 补偿: 预测好 = 奖励
        if last_error < threshold:
            self.energy += reward_gain
        
        # 4. 变异: 能量不足时尝试自救
        if self.energy < 5.0:
            # 随机变异
            mutation = np.random.normal(0, 0.5, size=len(self.dna))
            self.dna += mutation
            self.energy += 0.5  # 变异消耗能量
        
        # 5. 死亡
        if self.energy <= 0:
            self.alive = False
        
        return output


class Environment:
    """周期性环境"""
    def __init__(self):
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        return np.sin(self.phase * 0.5)  # 周期信号


def run():
    print('='*60)
    print('Question 1: Agent Design')
    print('='*60)
    print('\nDesign:')
    print('- DNA: random initial structure')
    print('- Energy: reserve')
    print('- Meta rate: cost of living')
    print('- Error < threshold: reward')
    print('- Energy < 5: mutate\n')
    
    # 初始化
    agents = [Agent(capacity=10) for _ in range(20)]
    env = Environment()
    
    print('Running...\n')
    
    history = {
        'energy': [],
        'alive': [],
        'dna_size': [],
        'error': []
    }
    
    for step in range(200):
        # 环境输入
        x = np.random.randn(10)
        target = env.generate()
        
        # 每个agent预测
        errors = []
        for agent in agents:
            if not agent.alive:
                continue
            
            output = agent.process(x)
            error = abs(output - target)
            errors.append(error)
            
            # Agent决策
            agent.step(x, error)
        
        if errors:
            avg_error = np.mean(errors)
            avg_energy = np.mean([a.energy for a in agents if a.alive])
            alive_count = sum(1 for a in agents if a.alive)
            avg_dna = np.mean([len(a.dna) for a in agents if a.alive])
            
            history['error'].append(avg_error)
            history['energy'].append(avg_energy)
            history['alive'].append(alive_count)
            history['dna_size'].append(avg_dna)
        
        # 补充新agent
        if alive_count < 5:
            agents.append(Agent(capacity=10))
        
        if step % 50 == 0:
            print(f'Step {step}: error={avg_error:.3f}, energy={avg_energy:.2f}, alive={alive_count}')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    print(f'\nInitial error: {history["error"][0]:.3f}')
    print(f'Final error: {history["error"][-1]:.3f}')
    print(f'Improvement: {history["error"][0] - history["error"][-1]:.3f}')
    
    # DNA变化
    print(f'\nInitial DNA size: {history["dna_size"][0]:.1f}')
    print(f'Final DNA size: {history["dna_size"][-1]:.1f}')
    
    # 存活
    print(f'\nInitial alive: {history["alive"][0]}')
    print(f'Final alive: {history["alive"][-1]}')


if __name__ == "__main__":
    run()
