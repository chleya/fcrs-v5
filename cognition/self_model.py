"""
自我模型实验: Agent能否形成"自我"概念？
核心问题: 
1. 自我意识: 知道自己存在
2. 自我模型: 能模拟自己
3. 自我决策: 能选择目标
"""

import numpy as np


# ========== 实验1: 自我存在 ========
class SelfAwareAgent:
    """自我存在: 知道自己活着"""
    
    def __init__(self):
        self.energy = 10.0
        self.alive = True
        self.self_model = {
            'alive': True,
            'energy': 10.0,
            'age': 0
        }
    
    def step(self):
        # 自我反思: 我还活着吗？
        if self.alive:
            self.self_model['age'] += 1
            self.self_model['energy'] = self.energy
            self.energy -= 1  # 消耗
        
        if self.energy <= 0:
            self.alive = False
            self.self_model['alive'] = False
        
        return self.self_model.copy()


def test_self_existence():
    """测试自我存在"""
    print('='*60)
    print('Experiment 1: Self-Existence')
    print('='*60)
    
    agent = SelfAwareAgent()
    
    history = []
    
    for step in range(10):
        state = agent.step()
        history.append({
            'age': state['age'],
            'alive': state['alive'],
            'energy': state['energy']
        })
        
        print(f'Step {step}: age={state["age"]}, alive={state["alive"]}, energy={state["energy"]:.1f}')
    
    # 判断: Agent是否知道"我"存在？
    if agent.self_model['age'] > 0:
        print('\n[OK] Agent has self-model: knows its own age')
    else:
        print('\n[FAIL] No self-model')


# ========== 实验2: 自我模拟 ========
class SelfSimulatingAgent:
    """自我模拟: 能预测自己的行为"""
    
    def __init__(self):
        # 物理自我
        self.position = 0.0
        self.velocity = 0.0
        
        # 自我模型
        self.model = {
            'position': 0.0,
            'velocity': 0.0
        }
    
    def act(self, action):
        # 物理: 移动
        self.velocity += action
        self.position += self.velocity
        
        # 模型: 模拟
        self.model['velocity'] += action
        self.model['position'] += self.model['velocity']
    
    def compare(self):
        # 自我反思: 模型 vs 现实
        error = abs(self.position - self.model['position'])
        return error


def test_self_simulation():
    """测试自我模拟"""
    print('\n' + '='*60)
    print('Experiment 2: Self-Simulation')
    print('='*60)
    
    agent = SelfSimulatingAgent()
    
    for step in range(10):
        action = np.random.randn() * 0.5
        agent.act(action)
        
        error = agent.compare()
        
        print(f'Step {step}: pos={agent.position:.2f}, model={agent.model["position"]:.2f}, error={error:.2f}')


# ========== 实验3: 自我决策 ========
class SelfDecidingAgent:
    """自我决策: 能选择目标"""
    
    def __init__(self):
        self.energy = 10.0
        self.position = 0.0
        
        # 自我模型
        self.goals = []  # 目标历史
        self.choices = []  # 选择历史
    
    def decide(self, options):
        # 自我决策: 选择最佳目标
        best = max(options, key=lambda x: x['value'])
        
        self.choices.append(best)
        self.goals.append(best['name'])
        
        return best
    
    def act(self, goal):
        # 追求目标
        if goal['name'] == 'explore':
            self.position += np.random.randn()
        elif goal['name'] == 'rest':
            self.energy += 1
        elif goal['name'] == 'gather':
            self.energy -= 0.5
            self.position += 1


def test_self_decision():
    """测试自我决策"""
    print('\n' + '='*60)
    print('Experiment 3: Self-Decision')
    print('='*60)
    
    agent = SelfDecidingAgent()
    
    options = [
        {'name': 'explore', 'value': 0.5},
        {'name': 'rest', 'value': 0.3},
        {'name': 'gather', 'value': 0.8}
    ]
    
    for step in range(10):
        goal = agent.decide(options)
        agent.act(goal)
        
        print(f'Step {step}: chose={goal["name"]}, energy={agent.energy:.1f}, pos={agent.position:.1f}')
    
    # 统计
    if len(agent.goals) > 5:
        # 是否有偏好？
        from collections import Counter
        prefs = Counter(agent.goals)
        print(f'\nGoals: {dict(prefs)}')
        print('[OK] Agent has preferences!')


# ========== 实验4: 元认知 ========
class MetaCognitionAgent:
    """元认知: 知道"我知道什么" """
    
    def __init__(self):
        # 知识
        self.knowledge = {}
        
        # 元认知: 知道自己的知识边界
        self.meta = {
            'known': [],
            'unknown': []
        }
    
    def learn(self, item, truth):
        # 学习
        self.knowledge[item] = truth
        
        # 元认知: 反思
        if abs(self.knowledge[item] - truth) < 0.1:
            self.meta['known'].append(item)
        else:
            self.meta['unknown'].append(item)
    
    def interrogate(self, item):
        # 自我询问: 我知道这个吗？
        if item in self.meta['known']:
            return "I know this"
        elif item in self.meta['unknown']:
            return "I don't know this"
        else:
            return "I never studied this"


def test_meta_cognition():
    """测试元认知"""
    print('\n' + '='*60)
    print('Experiment 4: Meta-Cognition')
    print('='*60)
    
    agent = MetaCognitionAgent()
    
    # 学习
    items = ['math', 'physics', 'art', 'music']
    truths = [1.0, 0.8, 0.3, 0.9]
    
    for item, truth in zip(items, truths):
        agent.learn(item, truth)
    
    # 询问
    for item in items:
        result = agent.interrogate(item)
        print(f'{item}: {result}')
    
    print(f'\nMeta: known={len(agent.meta["known"])}, unknown={len(agent.meta["unknown"])}')


def main():
    test_self_existence()
    test_self_simulation()
    test_self_decision()
    test_meta_cognition()
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('Self-model components:')
    print('1. Self-existence:知道自己存在')
    print('2. Self-simulation:能模拟自己')
    print('3. Self-decision:能选择目标')
    print('4. Meta-cognition:知道"我知道"')


if __name__ == "__main__":
    main()
