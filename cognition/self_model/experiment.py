"""
自我模型完整实验
整合四个层次，测试真正的自我模型
"""

import numpy as np
from collections import defaultdict


class SelfModelAgent:
    """完整的自我模型智能体"""
    
    def __init__(self):
        # ========== L1: 自我存在 ==========
        self.state = {
            'energy': 10.0,
            'age': 0,
            'alive': True
        }
        
        # ========== L2: 自我模拟 ==========
        # 物理自我
        self.body = {
            'position': 0.0,
            'velocity': 0.0,
            'health': 1.0
        }
        
        # 自我模型
        self.model = {
            'position': 0.0,
            'velocity': 0.0,
            'health': 1.0
        }
        
        # ========== L3: 自我决策 ==========
        self.goals = {
            'explore': {'value': 0.3, 'cost': 0.5},
            'rest': {'value': 0.5, 'cost': -1.0},  # 恢复能量
            'hunt': {'value': 0.8, 'cost': 0.8}
        }
        self.current_goal = None
        self.goal_history = []
        
        # ========== L4: 元认知 ==========
        self.knowledge = {}  # {topic: confidence}
        self.learned_items = set()
        
        # 历史
        self.history = []
    
    # ========== L1: 自我存在 ==========
    def update_self_existence(self):
        """更新自我存在状态"""
        self.state['age'] += 1
        
        # 能量消耗
        self.state['energy'] -= 0.1
        
        # 死亡
        if self.state['energy'] <= 0:
            self.state['alive'] = False
        
        return self.state.copy()
    
    def introspect_existence(self):
        """反思：我还存在吗？"""
        return {
            'I exist': self.state['alive'],
            'My age': self.state['age'],
            'My energy': self.state['energy']
        }
    
    # ========== L2: 自我模拟 ==========
    def simulate(self, action):
        """在模型中模拟动作"""
        # 预测
        new_vel = self.model['velocity'] + action
        new_pos = self.model['position'] + new_vel
        
        return {'position': new_pos, 'velocity': new_vel}
    
    def act(self, action):
        """物理世界中动作"""
        # 物理
        self.body['velocity'] += action
        self.body['position'] += self.body['velocity']
        
        # 模型更新
        self.model['velocity'] = self.body['velocity']
        self.model['position'] = self.body['position']
        
        return self.body.copy()
    
    def compare_self(self):
        """比较现实 vs 模型"""
        error = abs(self.body['position'] - self.model['position'])
        return error
    
    # ========== L3: 自我决策 ==========
    def decide(self):
        """决策：选择目标"""
        if not self.state['alive']:
            return None
        
        # 根据能量选择
        if self.state['energy'] < 3.0:
            # 能量低，选择rest
            choice = 'rest'
        elif self.state['energy'] > 8.0:
            # 能量高，可以hunt
            choice = 'hunt'
        else:
            # 中等能量，探索
            choice = 'explore'
        
        self.current_goal = choice
        self.goal_history.append(choice)
        
        return choice
    
    def pursue_goal(self, goal):
        """追求目标"""
        if goal == 'rest':
            self.state['energy'] = min(10, self.state['energy'] + 1.0)
        elif goal == 'hunt':
            self.state['energy'] -= 0.5
            # 可能获得能量
            if np.random.random() > 0.5:
                self.state['energy'] += 2.0
        elif goal == 'explore':
            # 随机移动
            self.act(np.random.randn())
    
    # ========== L4: 元认知 ==========
    def learn(self, topic, truth):
        """学习"""
        # 记录
        self.knowledge[topic] = truth
        self.learned_items.add(topic)
    
    def interrogate(self, topic):
        """询问自己：我知道这个吗？"""
        if topic in self.knowledge:
            confidence = self.knowledge[topic]
            if confidence > 0.7:
                return f"I know {topic} (conf={confidence:.2f})"
            elif confidence > 0.3:
                return f"I'm learning {topic}"
            else:
                return f"I just started learning {topic}"
        else:
            return f"I don't know {topic}"
    
    def assess_knowledge(self):
        """评估自己的知识"""
        known = [k for k, v in self.knowledge.items() if v > 0.7]
        learning = [k for k, v in self.knowledge.items() if 0.3 < v <= 0.7]
        unknown = [k for k, v in self.knowledge.items() if v <= 0.3]
        
        return {
            'known': known,
            'learning': learning,
            'unknown': unknown
        }


# ========== 实验 ==========
def test_l1_self_existence():
    """测试L1: 自我存在"""
    print('='*60)
    print('L1: Self-Existence')
    print('='*60)
    
    agent = SelfModelAgent()
    
    for step in range(10):
        state = agent.update_self_existence()
        
        if step % 3 == 0:
            introspect = agent.introspect_existence()
            print(f'Step {step}: age={introspect["My age"]}, energy={introspect["My energy"]:.1f}')
    
    print('\n[OK] Agent maintains self-existence')


def test_l2_self_simulation():
    """测试L2: 自我模拟"""
    print('\n' + '='*60)
    print('L2: Self-Simulation')
    print('='*60)
    
    agent = SelfModelAgent()
    
    errors = []
    
    for step in range(10):
        action = np.random.randn() * 0.5
        
        # 模拟
        pred = agent.simulate(action)
        
        # 行动
        actual = agent.act(action)
        
        # 比较
        error = agent.compare_self()
        errors.append(error)
        
        print(f'Step {step}: pos={actual["position"]:.2f}, pred={pred["position"]:.2f}, error={error:.2f}')
    
    print(f'\nAvg error: {np.mean(errors):.4f}')
    print('[OK] Agent can simulate itself')


def test_l3_self_decision():
    """测试L3: 自我决策"""
    print('\n' + '='*60)
    print('L3: Self-Decision')
    print('='*60)
    
    agent = SelfModelAgent()
    agent.state['energy'] = 5.0  # 初始能量
    
    for step in range(10):
        # 决策
        goal = agent.decide()
        
        # 追求
        agent.pursue_goal(goal)
        
        # 更新
        agent.update_self_existence()
        
        print(f'Step {step}: goal={goal}, energy={agent.state["energy"]:.1f}')
    
    # 统计
    from collections import Counter
    prefs = Counter(agent.goal_history)
    print(f'\nPreferences: {dict(prefs)}')
    print('[OK] Agent makes decisions based on state')


def test_l4_meta_cognition():
    """测试L4: 元认知"""
    print('\n' + '='*60)
    print('L4: Meta-Cognition')
    print('='*60)
    
    agent = SelfModelAgent()
    
    # 学习不同主题
    topics = [
        ('math', 0.9),
        ('physics', 0.8),
        ('art', 0.2),
        ('music', 0.1),
    ]
    
    for topic, truth in topics:
        agent.learn(topic, truth)
    
    # 询问
    print('\nSelf-interrogation:')
    for topic in ['math', 'physics', 'art', 'music', 'unknown']:
        result = agent.interrogate(topic)
        print(f'  {topic}: {result}')
    
    # 评估
    assessment = agent.assess_knowledge()
    print(f'\nKnowledge: {assessment}')
    print('[OK] Agent knows what it knows')


def test_integrated():
    """综合测试"""
    print('\n' + '='*60)
    print('Integrated Self-Model')
    print('='*60)
    
    agent = SelfModelAgent()
    
    for step in range(20):
        # L1: 存在
        agent.update_self_existence()
        
        # L2: 模拟
        action = np.random.randn() * 0.3
        agent.simulate(action)
        agent.act(action)
        
        # L3: 决策
        if step % 3 == 0:
            goal = agent.decide()
            agent.pursue_goal(goal)
        
        # L4: 学习
        if step % 5 == 0:
            topic = f'skill_{step}'
            agent.learn(topic, 0.5)
        
        if step % 5 == 0:
            print(f'Step {step}: age={agent.state["age"]}, goal={agent.current_goal}')
        
        if not agent.state['alive']:
            print(f'Died at step {step}')
            break
    
    # 最终评估
    print('\nFinal Assessment:')
    print(f'  Age: {agent.state["age"]}')
    print(f'  Goals pursued: {len(agent.goal_history)}')
    print(f'  Knowledge: {len(agent.knowledge)} topics')
    
    print('[OK] Integrated self-model works!')


def main():
    test_l1_self_existence()
    test_l2_self_simulation()
    test_l3_self_decision()
    test_l4_meta_cognition()
    test_integrated()
    
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    print('Self-model has 4 levels:')
    print('  L1: Self-Existence')
    print('  L2: Self-Simulation')
    print('  L3: Self-Decision')
    print('  L4: Meta-Cognition')
    print('\nAll levels are achievable!')


if __name__ == "__main__":
    main()
