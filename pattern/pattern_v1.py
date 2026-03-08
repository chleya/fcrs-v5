"""
规律检测实验: 智能的源头
"""

import numpy as np


class PatternEnv:
    """规律环境"""
    def __init__(self, type='periodic', period=10):
        self.type = type
        self.period = period
        self.step = 0
    
    def generate(self):
        self.step += 1
        
        if self.type == 'periodic':
            return np.sin(2 * np.pi * self.step / self.period)
        elif self.type == 'random':
            return np.random.randn()
        elif self.type == 'complex':
            base = np.sin(2 * np.pi * self.step / self.period)
            return base + np.random.randn() * 0.3
        return 0


class PatternSystem:
    """规律检测系统"""
    
    def __init__(self):
        self.memory = []
        self.attention = 0.5
    
    def detect(self, value):
        self.memory.append(value)
        
        if len(self.memory) < 10:
            return 0.5
        
        recent = self.memory[-10:]
        
        # 检测规律性: 方差小=有规律
        var = np.var(recent)
        
        if var < 0.3:
            regularity = 1.0
        elif var < 0.7:
            regularity = 0.5
        else:
            regularity = 0.0
        
        # 自适应: 有规律→低注意力, 无规律→高注意力
        if regularity > 0.7:
            self.attention *= 0.95
        else:
            self.attention *= 1.1
        
        self.attention = max(0.1, min(1.0, self.attention))
        
        return regularity


def main():
    print('='*60)
    print('Pattern Detection: Origin of Intelligence')
    print('='*60)
    
    # 测试不同规律
    tests = [
        ('Periodic', 'periodic'),
        ('Random', 'random'),
        ('Complex', 'complex'),
    ]
    
    for name, ptype in tests:
        print(f'\n=== {name} ===')
        
        env = PatternEnv(type=ptype)
        system = PatternSystem()
        
        # 运行
        for _ in range(100):
            v = env.generate()
            system.detect(v)
        
        print(f'Final Attention: {system.attention:.3f}')
        print(f'Memory var: {np.var(system.memory[-10:]):.3f}')
    
    print('\n' + '='*60)
    print('Expected:')
    print('  Periodic -> Low attention (detected pattern)')
    print('  Random -> High attention (no pattern)')
    print('  Complex -> Medium attention')


if __name__ == "__main__":
    main()
