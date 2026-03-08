"""
规律检测实验: 智能的源头 (修复版)
"""

import numpy as np


class PatternEnv:
    def __init__(self, type='periodic', period=10):
        self.type = type
        self.period = period
        self.step = 0
    
    def generate(self):
        self.step += 1
        
        if self.type == 'periodic':
            return np.sin(2 * np.pi * self.step / self.period)
        elif self.type == 'random':
            return np.random.randn() * 2
        elif self.type == 'complex':
            base = np.sin(2 * np.pi * self.step / self.period)
            return base + np.random.randn() * 0.5
        return 0


class PatternSystem:
    def __init__(self):
        self.memory = []
        self.attention = 0.5
    
    def detect_and_respond(self, value):
        self.memory.append(value)
        
        if len(self.memory) < 20:
            return self.attention
        
        # 检测规律性
        recent = self.memory[-20:]
        var = np.var(recent)
        
        # 方差判断规律性
        if var < 0.1:
            regularity = 1.0  # 强规律
        elif var < 0.5:
            regularity = 0.5  # 中等
        else:
            regularity = 0.0  # 无规律
        
        # 自适应响应
        # 有规律 → 放松(低注意力)
        # 无规律 → 警觉(高注意力)
        if regularity > 0.7:
            self.attention = self.attention * 0.9 + 0.1 * 0.2
        elif regularity < 0.3:
            self.attention = self.attention * 0.9 + 0.1 * 0.8
        else:
            pass  # 保持
        
        self.attention = max(0.1, min(1.0, self.attention))
        
        return regularity


def main():
    print('='*60)
    print('Pattern Detection: Origin of Intelligence')
    print('='*60)
    
    results = []
    
    for ptype in ['periodic', 'random', 'complex']:
        env = PatternEnv(type=ptype)
        system = PatternSystem()
        
        for _ in range(200):
            v = env.generate()
            system.detect_and_respond(v)
        
        final_attention = system.attention
        final_var = np.var(system.memory[-20:])
        
        results.append({
            'type': ptype,
            'attention': final_attention,
            'variance': final_var
        })
        
        print(f'\n{ptype}:')
        print(f'  Attention: {final_attention:.3f}')
        print(f'  Variance: {final_var:.3f}')
    
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    # 检查
    if results[0]['attention'] < results[1]['attention']:
        print('[OK] Periodic < Random: Pattern detected!')
    else:
        print('[WARN] No differentiation')


if __name__ == "__main__":
    main()
