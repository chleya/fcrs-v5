"""
规律检测实验: 智能的源头 (最终版)
"""

import numpy as np


class PatternEnv:
    """不同类型的规律环境"""
    def __init__(self, type='periodic', period=10):
        self.type = type
        self.period = period
        self.step = 0
    
    def generate(self):
        self.step += 1
        
        if self.type == 'periodic':
            # 纯周期 - 规律最强
            return np.sin(2 * np.pi * self.step / self.period)
        
        elif self.type == 'random':
            # 随机 - 无规律
            return np.random.randn() * 2
        
        elif self.type == 'complex':
            # 周期+噪声 - 中等规律
            base = np.sin(2 * np.pi * self.step / self.period)
            return base + np.random.randn() * 0.3
        
        return 0


class PatternSystem:
    """规律检测系统"""
    
    def __init__(self):
        self.memory = []
        self.attention = 0.5  # 初始注意力
        self.history = []
    
    def detect_and_respond(self, value):
        self.memory.append(value)
        self.history.append(value)
        
        if len(self.memory) < 30:
            return
        
        # 滑动窗口分析
        window = self.memory[-30:]
        
        # 计算方差 (规律性指标)
        var = np.var(window)
        
        # 计算自相关 (周期性指标)
        if len(window) > 10:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0
        else:
            autocorr = 0
        
        # 规律性得分
        # 低方差 + 高自相关 = 强规律
        regularity = 0
        
        if var < 0.3:
            regularity += 0.5
        elif var > 1.0:
            regularity -= 0.3
        
        if autocorr > 0.5:
            regularity += 0.5
        elif autocorr < -0.5:
            regularity -= 0.3
        
        regularity = max(0, min(1, regularity))
        
        # 响应规律
        # 规律强 → 低注意力(放松)
        # 规律弱 → 高注意力(警觉)
        
        if regularity > 0.7:
            # 检测到规律，降低注意力
            self.attention = self.attention * 0.95 + 0.2 * 0.05
        elif regularity < 0.3:
            # 无规律，提高注意力
            self.attention = self.attention * 0.95 + 0.8 * 0.05
        
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
        
        # 运行
        for _ in range(300):
            v = env.generate()
            system.detect_and_respond(v)
        
        # 最终状态
        final_attention = system.attention
        window = system.memory[-30:]
        final_var = np.var(window)
        
        if len(window) > 10:
            final_autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
            if np.isnan(final_autocorr):
                final_autocorr = 0
        else:
            final_autocorr = 0
        
        results.append({
            'type': ptype,
            'attention': final_attention,
            'variance': final_var,
            'autocorr': final_autocorr
        })
        
        print(f'\n{ptype}:')
        print(f'  Attention: {final_attention:.3f}')
        print(f'  Variance: {final_var:.3f}')
        print(f'  Autocorr: {final_autocorr:.3f}')
    
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    # 分析
    periodic_attn = results[0]['attention']
    random_attn = results[1]['attention']
    
    print(f'\nPeriodic attention: {periodic_attn:.3f}')
    print(f'Random attention: {random_attn:.3f}')
    
    if periodic_attn < random_attn:
        print('\n[OK] System shows ADAPTIVE pattern detection!')
        print('    Strong pattern (periodic) -> Low attention')
        print('    No pattern (random) -> High attention')
        print('\nThis is the origin of intelligence!')
    else:
        print('\n[WARN] No pattern detection')


if __name__ == "__main__":
    main()
