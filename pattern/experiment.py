"""
规律检测实验: 智能的源头
测试: 系统能否检测规律并响应？
"""

import numpy as np
import random


class Pattern:
    """规律"""
    def __init__(self, type='periodic', params=None):
        self.type = type
        self.params = params or {}
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        
        if self.type == 'periodic':
            # 周期性规律: sin(phase)
            period = self.params.get('period', 10)
            value = np.sin(2 * np.pi * self.phase / period)
            return value
        
        elif self.type == 'linear':
            # 线性规律: a*x + b
            a = self.params.get('a', 0.1)
            return a * self.phase
        
        elif self.type == 'random':
            # 无规律
            return np.random.randn()
        
        elif self.type == 'complex':
            # 复杂规律: sin + noise
            period = self.params.get('period', 20)
            base = np.sin(2 * np.pi * self.phase / period)
            noise = np.random.randn() * self.params.get('noise', 0.1)
            return base + noise
        
        return 0


class PatternDetector:
    """规律检测系统"""
    
    def __init__(self):
        self.memory = []
        self.predictions = []
        self.errors = []
        
        # 表征: 预测模型
        self.model_params = {
            'period': 10,
            'weight': 0.5
        }
    
    def detect(self, value):
        """检测规律"""
        self.memory.append(value)
        
        if len(self.memory) < 3:
            return None
        
        # 检测周期性
        recent = self.memory[-10:]
        
        # 计算自相关
        if len(recent) >= 5:
            mean = np.mean(recent)
            var = np.var(recent)
            
            # 如果方差小 = 有规律
            if var < 0.5:
                regularity = 1 - var
            else:
                regularity = 0
        else:
            regularity = 0
        
        return regularity
    
    def predict(self):
        """预测下一个值"""
        if len(self.memory) < 3:
            return 0
        
        # 简单预测: 移动平均
        recent = self.memory[-5:]
        pred = np.mean(recent)
        
        self.predictions.append(pred)
        return pred
    
    def update(self, actual):
        """更新模型"""
        error = 0
        
        if self.predictions:
            error = abs(actual - self.predictions[-1])
            self.errors.append(error)
        
        return error


class AdaptiveSystem:
    """自适应系统: 根据规律调整"""
    
    def __init__(self):
        self.detector = PatternDetector()
        self.attention = 0.5  # 注意力水平
        self.energy = 100
    
    def step(self, value):
        # 1. 检测规律
        regularity = self.detector.detect(value)
        
        # 2. 预测
        prediction = self.detector.predict()
        
        # 3. 实际值
        error = self.detector.update(value)
        
        # 4. 自适应注意力
        # 有规律 → 低注意力；无规律 → 高注意力
        if regularity is not None:
            if regularity > 0.7:
                # 高规律 → 放松
                self.attention *= 0.95
                self.energy = min(100, self.energy + 1)
            else:
                # 低规律 → 警觉
                self.attention *= 1.1
                self.energy = max(0, self.energy - 1)
        
        self.attention = max(0.1, min(1.0, self.attention))
        
        return {
            'value': value,
            'regularity': regularity,
            'prediction': prediction,
            'error': error,
            'attention': self.attention,
            'energy': self.energy
        }


def run_experiment():
    """运行实验"""
    print('='*60)
    print('Pattern Detection: Origin of Intelligence')
    print('='*60)
    
    # 测试不同规律
    patterns = [
        ('Periodic', 'periodic', {'period': 10}),
        ('Linear', 'linear', {'a': 0.1}),
        ('Random', 'random', {}),
        ('Complex', 'complex', {'period': 20, 'noise': 0.2}),
    ]
    
    results = []
    
    for name, ptype, params in patterns:
        print(f'\n=== {name} ===')
        
        pattern = Pattern(ptype, params)
        system = AdaptiveSystem()
        
        # 运行
        for _ in range(100):
            value = pattern.generate()
            result = system.step(value)
        
        # 统计
        regularity = [r['regularity'] for r in [system.detector.detect(system.detector.memory[i]) 
                                                  for i in range(10, 100)] 
                       if r is not None]
        
        avg_regularity = np.mean(regularity) if regularity else 0
        avg_error = np.mean(system.detector.errors[-50:])
        avg_attention = system.attention
        
        results.append({
            'pattern': name,
            'regularity': avg_regularity,
            'error': avg_error,
            'attention': avg_attention
        })
        
        print(f'Regularity: {avg_regularity:.2f}')
        print(f'Prediction Error: {avg_error:.3f}')
        print(f'Attention: {avg_attention:.2f}')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    print('\nPattern Detection:')
    for r in results:
        print(f"{r['pattern']}: regularity={r['regularity']:.2f}, attention={r['attention']:.2f}")
    
    # 检测是否工作
    periodic = results[0]
    random_ = results[2]
    
    if periodic['attention'] < random_['attention']:
        print('\n[OK] System detects patterns!')
        print('    Periodic -> Lower attention')
        print('    Random -> Higher attention')
    else:
        print('\n[WARN] No pattern detection')


if __name__ == "__main__":
    run_experiment()
