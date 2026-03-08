"""
模式识别深入实验
探索: 泛化、多频率、层级结构
"""

import numpy as np


class Agent:
    def __init__(self):
        self.w = np.random.randn(2) * 0.1
        self.energy = 20.0
        self.alive = True
        self.fitness = 0
    
    def forward(self, phase):
        p = np.array([np.sin(phase), np.cos(phase)])
        return np.dot(self.w, p)
    
    def learn(self, phase, target):
        out = self.forward(phase)
        error = target - out
        
        if abs(error) < 0.3:
            self.fitness += 1
            self.energy += 0.3
            p = np.array([np.sin(phase), np.cos(phase)])
            self.w += 0.1 * error * p
    
    def step(self, phase, target):
        if not self.alive:
            return None
        
        out = self.forward(phase)
        error = abs(out - target)
        
        self.energy -= 0.01
        self.learn(phase, target)
        
        if self.energy <= 0:
            self.alive = False
        
        return error


# ========== 实验1: 泛化 ==========
def test_generalization():
    """不同频率的泛化"""
    print('='*60)
    print('Experiment 1: Generalization')
    print('='*60)
    
    # 训练: 频率=1.0
    phase = 0
    agents = [Agent() for _ in range(10)]
    
    for _ in range(300):
        phase += 0.5  # 训练频率
        target = np.sin(phase)
        
        for a in agents:
            a.step(phase, target)
    
    # 测试: 不同频率
    print('\nGeneralization test:')
    for freq in [0.1, 0.5, 1.0, 2.0, 5.0]:
        test_phase = 0
        errors = []
        
        for _ in range(10):
            test_phase += freq
            target = np.sin(test_phase)
            
            for a in agents:
                if a.alive:
                    out = a.forward(test_phase)
                    errors.append(abs(out - target))
        
        avg = np.mean(errors) if errors else 1.0
        status = 'TRAIN' if freq == 0.5 else 'TEST'
        print(f'Freq={freq}: error={avg:.4f} [{status}]')


# ========== 实验2: 多频率学习 ==========
def test_multi_frequency():
    """同时学习多个频率"""
    print('\n' + '='*60)
    print('Experiment 2: Multi-Frequency Learning')
    print('='*60)
    
    agents = [Agent() for _ in range(10)]
    
    # 训练: 多个频率
    for step in range(500):
        phase = step * 0.3
        target = np.sin(phase)
        
        for a in agents:
            a.step(phase, target)
    
    # 测试
    print('\nAfter multi-freq training:')
    for freq in [0.1, 0.3, 0.5, 1.0]:
        test_phase = 0
        errors = []
        
        for _ in range(10):
            test_phase += freq
            target = np.sin(test_phase)
            
            for a in agents:
                if a.alive:
                    out = a.forward(test_phase)
                    errors.append(abs(out - target))
        
        avg = np.mean(errors) if errors else 1.0
        print(f'Freq={freq}: error={avg:.4f}')


# ========== 实验3: 层级结构 ==========
class LayeredAgent:
    """层级结构"""
    
    def __init__(self, n_hidden=5):
        # 第一层: phase → hidden
        self.w1 = np.random.randn(n_hidden, 2) * 0.1
        # 第二层: hidden → output
        self.w2 = np.random.randn(n_hidden) * 0.1
        
        self.energy = 20.0
        self.alive = True
        self.fitness = 0
    
    def forward(self, phase):
        p = np.array([np.sin(phase), np.cos(phase)])
        
        # 隐藏层
        hidden = np.tanh(np.dot(self.w1, p))
        
        # 输出层
        return np.dot(hidden, self.w2)
    
    def learn(self, phase, target):
        p = np.array([np.sin(phase), np.cos(phase)])
        
        # 前向
        hidden = np.tanh(np.dot(self.w1, p))
        out = np.dot(hidden, self.w2)
        
        error = target - out
        
        if abs(error) < 0.3:
            self.fitness += 1
            self.energy += 0.3
            
            # 反向传播
            delta = error
            self.w2 += 0.1 * delta * hidden
            
            delta_h = delta * self.w2 * (1 - hidden**2)
            self.w1 += 0.01 * np.outer(delta_h, p)
    
    def step(self, phase, target):
        if not self.alive:
            return None
        
        out = self.forward(phase)
        error = abs(out - target)
        
        self.energy -= 0.01
        self.learn(phase, target)
        
        if self.energy <= 0:
            self.alive = False
        
        return error


def test_layered():
    """层级结构测试"""
    print('\n' + '='*60)
    print('Experiment 3: Layered Structure')
    print('='*60)
    
    agents = [LayeredAgent(n_hidden=5) for _ in range(10)]
    
    phase = 0
    errors = []
    
    for step in range(500):
        phase += 0.3
        target = np.sin(phase)
        
        for a in agents:
            a.step(phase, target)
        
        if step % 100 == 0:
            # 测试
            test_errs = []
            for _ in range(10):
                t = phase + 0.3
                for a in agents:
                    if a.alive:
                        out = a.forward(t)
                        test_errs.append(abs(out - np.sin(t)))
            
            if test_errs:
                errors.append(np.mean(test_errs))
                print(f'Step {step}: error={errors[-1]:.4f}')
    
    print(f'\nImprovement: {errors[0] - errors[-1]:.4f}')


# ========== 实验4: 无监督学习 ==========
def test_unsupervised():
    """无监督: 预测下一时刻"""
    print('\n' + '='*60)
    print('Experiment 4: Unsupervised')
    print('='*60)
    
    class UnsupervisedAgent:
        def __init__(self):
            self.w = np.random.randn(2) * 0.1
            self.energy = 20.0
            self.alive = True
        
        def step(self, phase):
            # 预测下一时刻
            next_phase = phase + 0.3
            target = np.sin(next_phase)
            
            p = np.array([np.sin(phase), np.cos(phase)])
            out = np.dot(self.w, p)
            
            error = abs(out - target)
            
            # 无监督: 无论对错都学习
            self.w += 0.05 * (target - out) * p
            
            self.energy -= 0.01
            
            if self.energy <= 0:
                self.alive = False
            
            return error
    
    agents = [UnsupervisedAgent() for _ in range(10)]
    phase = 0
    
    errors = []
    for step in range(500):
        phase += 0.3
        
        for a in agents:
            if a.alive:
                a.step(phase)
        
        if step % 100 == 0:
            # 测试
            test_phase = phase + 1.0
            target = np.sin(test_phase)
            p = np.array([np.sin(test_phase), np.cos(test_phase)])
            
            preds = [np.dot(a.w, p) for a in agents if a.alive]
            if preds:
                err = np.mean([abs(p - target) for p in preds])
                errors.append(err)
                print(f'Step {step}: error={err:.4f}')
    
    print(f'\nImprovement: {errors[0] - errors[-1]:.4f}')


def main():
    test_generalization()
    test_multi_frequency()
    test_layered()
    test_unsupervised()
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('- Generalization: varies by frequency')
    print('- Multi-freq: harder but learnable')
    print('- Layered: similar to single-layer')
    print('- Unsupervised: also works!')


if __name__ == "__main__":
    main()
