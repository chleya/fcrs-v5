"""
模式识别 v3: 修复输入问题
问题: 输入应该是"有意义的"，不是随机噪声
"""

import numpy as np


class SmartAgent:
    """智能体: 输入应该是phase信息"""
    
    def __init__(self):
        # 输入是phase的表示 (sin/cos)
        self.weights = np.random.randn(2) * 0.1  # sin, cos
        self.energy = 20.0
        self.alive = True
        self.fitness = 0
    
    def forward(self, phase_repr):
        # phase_repr = [sin(phase), cos(phase)]
        return np.dot(self.weights, phase_repr)
    
    def learn(self, phase_repr, target, output):
        error = target - output
        
        if abs(error) < 0.3:
            self.fitness += 1
            self.energy += 0.3
            
            # 学习: 调整权重
            self.weights += 0.1 * error * phase_repr
    
    def step(self, phase, target):
        if not self.alive:
            return None
        
        # 相位表示
        phase_repr = np.array([np.sin(phase), np.cos(phase)])
        
        output = self.forward(phase_repr)
        error = abs(output - target)
        
        self.energy -= 0.01
        self.learn(phase_repr, target, output)
        
        if self.energy <= 0:
            self.alive = False
        
        return error


class PhaseEnv:
    """相位环境: 给定phase，预测sin"""
    def __init__(self):
        self.phase = 0
    
    def generate(self):
        self.phase += 0.2
        return np.sin(self.phase)


def test_direct_phase():
    """直接相位学习"""
    print('='*60)
    print('Pattern V3: Direct Phase Learning')
    print('='*60)
    print('Input: [sin(phase), cos(phase)]')
    print('Output: sin(phase)\n')
    
    agents = [SmartAgent() for _ in range(10)]
    env = PhaseEnv()
    
    errors = []
    
    for step in range(500):
        phase = env.phase
        target = env.generate()
        
        for a in agents:
            if a.alive:
                a.step(phase, target)
        
        if step % 50 == 0:
            # 测试
            test_errs = []
            for _ in range(20):
                p = env.phase
                t = env.generate()
                p_repr = np.array([np.sin(p), np.cos(p)])
                
                for a in agents:
                    if a.alive:
                        out = a.forward(p_repr)
                        test_errs.append(abs(out - t))
            
            if test_errs:
                errors.append(np.mean(test_errs))
                print(f'Step {step}: error={errors[-1]:.4f}')
    
    print('\n=== Result ===')
    print(f'Initial: {errors[0]:.4f}')
    print(f'Final: {errors[-1]:.4f}')
    print(f'Improvement: {errors[0] - errors[-1]:.4f}')
    
    # 学的什么
    if agents:
        w = agents[0].weights
        print(f'\nLearned weights: {w}')
        print(f'Expected: w=[1, 0] (since sin = sin*1 + cos*0)')
    
    improved = errors[-1] < errors[0] * 0.5
    print(f'\nResult: {"PASS" if improved else "FAIL"}')
    
    return improved


def test_generalization_v2():
    """测试泛化"""
    print('\n' + '='*60)
    print('Test: Generalization')
    print('='*60)
    
    # 训练: 频率=0.2
    env = PhaseEnv()
    env.phase = 0
    
    agents = [SmartAgent() for _ in range(10)]
    
    for _ in range(300):
        p = env.phase
        t = env.generate()
        for a in agents:
            a.step(p, t)
    
    # 测试: 不同的phase
    print('\nTest at different phases:')
    for test_phase in [0, 1, 3.14, 6.28]:
        target = np.sin(test_phase)
        p_repr = np.array([np.sin(test_phase), np.cos(test_phase)])
        
        preds = [a.forward(p_repr) for a in agents if a.alive]
        if preds:
            avg_pred = np.mean(preds)
            print(f'Phase={test_phase:.2f}: target={target:.3f}, pred={avg_pred:.3f}')


def main():
    passed = test_direct_phase()
    test_generalization_v2()
    
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    
    if passed:
        print('[OK] Pattern learned!')
    else:
        print('[Key insight] Input must be meaningful phase info')


if __name__ == "__main__":
    main()
