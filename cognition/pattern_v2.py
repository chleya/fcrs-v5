"""
认知研究: 深入模式识别
问题: 为什么模式识别失败？如何改进？
"""

import numpy as np


class ImprovedAgent:
    """改进的智能体: 更强学习"""
    
    def __init__(self):
        # 更大网络
        self.weights1 = np.random.randn(20, 10) * 0.1  # 输入→隐藏
        self.weights2 = np.random.randn(20) * 0.1       # 隐藏→输出
        self.hidden = np.zeros(20)
        
        self.energy = 20.0
        self.alive = True
        self.fitness = 0
    
    def forward(self, x):
        # 隐藏层
        self.hidden = np.dot(self.weights1, x)
        self.hidden = np.tanh(self.hidden)
        
        # 输出层
        output = np.dot(self.hidden, self.weights2)
        return output
    
    def learn(self, x, target, output):
        error = target - output
        
        # 强学习: 反向传播
        if abs(error) < 0.5:
            self.fitness += 1
            self.energy += 0.5
            
            # 反向梯度
            delta2 = error  # 输出层误差
            
            # 更新输出层
            self.weights2 += 0.1 * delta2 * self.hidden
            
            # 更新输入层
            delta1 = delta2 * self.weights2 * (1 - self.hidden**2)
            self.weights1 += 0.01 * np.outer(delta1, x)
    
    def step(self, x, target):
        if not self.alive:
            return None
        
        output = self.forward(x)
        error = abs(output - target)
        
        # 消耗
        self.energy -= 0.01
        
        # 学习
        self.learn(x, target, output)
        
        # 死亡
        if self.energy <= 0:
            self.alive = False
        
        return error


class PeriodicEnv:
    """周期环境"""
    def __init__(self, frequency=0.5):
        self.phase = 0
        self.freq = frequency
    
    def generate(self):
        self.phase += 1
        return np.sin(self.phase * self.freq)


def test_pattern_recognition_v2():
    """模式识别 v2"""
    print('='*60)
    print('Pattern Recognition V2: Stronger Learning')
    print('='*60)
    
    agents = [ImprovedAgent() for _ in range(10)]
    env = PeriodicEnv(frequency=0.5)
    
    errors = []
    
    # 训练
    for step in range(500):
        x = np.random.randn(10)
        target = env.generate()
        
        for a in agents:
            if a.alive:
                a.step(x, target)
        
        # 每50步测试
        if step % 50 == 0:
            test_errs = []
            for _ in range(20):
                x = np.random.randn(10)
                t = env.generate()
                for a in agents:
                    if a.alive:
                        out = a.forward(x)
                        test_errs.append(abs(out - t))
            
            if test_errs:
                errors.append(np.mean(test_errs))
                print(f'Step {step}: error={errors[-1]:.4f}')
    
    # 结果
    print('\n=== Result ===')
    print(f'Initial error: {errors[0]:.4f}')
    print(f'Final error: {errors[-1]:.4f}')
    print(f'Improvement: {errors[0] - errors[-1]:.4f}')
    
    # 适应度
    living = [a for a in agents if a.alive]
    if living:
        fitness = [a.fitness for a in living]
        print(f'Fitness: max={max(fitness)}, avg={np.mean(fitness):.1f}')
    
    # 判断
    improved = errors[-1] < errors[0] * 0.8
    print(f'\nResult: {"PASS" if improved else "FAIL"}')
    
    return improved


def test_different_frequencies():
    """测试不同频率"""
    print('\n' + '='*60)
    print('Test: Different Frequencies')
    print('='*60)
    
    # 训练频率
    train_freq = 0.5
    env_train = PeriodicEnv(frequency=train_freq)
    
    agents = [ImprovedAgent() for _ in range(10)]
    
    # 训练
    for _ in range(300):
        x = np.random.randn(10)
        target = env_train.generate()
        for a in agents:
            a.step(x, target)
    
    # 测试不同频率
    print('\nGeneralization test:')
    for freq in [0.3, 0.5, 0.7, 1.0]:
        env_test = PeriodicEnv(frequency=freq)
        
        test_errs = []
        for _ in range(20):
            x = np.random.randn(10)
            t = env_test.generate()
            for a in agents:
                if a.alive:
                    test_errs.append(abs(a.forward(x) - t))
        
        avg = np.mean(test_errs) if test_errs else 1.0
        marker = 'TRAIN' if freq == train_freq else 'TEST'
        print(f'Freq={freq}: error={avg:.4f} [{marker}]')


def main():
    passed = test_pattern_recognition_v2()
    test_different_frequencies()
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    if passed:
        print('[OK] Stronger learning helps pattern recognition!')
    else:
        print('[Need work] Still struggling with patterns')


if __name__ == "__main__":
    main()
