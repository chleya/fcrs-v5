"""
问题2: 如何确定"长出认知"？
测试认知标志: 模式识别、泛化、记忆、能动
"""

import numpy as np


class Agent:
    """无主体智能体"""
    
    def __init__(self, capacity=10):
        self.dna = np.random.randn(capacity)
        self.energy = 10.0
        self.meta_rate = 0.01
        self.alive = True
        self.fitness = 0
        self.memory = []  # 记忆
    
    def process(self, x):
        return np.dot(self.dna, x)
    
    def step(self, x, target, threshold=0.5):
        if not self.alive:
            return None
        
        output = self.process(x)
        error = abs(output - target)
        
        # 消耗
        self.energy -= self.meta_rate * len(self.dna)
        
        # 学习
        if error < threshold:
            self.energy += 1.0
            self.fitness += 1
            # 学习调整
            self.dna += (target - output) * x * 0.1
        
        # 记忆
        self.memory.append({
            'dna': self.dna.copy(),
            'fitness': self.fitness
        })
        if len(self.memory) > 50:
            self.memory.pop(0)
        
        # 死亡
        if self.energy <= 0:
            self.alive = False
        
        return error


def test_pattern_recognition():
    """测试1: 模式识别"""
    print('='*60)
    print('Test 1: Pattern Recognition')
    print('='*60)
    
    env_phase = 0
    def env():
        nonlocal env_phase
        env_phase += 1
        return np.sin(env_phase * 0.5)
    
    agents = [Agent() for _ in range(10)]
    
    errors = []
    for step in range(100):
        x = np.random.randn(10)
        target = env()
        
        for a in agents:
            if a.alive:
                a.step(x, target)
        
        if step % 20 == 0:
            # 测试
            test_x = np.random.randn(10)
            test_target = env()
            errs = [abs(a.process(test_x) - test_target) for a in agents if a.alive]
            if errs:
                errors.append(np.mean(errs))
                print(f'Step {step}: error={errors[-1]:.3f}')
    
    # 判断
    improved = errors[-1] < errors[0] if len(errors) > 1 else False
    print(f'\nResult: {"PASS" if improved else "FAIL"}')
    return improved


def test_generalization():
    """测试2: 泛化"""
    print('\n' + '='*60)
    print('Test 2: Generalization')
    print('='*60)
    
    # 训练环境: 周期性
    agents = [Agent() for _ in range(10)]
    
    for step in range(50):
        x = np.random.randn(10)
        target = np.sin(step * 0.5)
        
        for a in agents:
            a.step(x, target)
    
    # 测试环境: 不同的周期性
    test_errors = []
    for step in range(20):
        x = np.random.randn(10)
        target = np.sin(step * 0.3)  # 不同频率
        
        for a in agents:
            if a.alive:
                err = abs(a.process(x) - target)
                test_errors.append(err)
    
    avg_error = np.mean(test_errors)
    print(f'Generalization error: {avg_error:.3f}')
    
    result = avg_error < 1.0
    print(f'Result: {"PASS" if result else "FAIL"}')
    return result


def test_memory():
    """测试3: 记忆"""
    print('\n' + '='*60)
    print('Test 3: Memory')
    print('='*60)
    
    agents = [Agent() for _ in range(10)]
    
    # 训练
    for step in range(50):
        x = np.random.randn(10)
        target = np.sin(step * 0.5)
        
        for a in agents:
            a.step(x, target)
    
    # 检查记忆
    living = [a for a in agents if a.alive]
    if living:
        # DNA应该稳定
        dnas = [a.dna for a in living]
        variance = np.var([np.mean(d) for d in dnas])
        print(f'DNA variance: {variance:.4f}')
        
        # 记忆存在
        memory_exists = any(len(a.memory) > 10 for a in living)
        print(f'Has memory: {memory_exists}')
        
        result = memory_exists
        print(f'Result: {"PASS" if result else "FAIL"}')
        return result
    
    return False


def test_agency():
    """测试4: 能动性"""
    print('\n' + '='*60)
    print('Test 4: Agency')
    print('='*60)
    
    agents = [Agent() for _ in range(10)]
    
    # 统计行为
    total_actions = 0
    successful_actions = 0
    
    for step in range(50):
        x = np.random.randn(10)
        target = np.sin(step * 0.5)
        
        for a in agents:
            if a.alive:
                err = a.step(x, target)
                total_actions += 1
                if err is not None:
                    successful_actions += 1
    
    # 存活率
    survival_rate = sum(1 for a in agents if a.alive) / len(agents)
    print(f'Survival rate: {survival_rate:.1%}')
    
    # 活跃度
    activity = successful_actions / total_actions if total_actions > 0 else 0
    print(f'Activity: {activity:.1%}')
    
    result = survival_rate > 0.5
    print(f'Result: {"PASS" if result else "FAIL"}')
    return result


def main():
    results = {}
    
    results['pattern'] = test_pattern_recognition()
    results['generalization'] = test_generalization()
    results['memory'] = test_memory()
    results['agency'] = test_agency()
    
    print('\n' + '='*60)
    print('Summary: Cognition Markers')
    print('='*60)
    
    for name, passed in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'{status} - {name}')
    
    # 总体判断
    passed = sum(results.values())
    print(f'\nTotal: {passed}/4 passed')
    
    if passed >= 3:
        print('\n[RESULT] System shows cognition!')
    else:
        print('\n[RESULT] More work needed')


if __name__ == "__main__":
    main()
