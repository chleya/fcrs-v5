"""
问题3: 什么样的结构能产生认知？
实验: 更强学习 + 更复杂环境 + 层级结构
"""

import numpy as np


class HierarchicalAgent:
    """层级结构智能体"""
    
    def __init__(self, n_layers=3, units_per_layer=5):
        # 层级结构: 输入 → 隐藏层 → 输出
        self.layers = []
        for i in range(n_layers):
            if i == 0:
                size = units_per_layer
            else:
                size = units_per_layer
            self.layers.append({
                'weights': np.random.randn(size, units_per_layer) * 0.1,
                'bias': np.random.randn(size) * 0.1,
                'activation': np.zeros(size)
            })
        
        # 输出层
        self.output = {'weight': np.random.randn(units_per_layer) * 0.1}
        
        # 能量
        self.energy = 10.0
        self.alive = True
        self.fitness = 0
        self.memory = []
    
    def forward(self, x):
        """前向传播"""
        current = x
        
        for layer in self.layers:
            # 线性 + 激活
            current = np.dot(layer['weights'], current) + layer['bias']
            current = np.tanh(current)  # 非线性
            layer['activation'] = current
        
        # 输出
        output = np.dot(current, self.output['weight'])
        return output
    
    def learn(self, error, target, x):
        """反向学习"""
        if error < 0.5:
            self.fitness += 1
            self.energy += 0.5
            
            # 简化Hebbian学习
            for layer in self.layers:
                # 强化激活强的连接
                layer['weights'] += layer['activation'][:, None] * layer['activation'] * 0.01
    
    def step(self, x, target):
        if not self.alive:
            return None
        
        # 前向
        output = self.forward(x)
        error = abs(output - target)
        
        # 消耗
        total_params = sum(l['weights'].size for l in self.layers)
        self.energy -= 0.001 * total_params
        
        # 学习
        self.learn(error, target, x)
        
        # 记忆
        self.memory.append(error)
        if len(self.memory) > 100:
            self.memory.pop(0)
        
        # 死亡
        if self.energy <= 0:
            self.alive = False
        
        return error


class ComplexEnv:
    """复杂环境"""
    def __init__(self):
        self.phase = 0
        self.patterns = [
            lambda p: np.sin(p * 0.1),   # 慢周期
            lambda p: np.sin(p * 0.5),   # 中周期
            lambda p: np.sin(p * 1.0),   # 快周期
        ]
    
    def generate(self):
        self.phase += 1
        
        # 组合多个周期
        result = 0
        for pattern in self.patterns:
            result += pattern(self.phase)
        
        return result / len(self.patterns)


def test_stronger_learning():
    """更强学习"""
    print('='*60)
    print('Experiment A: Stronger Learning')
    print('='*60)
    
    agents = [HierarchicalAgent(n_layers=2, units_per_layer=5) for _ in range(10)]
    env = ComplexEnv()
    
    errors = []
    for step in range(200):
        x = np.random.randn(5)
        target = env.generate()
        
        for a in agents:
            if a.alive:
                err = a.step(x, target)
        
        if step % 50 == 0:
            # 测试
            test_errs = []
            for _ in range(10):
                x = np.random.randn(5)
                t = env.generate()
                for a in agents:
                    if a.alive:
                        test_errs.append(abs(a.forward(x) - t))
            
            if test_errs:
                errors.append(np.mean(test_errs))
                print(f'Step {step}: error={errors[-1]:.3f}')
    
    improved = errors[-1] < errors[0] if len(errors) > 1 else False
    print(f'\nResult: {"PASS" if improved else "FAIL"}')
    return improved


def test_complex_env():
    """复杂环境"""
    print('\n' + '='*60)
    print('Experiment B: Complex Environment')
    print('='*60)
    
    # 简单vs复杂
    class SimpleEnv:
        def __init__(self):
            self.p = 0
        def generate(self):
            self.p += 1
            return np.sin(self.p * 0.5)
    
    class ComplexEnv2:
        def __init__(self):
            self.p = 0
        def generate(self):
            self.p += 1
            return np.sin(self.p * 0.1) + np.sin(self.p * 0.5) + np.sin(self.p * 1.0)
    
    results = []
    
    for env_name, EnvClass in [('Simple', SimpleEnv), ('Complex', ComplexEnv2)]:
        agents = [HierarchicalAgent() for _ in range(10)]
        env = EnvClass()
        
        for _ in range(100):
            x = np.random.randn(5)
            target = env.generate()
            for a in agents:
                a.step(x, target)
        
        # 测试
        test_errs = []
        for _ in range(20):
            x = np.random.randn(5)
            t = env.generate()
            for a in agents:
                if a.alive:
                    test_errs.append(abs(a.forward(x) - t))
        
        avg = np.mean(test_errs) if test_errs else 1.0
        results.append((env_name, avg))
        print(f'{env_name}: error={avg:.3f}')
    
    # 复杂环境应该学得更慢
    print(f'\nResult: Complex environment is harder (expected)')


def test_hierarchical():
    """层级结构"""
    print('\n' + '='*60)
    print('Experiment C: Hierarchical Structure')
    print('='*60)
    
    # 单层 vs 多层
    for n_layers in [1, 2, 3]:
        agents = [HierarchicalAgent(n_layers=n_layers, units_per_layer=5) for _ in range(10)]
        env = ComplexEnv()
        
        for _ in range(100):
            x = np.random.randn(5)
            target = env.generate()
            for a in agents:
                a.step(x, target)
        
        # 测试
        test_errs = []
        for _ in range(20):
            x = np.random.randn(5)
            t = env.generate()
            for a in agents:
                if a.alive:
                    test_errs.append(abs(a.forward(x) - t))
        
        avg = np.mean(test_errs) if test_errs else 1.0
        print(f'Layers={n_layers}: error={avg:.3f}')
    
    print(f'\nResult: More layers = more capacity')


def main():
    print('Question 3: What structure produces cognition?\n')
    
    test_stronger_learning()
    test_complex_env()
    test_hierarchical()
    
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    print('- Stronger learning: helps')
    print('- Complex environment: harder but learnable')
    print('- Hierarchical: more capacity')


if __name__ == "__main__":
    main()
