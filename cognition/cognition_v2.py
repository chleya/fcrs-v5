"""
认知涌现实验 v2: 改进版
"""

import numpy as np


class Env:
    def __init__(self, mode='periodic'):
        self.mode = mode
        self.phase = 0
    
    def generate(self):
        self.phase += 1
        
        if self.mode == 'periodic':
            return np.sin(self.phase * 0.5)
        elif self.mode == 'random':
            return np.random.randn()
        return 0


class Agent:
    def __init__(self, has_evolution=False):
        self.has_evolution = has_evolution
        self.weights = np.random.randn(10) * 0.2
        self.energy = 10
        self.fitness = 0
        self.alive = True
    
    def forward(self, x):
        return np.dot(self.weights, x)
    
    def step(self, x, target):
        pred = self.forward(x)
        error = abs(pred - target)
        
        # 学习
        self.weights += (target - pred) * x * 0.01
        
        # 能量
        if self.has_evolution:
            self.energy -= 0.1
            if error < 0.5:
                self.energy += 0.3
                self.fitness += 1
            if self.energy <= 0:
                self.alive = False
        
        return error


def run_experiment():
    print('='*60)
    print('Cognition Emergence Experiment')
    print('='*60)
    
    # 测试不同配置
    configs = [
        ('No Evo, Periodic', False, 'periodic'),
        ('No Evo, Random', False, 'random'),
        ('With Evo, Periodic', True, 'periodic'),
        ('With Evo, Random', True, 'random'),
    ]
    
    results = []
    
    for name, has_evo, mode in configs:
        print(f'\n=== {name} ===')
        
        env = Env(mode)
        
        # 初始种群
        agents = [Agent(has_evolution=has_evo) for _ in range(50)]
        
        errors = []
        
        for gen in range(100):
            # 生成数据
            x = np.random.randn(10)
            target = env.generate()
            
            # 每个agent预测
            gen_errors = []
            for agent in agents:
                if not agent.alive:
                    continue
                err = agent.step(x, target)
                gen_errors.append(err)
            
            if gen_errors:
                errors.append(np.mean(gen_errors))
            
            # 演化
            if has_evo and gen % 10 == 9:
                # 选择
                survivors = sorted(agents, key=lambda a: a.fitness, reverse=True)[:10]
                
                # 复制
                new_agents = []
                for parent in survivors:
                    for _ in range(5):
                        child = Agent(has_evolution=True)
                        child.weights = parent.weights + np.random.randn(10) * 0.1
                        child.energy = 5
                        new_agents.append(child)
                
                agents = new_agents[:50]
        
        final_error = errors[-1] if errors else 1.0
        print(f'Error: {errors[0]:.3f} -> {final_error:.3f}')
        
        results.append({
            'name': name,
            'initial': errors[0],
            'final': final_error,
            'improvement': errors[0] - final_error
        })
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    print('\nCognition Emergence:')
    for r in results:
        status = 'OK' if r['improvement'] > 0 else 'WARN'
        print(f'{r["name"]}: {r["improvement"]:+.3f} [{status}]')
    
    # 关键发现
    print('\nKey Finding:')
    evo_periodic = [r for r in results if 'Evo' in r['name'] and 'Periodic' in r['name']]
    if evo_periodic and evo_periodic[0]['improvement'] > 0:
        print('[OK] Evolution + Pattern -> Cognition!')
    else:
        print('[Need more work]')


if __name__ == "__main__":
    run_experiment()
