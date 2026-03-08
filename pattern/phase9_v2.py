"""
Phase 9 v2: The Emergent Agent - Harder Challenge
更强的生存压力 + 更复杂的环境
"""

import numpy as np
import random


class ComplexEnv:
    """复杂环境: 周期性 + 随机噪声混合"""
    def __init__(self):
        self.phase = 0
        self.pattern = 'periodic'  # 初始模式
    
    def generate(self):
        self.phase += 1
        
        # 每100步切换模式
        if self.phase % 100 == 0:
            self.pattern = 'random' if self.pattern == 'periodic' else 'periodic'
        
        if self.pattern == 'periodic':
            # 强周期性
            return np.sin(2 * np.pi * self.phase / 10)
        else:
            # 纯随机
            return np.random.randn() * 3


class EmergentAgent:
    """涌现的主体 - 更强压力"""
    
    def __init__(self, n_units=50):
        # 随机神经汤
        self.units = []
        for _ in range(n_units):
            self.units.append({
                'weights': np.random.randn(10) * 0.2,
                'bias': 0,
                'energy': 1.0,
                'age': 0,
                'fitness': 0,  # 适应度
            })
        
        self.energy_pool = 50
        self.history = []
    
    def forward(self, x):
        """每个单元计算输出"""
        outputs = []
        for unit in self.units:
            out = np.dot(unit['weights'], x) + unit['bias']
            outputs.append(out)
        return outputs
    
    def step(self, x, env):
        # 获取环境输出
        actual = env.generate()
        
        # 前向传播
        predictions = self.forward(x)
        
        # 评估
        if not self.units:
            return 1.0
            
        errors = [abs(p - actual) for p in predictions]
        best_idx = np.argmin(errors)
        best_error = errors[best_idx]
        
        # === 强生存压力 ===
        
        # 1. 计算消耗能量 (所有单元)
        for unit in self.units:
            unit['energy'] -= 0.02  # 提高消耗
        
        # 2. 奖励预测好的
        if best_error < 0.5:
            self.units[best_idx]['energy'] += 1.0
            self.units[best_idx]['fitness'] += 1
            self.energy_pool += 0.5
        
        # 3. 惩罚预测差的
        for i, unit in enumerate(self.units):
            if errors[i] > 1.0:
                unit['energy'] -= 0.1  # 额外惩罚
        
        # 4. 死亡机制
        dead_indices = [i for i, u in enumerate(self.units) if u['energy'] < 0]
        
        # 5. 出生机制 (精英复制)
        # 适应度最高的单元复制
        fitnesses = [u['fitness'] for u in self.units]
        if fitnesses and max(fitnesses) > 5:
            best_unit_idx = np.argmax(fitnesses)
            best_unit = self.units[best_unit_idx]
            
            # 变异复制
            new_unit = {
                'weights': best_unit['weights'] + np.random.randn(10) * 0.1,
                'bias': best_unit['bias'] + np.random.randn() * 0.1,
                'energy': 1.0,
                'age': 0,
                'fitness': 0,
            }
            
            self.units.append(new_unit)
            self.units[best_unit_idx]['energy'] -= 0.5  # 分裂消耗
        
        # 删除死亡单元
        for i in sorted(dead_indices, reverse=True):
            del self.units[i]
        
        # 安全网: 如果单元太少，补充随机单元
        if len(self.units) < 5:
            for _ in range(10):
                self.units.append({
                    'weights': np.random.randn(10) * 0.2,
                    'bias': 0,
                    'energy': 1.0,
                    'age': 0,
                    'fitness': 0,
                })
        
        # 记录
        self.history.append({
            'n_units': len(self.units),
            'best_error': best_error,
            'avg_energy': np.mean([u['energy'] for u in self.units]) if self.units else 0,
        })
        
        return best_error


def main():
    print('='*60)
    print('Phase 9 v2: Emergent Agent - Harder')
    print('='*60)
    print('\nChallenge:')
    print('  - Complex environment: periodic + random')
    print('  - Strong pressure: high cost, low reward')
    print('  - Goal: structure emerges from nothing\n')
    
    env = ComplexEnv()
    agent = EmergentAgent(n_units=50)
    
    # 运行
    for step in range(1000):
        x = np.random.randn(10)
        error = agent.step(x, env)
        
        if step % 200 == 0:
            print(f'Step {step}: units={len(agent.units)}, error={error:.3f}')
    
    # 分析
    print('\n=== Final Analysis ===')
    print(f'Units: {len(agent.units)}')
    
    if len(agent.units) > 1:
        # 检查适应度分布
        fitnesses = [u['fitness'] for u in agent.units]
        print(f'Fitness: max={max(fitnesses)}, mean={np.mean(fitnesses):.1f}')
        
        # 检查权重分化
        weights = np.array([u['weights'] for u in agent.units])
        weight_norms = [np.linalg.norm(w) for w in weights]
        print(f'Weight norms: max={max(weight_norms):.3f}, min={min(weight_norms):.3f}')
        
        # 结构分化?
        if max(fitnesses) > 10:
            print('\n[OK] Some units became specialists!')
            
            # 找到最佳单元
            best_idx = np.argmax(fitnesses)
            best_weights = agent.units[best_idx]['weights']
            print(f'Best unit weights: {best_weights[:3]}...')
        else:
            print('\n[WARN] No specialization yet')


if __name__ == "__main__":
    main()
