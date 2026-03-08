"""
Ablation实验: 固定维度 vs 动态维度
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem, RepresentationPool, EnvironmentLoop
import numpy as np


class FixedDimSystem:
    """固定维度系统"""
    
    def __init__(self, pool_capacity=10, vector_dim=10):
        self.env = EnvironmentLoop(input_dim=10)
        self.pool = RepresentationPool(pool_capacity, vector_dim)
        # 初始化表征
        for i in range(3):
            x = np.random.randn(10)
            self.pool.add(x)
        
        self.step_count = 0
        self.dim_history = []
        
    def step(self):
        self.step_count += 1
        x = self.env.generate_input()
        
        act, active = self.pool.activate(x)
        
        if active:
            active.activation_count += 1
            active.reuse += 1
            error = self.env.calculate_error(active.vector, x)
            active.fitness_history.append(-error)
        
        for rep in self.pool.representations:
            rep.age += 1
        
        if self.step_count % 100 == 0:
            self.dim_history.append(self.pool.get_total_dims())
    
    def run(self, steps):
        for i in range(steps):
            self.step()
        return self.get_stats()
    
    def get_stats(self):
        return {
            'total_dims': self.pool.get_total_dims(),
            'dim_history': self.dim_history,
            'pool_size': len(self.pool)
        }


class DynamicDimSystem:
    """动态维度系统 - 使用完整FCRS"""
    
    def __init__(self, pool_capacity=10, vector_dim=10):
        # 使用完整FCRS系统
        self.system = FCRSystem(pool_capacity=pool_capacity, vector_dim=vector_dim)
        # 降低阈值
        self.system.engine.spawn_reuse_threshold = 2
        self.system.engine.min_compression_gain = 0.00001
        
    def step(self):
        self.system.step()
    
    def run(self, steps):
        for i in range(steps):
            self.step()
        return self.get_stats()
    
    def get_stats(self):
        return {
            'total_dims': self.system.pool.get_total_dims(),
            'dim_history': self.system.dim_history,
            'pool_size': len(self.system.pool),
            'new_dims': len(self.system.engine.new_dim_history)
        }


def run_ablation():
    print('='*60)
    print('Ablation实验: 固定维度 vs 动态维度')
    print('='*60)
    
    steps = 1000
    
    # 固定维度系统
    print('\n[固定维度系统]')
    fixed = FixedDimSystem(pool_capacity=10, vector_dim=10)
    fixed_stats = fixed.run(steps)
    print('  总维度: ' + str(fixed_stats['total_dims']))
    print('  维度历史: ' + str(fixed_stats['dim_history']))
    
    # 动态维度系统
    print('\n[动态维度系统]')
    dynamic = DynamicDimSystem(pool_capacity=10, vector_dim=10)
    dynamic_stats = dynamic.run(steps)
    print('  总维度: ' + str(dynamic_stats['total_dims']))
    print('  维度历史: ' + str(dynamic_stats['dim_history']))
    print('  新维度诞生: ' + str(dynamic_stats['new_dims']))
    
    # 对比
    print('\n' + '='*60)
    print('对比结果')
    print('='*60)
    print('固定维度: ' + str(fixed_stats['total_dims']))
    print('动态维度: ' + str(dynamic_stats['total_dims']))
    print('差异: ' + str(dynamic_stats['total_dims'] - fixed_stats['total_dims']))
    
    return fixed_stats, dynamic_stats


if __name__ == "__main__":
    run_ablation()
