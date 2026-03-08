"""
FCRS-v5 完整演示
展示所有核心功能
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np


def demo_basic():
    """基础演示"""
    print('='*60)
    print('1. 基础功能演示')
    print('='*60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=5, vector_dim=10)
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.001
    
    # 运行
    for i in range(200):
        system.step()
    
    # 统计
    stats = system.get_statistics()
    
    print('总步数: ' + str(stats['step']))
    print('表征数: ' + str(stats['pool_size']))
    print('总维度: ' + str(stats['total_dims']))
    print('新维度诞生: ' + str(stats['new_dims_born']))
    print('')


def demo_comparison():
    """对比演示"""
    print('='*60)
    print('2. 固定维度 vs 动态维度')
    print('='*60)
    
    # 固定维度
    fixed = FCRSystem(pool_capacity=5, vector_dim=10)
    fixed.engine.spawn_reuse_threshold = 100  # 禁用新维度
    
    for i in range(200):
        fixed.step()
    
    # 动态维度
    dynamic = FCRSystem(pool_capacity=5, vector_dim=10)
    dynamic.engine.spawn_reuse_threshold = 2
    
    for i in range(200):
        dynamic.step()
    
    print('固定维度: ' + str(fixed.pool.get_total_dims()))
    print('动态维度: ' + str(dynamic.pool.get_total_dims()))
    print('提升: ' + str(dynamic.pool.get_total_dims() - fixed.pool.get_total_dims()) + ' 维')
    print('')


def demo_parameter_sweep():
    """参数扫描演示"""
    print('='*60)
    print('3. 参数敏感性')
    print('='*60)
    
    thresholds = [1, 2, 3, 5]
    
    for th in thresholds:
        system = FCRSystem(pool_capacity=5, vector_dim=10)
        system.engine.spawn_reuse_threshold = th
        system.engine.min_compression_gain = 0.001
        
        for i in range(200):
            system.step()
        
        dims = system.pool.get_total_dims()
        new_dims = len(system.engine.new_dim_history)
        
        print('threshold=' + str(th) + ': ' + str(dims) + '维, ' + str(new_dims) + '新维度')
    print('')


def demo_environment():
    """环境演示"""
    print('='*60)
    print('4. 不同环境')
    print('='*60)
    
    # 创建多个系统
    for name in ['简单', '中等', '复杂']:
        system = FCRSystem(pool_capacity=5, vector_dim=10)
        system.engine.spawn_reuse_threshold = 2
        system.engine.min_compression_gain = 0.001
        
        # 根据环境调整
        if name == '中等':
            system.engine.spawn_reuse_threshold = 1
        
        for i in range(200):
            system.step()
        
        dims = system.pool.get_total_dims()
        print(name + ': ' + str(dims) + '维')
    print('')


def demo_statistics():
    """统计演示"""
    print('='*60)
    print('5. 详细统计')
    print('='*60)
    
    system = FCRSystem(pool_capacity=5, vector_dim=10)
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.001
    
    for i in range(200):
        system.step()
    
    stats = system.get_statistics()
    
    print('步骤统计:')
    for key, value in stats.items():
        if key != 'dim_history':
            print('  ' + key + ': ' + str(value))
    print('')


def main():
    """主函数"""
    print('')
    print('*'*60)
    print('FCRS-v5 完整演示')
    print('*'*60)
    print('')
    
    demo_basic()
    demo_comparison()
    demo_parameter_sweep()
    demo_environment()
    demo_statistics()
    
    print('*'*60)
    print('演示完成!')
    print('*'*60)


if __name__ == "__main__":
    main()
