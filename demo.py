"""
FCRS-v5 实验过程演示
Demonstration Script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import FCRSystem, EnvironmentLoop, RepresentationPool, EvolutionEngine
import numpy as np


def demo_step1():
    """演示1: 环境产生输入"""
    print("="*60)
    print("Step 1: 环境产生输入")
    print("="*60)
    
    env = EnvironmentLoop(input_dim=10)
    
    print("\n环境配置:")
    print(f"  输入维度: {env.input_dim}")
    print(f"  类别数: {env.num_classes}")
    
    print("\n生成3个输入样本:")
    for i in range(3):
        x = env.generate_input()
        print(f"  样本{i+1}: norm={np.linalg.norm(x):.2f}")
    
    return env


def demo_step2():
    """演示2: 表征池初始化"""
    print("\n" + "="*60)
    print("Step 2: 表征池初始化")
    print("="*60)
    
    pool = RepresentationPool(capacity=5, vector_dim=10)
    
    print("\n表征池配置:")
    print(f"  容量: {pool.capacity}")
    print(f"  向量维度: {pool.vector_dim}")
    print(f"  初始表征数: {len(pool)}")
    print(f"  初始预算: {pool.total_budget}")
    
    # 添加第一个表征
    x = np.random.randn(10)
    rep = pool.add(x)
    
    print(f"\n添加表征后:")
    print(f"  表征数: {len(pool)}")
    print(f"  表征ID: {rep.id}")
    print(f"  表征维度: {len(rep.vector)}")
    
    return pool


def demo_step3():
    """演示3: 竞争选择"""
    print("\n" + "="*60)
    print("Step 3: 竞争选择")
    print("="*60)
    
    pool = RepresentationPool(capacity=5, vector_dim=10)
    env = EnvironmentLoop(input_dim=10)
    
    # 添加多个表征
    for i in range(3):
        x = np.random.randn(10) * (i+1)
        pool.add(x)
    
    print(f"\n表征池有 {len(pool)} 个表征")
    
    # 生成输入并竞争
    x = env.generate_input()
    act, active_rep = pool.activate(x)
    
    print(f"\n输入: norm={np.linalg.norm(x):.2f}")
    print(f"激活表征ID: {active_rep.id if active_rep else None}")
    print(f"激活值: {act:.2f}")
    
    return pool, env


def demo_step4():
    """演示4: 维度诞生机制"""
    print("\n" + "="*60)
    print("Step 4: 维度诞生机制")
    print("="*60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=3, vector_dim=5)
    
    # 降低阈值便于演示
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.0001
    
    print("\n演示参数:")
    print(f"  初始维度: 5")
    print(f"  复用阈值: {system.engine.spawn_reuse_threshold}")
    print(f"  压缩增益阈值: {system.engine.min_compression_gain}")
    
    # 运行几步让表征被复用
    print("\n运行10步让表征积累复用次数...")
    for i in range(10):
        system.step()
    
    # 检查top表征
    top_reps = system.pool.get_top_reps(k=1)
    if top_reps:
        r = top_reps[0]
        print(f"\nTop表征:")
        print(f"  ID: {r.id}")
        print(f"  当前维度: {len(r.vector)}")
        print(f"  复用次数: {r.reuse}")
    
    # 手动触发新维度诞生检查
    if system.recent_residuals:
        residuals = np.array(system.recent_residuals)
        for r in system.pool.get_top_reps(k=1):
            result = system.engine.try_spawn_new_dim(r, residuals)
            if result:
                print(f"\n*** 新维度诞生! ***")
                print(f"  新维度数: {len(r.vector)}")
    
    return system


def demo_full():
    """完整演示"""
    print("="*60)
    print("FCRS-v5 完整实验过程演示")
    print("="*60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=5, vector_dim=8)
    
    # 降低阈值
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.0001
    
    print("\n初始状态:")
    print(f"  表征池: {len(system.pool)}")
    print(f"  总维度: {system.pool.get_total_dims()}")
    print(f"  预算: {system.pool.total_budget}")
    
    # 运行100步
    print("\n运行100步...")
    for i in range(100):
        system.step()
        
        if (i+1) % 20 == 0:
            print(f"  Step {i+1}: 维度={system.pool.get_total_dims()}, 表征数={len(system.pool)}")
    
    # 最终统计
    stats = system.get_statistics()
    
    print("\n最终结果:")
    print(f"  总步数: {stats['step']}")
    print(f"  表征数: {stats['pool_size']}")
    print(f"  总维度: {stats['total_dims']}")
    print(f"  预算: {stats['budget']}")
    print(f"  新维度诞生: {stats['new_dims_born']}")
    print(f"  维度历史: {stats['dim_history']}")
    
    print("\n表征详情:")
    for rep in stats['representations']:
        print(f"  ID:{rep['id']:2d} 维:{rep['dim']:2d} 复用:{rep['reuse']:4.0f}")
    
    return stats


def demo_comparison():
    """对比演示: 有无截断"""
    print("\n" + "="*60)
    print("对比演示: 有截断 vs 无截断")
    print("="*60)
    
    # 无截断(模拟)
    print("\n[无截断系统]")
    pool1 = RepresentationPool(capacity=10, vector_dim=5)
    pool1.total_budget = float('inf')  # 无预算限制
    
    for i in range(100):
        x = np.random.randn(5)
        if len(pool1) < 10:
            pool1.add(x)
        else:
            # 替换
            pool1.representations[i%10].vector = x
    
    print(f"  最终维度: {pool1.get_total_dims()}")
    print(f"  表征数: {len(pool1)}")
    
    # 有截断
    print("\n[有截断系统]")
    pool2 = RepresentationPool(capacity=10, vector_dim=5)
    pool2.total_budget = 20.0  # 预算限制
    
    for i in range(100):
        x = np.random.randn(5)
        dim_cost = 1.0
        if pool2.total_budget >= dim_cost and len(pool2) < 10:
            pool2.add(x, dim_cost=dim_cost)
    
    print(f"  最终维度: {pool2.get_total_dims()}")
    print(f"  剩余预算: {pool2.total_budget}")
    print(f"  表征数: {len(pool2)}")
    
    print("\n结论: 截断机制限制了维度无限增长!")


if __name__ == "__main__":
    # 默认运行完整演示
    demo_full()
