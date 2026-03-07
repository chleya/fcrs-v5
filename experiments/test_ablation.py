"""
Ablation对比实验 v2
对比"有竞争" vs "无竞争"机制 - 调整版
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def run_ablation():
    print("=" * 60)
    print("Ablation对比实验 v2")
    print("=" * 60)
    
    # 测试1: 有竞争机制（新维度+清理）
    print("\n=== 测试1: 有竞争机制 ===")
    system1 = FCRSystem(pool_capacity=50, vector_dim=10)  # 增大容量
    system1.engine.spawn_reuse_threshold = 30
    system1.engine.min_compression_gain = 0.2
    
    for step in range(1000):
        system1.step()
    
    dims1 = system1.pool.get_total_dims()
    events1 = system1.engine.new_dim_history
    print(f"最终维度: {dims1}")
    print(f"新维度诞生: {len(events1)}")
    
    # 测试2: 禁用新维度诞生（只保留清理）
    print("\n=== 测试2: 禁用新维度诞生 ===")
    system2 = FCRSystem(pool_capacity=50, vector_dim=10)
    system2.engine.spawn_reuse_threshold = 999999  # 禁用
    system2.engine.min_compression_gain = 999999
    
    for step in range(1000):
        system2.step()
    
    dims2 = system2.pool.get_total_dims()
    events2 = system2.engine.new_dim_history
    print(f"最终维度: {dims2}")
    print(f"新维度诞生: {len(events2)}")
    
    # 测试3: 禁用维度清理（只保留新维度诞生）
    print("\n=== 测试3: 禁用维度清理 ===")
    system3 = FCRSystem(pool_capacity=50, vector_dim=10)
    system3.engine.spawn_reuse_threshold = 30
    system3.engine.min_compression_gain = 0.2
    
    # 禁用清理
    system3.engine.prune_low_contrib_dims = lambda: None
    
    for step in range(1000):
        system3.step()
    
    dims3 = system3.pool.get_total_dims()
    events3 = system3.engine.new_dim_history
    print(f"最终维度: {dims3}")
    print(f"新维度诞生: {len(events3)}")
    
    # 对比总结
    print("\n=== 对比总结 ===")
    print(f"{'机制':<25} {'最终维度':<12} {'新维度':<8}")
    print("-" * 50)
    print(f"{'有竞争(完整)':<25} {dims1:<12} {len(events1):<8}")
    print(f"{'无新维度诞生':<25} {dims2:<12} {len(events2):<8}")
    print(f"{'无维度清理':<25} {dims3:<12} {len(events3):<8}")
    
    # 结论
    print("\n=== 结论 ===")
    print(f"有竞争 vs 无新维度: {dims1} vs {dims2} (差异:{abs(dims1-dims2)})")
    print(f"有竞争 vs 无清理: {dims1} vs {dims3} (差异:{abs(dims1-dims3)})")
    
    if dims3 > dims1:
        print("\nv 清理机制有效限制了维度膨胀")
    else:
        print("\nX 需要更多实验验证")
    
    return {
        'with_competition': dims1,
        'without_new_dim': dims2,
        'without_prune': dims3
    }


if __name__ == "__main__":
    run_ablation()
