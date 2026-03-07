"""
维度演化可视化 - 纯文本版
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def run_visualization():
    print("=" * 60)
    print("维度演化可视化")
    print("=" * 60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    
    # 参数
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.001
    
    # 记录数据
    dim_history = []
    
    print("\n运行1000步...")
    for step in range(1000):
        system.step()
        
        if step % 50 == 0:
            dim_history.append(system.pool.get_total_dims())
    
    stats = system.get_statistics()
    
    # 打印维度演化
    print("\n=== 维度演化 ===")
    print(f"步数: {'->'.join([str(i*50) for i in range(len(dim_history))])}")
    print(f"维度: {'->'.join(map(str, dim_history))}")
    
    # 打印新维度事件
    print("\n=== 新维度诞生事件 ===")
    if system.engine.new_dim_history:
        for i, e in enumerate(system.engine.new_dim_history):
            print(f"  事件{i+1}: 压缩增益={e['gain']:.3f}, 新维度={e['new_dim']}")
    else:
        print("  无")
    
    # 打印表征详情
    print("\n=== 表征详情 ===")
    print(f"{'ID':<6} {'维度':<8} {'复用':<10} {'贡献':<15}")
    print("-" * 45)
    
    for r in system.pool.representations:
        contrib = np.sum(r.dim_contrib)
        print(f"{r.id:<6} {len(r.vector):<8} {r.reuse:<10.0f} {contrib:<15.2f}")
    
    # 维度分布图（文本）
    print("\n=== 维度分布 ===")
    dims = [len(r.vector) for r in system.pool.representations]
    for d in range(10, 16):
        count = dims.count(d)
        bar = "█" * count
        print(f"  {d:2d}维: {bar} ({count}个)")
    
    print(f"\n=== 统计 ===")
    print(f"总维度: {stats['total_dims']}")
    print(f"新维度诞生: {stats['new_dims_born']}")
    print(f"池大小: {stats['pool_size']}")
    
    return dim_history, stats


if __name__ == "__main__":
    run_visualization()
