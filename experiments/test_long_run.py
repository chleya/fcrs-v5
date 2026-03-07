"""
长时运行测试 - 2000步
验证维度是否会趋于稳定
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def run_long_test():
    print("=" * 60)
    print("长时运行测试 - 2000步")
    print("=" * 60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    
    # 参数
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.001
    
    # 记录数据
    dim_history = []
    pool_size_history = []
    new_dim_events = []
    
    print("\n运行2000步...")
    for step in range(2000):
        system.step()
        
        if step % 100 == 0:
            dim_history.append(system.pool.get_total_dims())
            pool_size_history.append(len(system.pool))
            
            # 记录这个阶段的新维度事件
            events_in_period = [e for e in system.engine.new_dim_history 
                              if e['step'] == step or (step > 0 and 'step' in e)]
            if events_in_period:
                new_dim_events.extend(events_in_period)
    
    stats = system.get_statistics()
    
    # 分析
    print("\n=== 维度演化分析 ===")
    print(f"初始维度: {dim_history[0]}")
    print(f"最终维度: {dim_history[-1]}")
    print(f"最大维度: {max(dim_history)}")
    print(f"维度变化: {max(dim_history) - min(dim_history)}")
    
    # 检查是否趋于稳定
    last_10 = dim_history[-10:]
    stability = max(last_10) - min(last_10)
    print(f"最后10个采样维度差异: {stability}")
    
    if stability <= 10:
        print("v 维度趋于稳定")
    else:
        print("X 维度仍在波动")
    
    # 新维度事件
    print(f"\n=== 新维度事件 ===")
    print(f"总共诞生: {stats['new_dims_born']} 个新维度")
    
    # 每个表征的最终状态
    print(f"\n=== 表征最终状态 ===")
    print(f"{'ID':<6} {'维度':<8} {'复用':<10} {'贡献':<15}")
    print("-" * 45)
    
    for r in system.pool.representations:
        contrib = np.sum(r.dim_contrib)
        print(f"{r.id:<6} {len(r.vector):<8} {r.reuse:<10.0f} {contrib:<15.2f}")
    
    return {
        'dim_history': dim_history,
        'final_dims': dim_history[-1],
        'stability': stability,
        'new_dims': stats['new_dims_born']
    }


if __name__ == "__main__":
    result = run_long_test()
