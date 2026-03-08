"""
2000步长期测试
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np


def run_long_test(steps=2000):
    print(f'Running {steps} steps test...')
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.0001
    
    # 运行
    for i in range(steps):
        system.step()
        if (i+1) % 500 == 0:
            print(f'Step {i+1}: dims={system.pool.get_total_dims()}, reps={len(system.pool)}, budget={system.pool.total_budget:.1f}')
    
    stats = system.get_statistics()
    
    print(f'\n=== {steps}步结果 ===')
    print(f'总步数: {stats["step"]}')
    print(f'表征数: {stats["pool_size"]}')
    print(f'总维度: {stats["total_dims"]}')
    print(f'剩余预算: {stats["budget"]}')
    print(f'新维度诞生: {stats["new_dims_born"]}')
    print(f'维度历史: {stats["dim_history"]}')
    
    return stats


if __name__ == "__main__":
    run_long_test(2000)
