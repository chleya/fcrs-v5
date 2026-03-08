"""
2000步长期测试
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np


def run_long_test(steps=5000):
    print('Running 5000 steps test...')
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.0001
    
    # 运行
    for i in range(steps):
        system.step()
        if (i+1) % 1000 == 0:
            print('Step ' + str(i+1) + ': dims=' + str(system.pool.get_total_dims()))
    
    stats = system.get_statistics()
    
    print('')
    print('=== 5000步结果 ===')
    print('总维度: ' + str(stats['total_dims']))
    print('新维度诞生: ' + str(stats['new_dims_born']))
    print('表征数: ' + str(stats['pool_size']))
    print('预算: ' + str(stats['budget']))
    
    return stats


if __name__ == "__main__":
    run_long_test(5000)
