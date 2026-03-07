"""
新维度诞生机制测试 - 极低阈值
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def test_new_dim():
    print("=" * 50)
    print("新维度诞生测试 - 极低阈值")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=5, vector_dim=5)
    
    # 极低阈值
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.0001  # 极低
    
    print(f"\n参数:")
    print(f"  复用阈值: {system.engine.spawn_reuse_threshold}")
    print(f"  压缩增益阈值: {system.engine.min_compression_gain}")
    
    # 运行500步
    print("\n运行500步...")
    for i in range(500):
        system.step()
    
    stats = system.get_statistics()
    print(f"\n结果:")
    print(f"  总维度: {stats['total_dims']}")
    print(f"  新维度诞生: {stats['new_dims_born']}")
    print(f"  维度历史: {stats['dim_history']}")
    
    # 打印每个表征的详情
    print(f"\n表征详情:")
    for r in stats['representations']:
        print(f"  ID:{r['id']} 维:{r['dim']} 复用:{r['reuse']:.0f}")


if __name__ == "__main__":
    test_new_dim()
