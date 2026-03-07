"""
新维度诞生机制测试
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem, EnvironmentLoop, RepresentationPool, EvolutionEngine


def test_new_dim():
    print("=" * 50)
    print("新维度诞生测试")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=5, vector_dim=5)
    
    # 调整参数
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.01
    
    print(f"\n参数:")
    print(f"  复用阈值: {system.engine.spawn_reuse_threshold}")
    print(f"  压缩增益阈值: {system.engine.min_compression_gain}")
    
    # 运行200步
    print("\n运行200步...")
    for i in range(200):
        system.step()
        
        if i % 20 == 0:
            # 检查top表征
            top_reps = system.pool.get_top_reps(k=2)
            if top_reps:
                r = top_reps[0]
                residuals = np.array(system.recent_residuals) if system.recent_residuals else np.array([0])
                print(f"  Step {i}: top复用={r.reuse:.0f}, 残差形状={residuals.shape}")
    
    stats = system.get_statistics()
    print(f"\n结果:")
    print(f"  总维度: {stats['total_dims']}")
    print(f"  新维度诞生: {stats['new_dims_born']}")
    print(f"  维度历史: {stats['dim_history']}")


if __name__ == "__main__":
    test_new_dim()
