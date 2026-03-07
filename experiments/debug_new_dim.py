"""
新维度诞生机制测试 - 带调试信息
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def debug_try_spawn():
    print("=" * 50)
    print("调试: 新维度诞生条件")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=3, vector_dim=5)
    
    # 调整参数 - 更激进
    system.engine.spawn_reuse_threshold = 5
    system.engine.min_compression_gain = 0.001
    
    # 运行一些步数让表征发展
    print("\n预热100步...")
    for _ in range(100):
        system.step()
    
    # 检查top表征
    top_reps = system.pool.get_top_reps(k=1)
    if not top_reps:
        print("没有表征")
        return
    
    r = top_reps[0]
    print(f"\nTop表征: ID={r.id}, 复用={r.reuse}, 向量维度={len(r.vector)}")
    
    # 手动检查条件
    residuals = np.array(system.recent_residuals)
    print(f"残差数组形状: {residuals.shape}")
    
    if len(residuals) == 0:
        print("没有残差数据")
        return
    
    old_v = r.vector.copy()
    old_len = len(old_v)
    
    # 计算残差均值
    if residuals.ndim > 1:
        residual_mean = np.mean(residuals[:, :old_len], axis=0)
    else:
        residual_mean = residuals[:old_len]
    
    old_error = np.mean(np.abs(residual_mean)) + 1e-8
    print(f"旧误差: {old_error:.6f}")
    
    # 模拟新维度
    pca_dir = residual_mean / (np.linalg.norm(residual_mean) + 1e-8)
    new_dim = pca_dir * 0.1 + np.random.normal(0, 0.05, old_len)
    new_v = np.append(old_v, [np.mean(new_dim)])
    
    # 新误差
    new_error = np.mean(np.abs(residual_mean - new_v[:old_len]))
    print(f"新误差: {new_error:.6f}")
    
    # 压缩增益
    gain = (old_error - new_error) / old_error
    print(f"压缩增益: {gain:.6f}")
    print(f"阈值: {system.engine.min_compression_gain}")
    
    if gain > system.engine.min_compression_gain:
        print("\nv 应该诞生新维度!")
    else:
        print("\nX 不会诞生新维度")


if __name__ == "__main__":
    debug_try_spawn()
