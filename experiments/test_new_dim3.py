"""
新维度诞生机制测试 - 打印详细信息
"""

import numpy as np
import sys
import os

# 相对导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import FCRSystem


def test_new_dim():
    print("=" * 50)
    print("新维度诞生测试 - 详细")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=3, vector_dim=5)
    
    # 极低阈值
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.0001
    
    print(f"\n参数:")
    print(f"  复用阈值: {system.engine.spawn_reuse_threshold}")
    print(f"  压缩增益阈值: {system.engine.min_compression_gain}")
    
    # 运行300步，每步检查
    print("\n运行300步...")
    for i in range(300):
        system.step()
        
        # 每30步检查top表征
        if i > 0 and i % 30 == 0:
            top_reps = system.pool.get_top_reps(k=1)
            if top_reps:
                r = top_reps[0]
                residuals = np.array(system.recent_residuals) if system.recent_residuals else np.array([])
                
                if len(residuals) > 0 and r.reuse >= system.engine.spawn_reuse_threshold:
                    # 手动计算压缩增益
                    old_v = r.vector.copy()
                    old_len = len(old_v)
                    
                    if residuals.ndim > 1:
                        actual_len = min(residuals.shape[1], old_len)
                        res_mean = np.mean(residuals[:, :actual_len], axis=0)
                        if actual_len < old_len:
                            res_mean = np.pad(res_mean, (0, old_len - actual_len))
                    else:
                        actual_len = min(len(residuals), old_len)
                        res_mean = residuals[:actual_len]
                        if actual_len < old_len:
                            res_mean = np.pad(res_mean, (0, old_len - actual_len))
                    
                    old_err = np.mean(np.abs(res_mean)) + 1e-8
                    
                    pca_dir = res_mean / (np.linalg.norm(res_mean) + 1e-8)
                    new_dim = pca_dir * 0.1 + np.random.normal(0, 0.05, old_len)
                    new_v = np.append(old_v, [np.mean(new_dim)])
                    new_err = np.mean(np.abs(res_mean - new_v[:old_len]))
                    
                    gain = (old_err - new_err) / old_err
                    
                    print(f"  Step {i}: top复用={r.reuse:.0f}, 旧err={old_err:.4f}, 新err={new_err:.4f}, gain={gain:.4f}")
    
    stats = system.get_statistics()
    print(f"\n结果:")
    print(f"  总维度: {stats['total_dims']}")
    print(f"  新维度诞生: {stats['new_dims_born']}")
    print(f"  维度历史: {stats['dim_history']}")


if __name__ == "__main__":
    test_new_dim()
