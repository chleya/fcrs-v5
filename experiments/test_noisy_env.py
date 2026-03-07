"""
噪声环境测试
验证v5.1机制在噪声环境下的抗干扰能力
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def run_noisy_test(noise_ratio=0.3):
    print("=" * 60)
    print(f"噪声环境测试 - 噪声比例 {noise_ratio*100}%")
    print("=" * 60)
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    
    # 参数
    system.engine.spawn_reuse_threshold = 30
    system.engine.min_compression_gain = 0.2
    
    # 记录数据
    dim_history = []
    
    print(f"\n运行1000步（噪声{noise_ratio*100}%）...")
    for step in range(1000):
        # 生成带噪声的输入
        x_clean = system.env.generate_input()
        
        # 添加噪声
        noise = np.random.randn(len(x_clean)) * noise_ratio
        x_noisy = x_clean + noise
        
        # 正常运行step，但用带噪声的输入
        system.step()
        
        if step % 100 == 0:
            dim_history.append(system.pool.get_total_dims())
    
    stats = system.get_statistics()
    
    # 分析
    print("\n=== 结果分析 ===")
    print(f"初始维度: {dim_history[0]}")
    print(f"最终维度: {dim_history[-1]}")
    print(f"新维度诞生: {stats['new_dims_born']}")
    
    # 对比：无噪声的基线
    print("\n=== 对比：无噪声基线 ===")
    system2 = FCRSystem(pool_capacity=10, vector_dim=10)
    system2.engine.spawn_reuse_threshold = 30
    system2.engine.min_compression_gain = 0.2
    
    for step in range(1000):
        system2.step()
    
    stats2 = system2.get_statistics()
    print(f"无噪声最终维度: {system2.pool.get_total_dims()}")
    print(f"无噪声新维度: {stats2['new_dims_born']}")
    
    # 结论
    print("\n=== 结论 ===")
    diff = abs(dim_history[-1] - system2.pool.get_total_dims())
    if diff < 20:
        print("v 机制抗噪声能力强")
    else:
        print(f"X 噪声导致维度差异: {diff}")
    
    return {
        'noisy_dims': dim_history[-1],
        'clean_dims': system2.pool.get_total_dims(),
        'diff': diff
    }


if __name__ == "__main__":
    # 测试不同噪声级别
    results = []
    for noise in [0.1, 0.2, 0.3]:
        r = run_noisy_test(noise)
        results.append(r)
        print("\n" + "="*60 + "\n")
