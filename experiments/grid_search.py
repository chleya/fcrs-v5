"""
FCRS参数系统性搜索
使用网格搜索找最优参数
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from fcrs_architecture import create_fcrs, VectorRepresentation
from fcrs_architecture import StructuredEnvironment


def grid_search():
    """网格搜索"""
    print('='*60)
    print('Grid Search for Optimal Parameters')
    print('='*60)
    
    # 参数网格
    lr_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    pool_sizes = [3, 5, 10]
    n_classes_values = [3, 5, 10]
    
    results = []
    
    total = len(lr_values) * len(pool_sizes) * len(n_classes_values)
    count = 0
    
    for lr in lr_values:
        for pool_size in pool_sizes:
            for n_classes in n_classes_values:
                count += 1
                print(f'\n[{count}/{total}] lr={lr}, pool={pool_size}, classes={n_classes}')
                
                # 创建环境
                env = StructuredEnvironment(input_dim=10, n_classes=n_classes)
                
                # 创建系统
                fcrs = create_fcrs(
                    env_type='structured',
                    pool_capacity=pool_size,
                    input_dim=10,
                    n_classes=n_classes,
                    lr=lr
                )
                
                # 初始化表征
                for _ in range(3):
                    rep = VectorRepresentation(env.generate_input())
                    fcrs.pool.add(rep)
                
                # 运行
                errors = []
                for _ in range(500):
                    fcrs.step()
                    if len(fcrs.errors) >= 100:
                        errors.append(fcrs.get_avg_error())
                
                avg_error = np.mean(errors[-20:]) if errors else float('inf')
                
                print(f'  Error: {avg_error:.4f}')
                
                results.append({
                    'lr': lr,
                    'pool_size': pool_size,
                    'n_classes': n_classes,
                    'error': avg_error
                })
    
    # 排序
    results.sort(key=lambda x: x['error'])
    
    print('\n' + '='*60)
    print('Top 5 Configurations')
    print('='*60)
    
    for i, r in enumerate(results[:5], 1):
        print(f'{i}. lr={r["lr"]}, pool={r["pool_size"]}, classes={r["n_classes"]} -> error={r["error"]:.4f}')
    
    # 最佳配置
    best = results[0]
    print(f'\nBest: lr={best["lr"]}, pool={best["pool_size"]}, classes={best["n_classes"]}')
    print(f'Error: {best["error"]:.4f}')
    
    return results


if __name__ == "__main__":
    results = grid_search()
