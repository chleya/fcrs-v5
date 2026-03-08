"""
生成完整可视化图表
包括: 维度演化、适应度曲线、reuse分布
"""

import sys
import os
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_test_with_history(steps=2000):
    """运行测试并记录完整历史"""
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    system.engine.spawn_reuse_threshold = 3
    system.engine.min_compression_gain = 0.0001
    
    dim_history = []
    fitness_history = []
    reuse_history = []
    
    for i in range(steps):
        system.step()
        
        if i % 50 == 0:
            dim_history.append(system.pool.get_total_dims())
            
            # 计算平均适应度
            if system.pool.representations:
                fitnesses = [np.mean(r.fitness_history[-10:]) if r.fitness_history else 0 
                           for r in system.pool.representations]
                avg_fitness = np.mean(fitnesses)
                fitness_history.append(avg_fitness)
                
                # reuse分布
                reuses = [r.reuse for r in system.pool.representations]
                avg_reuse = np.mean(reuses)
                reuse_history.append(avg_reuse)
    
    return {
        'dim_history': dim_history,
        'fitness_history': fitness_history,
        'reuse_history': reuse_history,
        'final_dims': system.pool.get_total_dims(),
        'new_dims': len(system.engine.new_dim_history)
    }


def visualize_all():
    """生成所有图表"""
    # 设置工作目录
    os.chdir('F:/fcrs-v5')
    
    print('运行测试...')
    result = run_test_with_history(2000)
    
    # 图1: 维度演化
    plt.figure(figsize=(10, 6))
    plt.plot(result['dim_history'], 'b-', linewidth=2)
    plt.xlabel('Time (x50 steps)')
    plt.ylabel('Total Dimensions')
    plt.title('Dimension Evolution Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/fig_dimension_evolution.png', dpi=150)
    plt.close()
    print('Saved: fig_dimension_evolution.png')
    
    # 图2: 适应度变化
    plt.figure(figsize=(10, 6))
    plt.plot(result['fitness_history'], 'g-', linewidth=2)
    plt.xlabel('Time (x50 steps)')
    plt.ylabel('Average Fitness')
    plt.title('Fitness Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/fig_fitness.png', dpi=150)
    plt.close()
    print('Saved: fig_fitness.png')
    
    # 图3: Reuse分布
    plt.figure(figsize=(10, 6))
    plt.plot(result['reuse_history'], 'r-', linewidth=2)
    plt.xlabel('Time (x50 steps)')
    plt.ylabel('Average Reuse')
    plt.title('Reuse Frequency Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/fig_reuse.png', dpi=150)
    plt.close()
    print('Saved: fig_reuse.png')
    
    # 图4: 对比图
    plt.figure(figsize=(10, 6))
    steps = range(len(result['dim_history']))
    plt.plot(steps, result['dim_history'], 'b-', label='Dimensions', linewidth=2)
    plt.plot(steps, result['reuse_history'], 'r--', label='Reuse', linewidth=2)
    plt.xlabel('Time (x50 steps)')
    plt.ylabel('Value')
    plt.title('Dimensions vs Reuse')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/fig_dim_vs_reuse.png', dpi=150)
    plt.close()
    print('Saved: fig_dim_vs_reuse.png')
    
    print('')
    print('Final dimensions: ' + str(result['final_dims']))
    print('New dimensions born: ' + str(result['new_dims']))


if __name__ == "__main__":
    visualize_all()
