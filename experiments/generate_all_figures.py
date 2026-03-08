"""
FCRS完整可视化套件
生成所有实验图表
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_full_test(steps=1000):
    """运行完整测试并记录数据"""
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.001
    
    # 记录数据
    dim_history = []
    fitness_history = []
    reuse_history = []
    new_dim_history = []
    
    for i in range(steps):
        system.step()
        
        if i % 20 == 0:
            dim_history.append(system.pool.get_total_dims())
            
            # 适应度
            if system.pool.representations:
                fitness = [np.mean(r.fitness_history[-10:]) if r.fitness_history else 0 
                          for r in system.pool.representations]
                fitness_history.append(np.mean(fitness))
                
                # 复用
                reuse = [r.reuse for r in system.pool.representations]
                reuse_history.append(np.mean(reuse))
            
            # 新维度
            new_dim_history.append(len(system.engine.new_dim_history))
    
    return {
        'dim': dim_history,
        'fitness': fitness_history,
        'reuse': reuse_history,
        'new_dim': new_dim_history,
        'final_dims': system.pool.get_total_dims()
    }


def create_figures():
    """生成所有图表"""
    print('Running test...')
    data = run_full_test(1000)
    
    os = __import__('os')
    os.chdir('F:/fcrs-v5/paper')
    
    # 1. 维度演化
    plt.figure(figsize=(10, 6))
    plt.plot(data['dim'], 'b-', linewidth=2)
    plt.xlabel('Time (x20 steps)')
    plt.ylabel('Total Dimensions')
    plt.title('Dimension Evolution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig1_dimension.png', dpi=150)
    plt.close()
    print('Saved: fig1_dimension.png')
    
    # 2. 适应度
    plt.figure(figsize=(10, 6))
    plt.plot(data['fitness'], 'g-', linewidth=2)
    plt.xlabel('Time (x20 steps)')
    plt.ylabel('Average Fitness')
    plt.title('Fitness Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig2_fitness.png', dpi=150)
    plt.close()
    print('Saved: fig2_fitness.png')
    
    # 3. 复用
    plt.figure(figsize=(10, 6))
    plt.plot(data['reuse'], 'r-', linewidth=2)
    plt.xlabel('Time (x20 steps)')
    plt.ylabel('Average Reuse')
    plt.title('Reuse Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_reuse.png', dpi=150)
    plt.close()
    print('Saved: fig3_reuse.png')
    
    # 4. 新维度诞生
    plt.figure(figsize=(10, 6))
    plt.plot(data['new_dim'], 'm-', linewidth=2)
    plt.xlabel('Time (x20 steps)')
    plt.ylabel('Cumulative New Dimensions')
    plt.title('New Dimension Birth')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_new_dim.png', dpi=150)
    plt.close()
    print('Saved: fig4_new_dim.png')
    
    # 5. 综合对比
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data['dim'], 'b-')
    plt.title('Dimensions')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(data['fitness'], 'g-')
    plt.title('Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(data['reuse'], 'r-')
    plt.title('Reuse')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(data['new_dim'], 'm-')
    plt.title('New Dimensions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig5_overview.png', dpi=150)
    plt.close()
    print('Saved: fig5_overview.png')
    
    print('\nAll figures saved!')
    print('Final dimensions: ' + str(data['final_dims']))


if __name__ == "__main__":
    create_figures()
