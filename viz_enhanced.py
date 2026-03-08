"""
FCRS可视化套件
增强可视化：学习曲线、误差分布、表征演化
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fcrs_architecture import create_fcrs, VectorRepresentation


def run_full_visualization(steps=1000):
    """完整可视化"""
    from fcrs_architecture import StructuredEnvironment
    
    # 创建系统
    env = StructuredEnvironment(input_dim=10, n_classes=5)
    fcrs = create_fcrs(env_type='structured', pool_capacity=5, input_dim=10, n_classes=5, lr=0.01)
    
    # 初始化表征
    for _ in range(3):
        rep = VectorRepresentation(env.generate_input())
        fcrs.pool.add(rep)
    
    # 运行并记录
    errors = []
    fitnesses = []
    
    for i in range(steps):
        fcrs.step()
        
        if i % 10 == 0:
            errors.append(fcrs.get_avg_error())
            
            # 记录适应度
            if fcrs.pool.representations:
                fit = [r.get_fitness() for r in fcrs.pool.representations]
                fitnesses.append(np.mean(fit))
    
    return errors, fitnesses, fcrs


def plot_all():
    """生成所有图表"""
    print('Running visualization...')
    errors, fitnesses, fcrs = run_full_visualization(1000)
    
    import os
    os.makedirs('paper', exist_ok=True)
    
    # 1. 学习曲线
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', linewidth=2)
    plt.xlabel('Steps (x10)')
    plt.ylabel('Average Error')
    plt.title('FCRS Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/viz_learning_curve.png', dpi=150)
    plt.close()
    print('Saved: viz_learning_curve.png')
    
    # 2. 适应度变化
    plt.figure(figsize=(10, 6))
    plt.plot(fitnesses, 'g-', linewidth=2)
    plt.xlabel('Steps (x10)')
    plt.ylabel('Average Fitness')
    plt.title('Fitness Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/viz_fitness.png', dpi=150)
    plt.close()
    print('Saved: viz_fitness.png')
    
    # 3. 误差分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(fcrs.errors[-500:], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution (last 500 steps)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/viz_error_dist.png', dpi=150)
    plt.close()
    print('Saved: viz_error_dist.png')
    
    # 4. 综合面板
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(errors, 'b-')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(fitnesses, 'g-')
    axes[0, 1].set_title('Fitness')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Fitness')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(fcrs.errors[-500:], bins=30, alpha=0.7)
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].set_xlabel('Error')
    axes[1, 0].set_ylabel('Frequency')
    
    # 表征分布
    if fcrs.pool.representations:
        vectors = np.array([r.get_vector() for r in fcrs.pool.representations])
        axes[1, 1].scatter(vectors[:, 0], vectors[:, 1], s=100, alpha=0.7)
        axes[1, 1].set_title('Representation Space (first 2 dims)')
        axes[1, 1].set_xlabel('Dim 1')
        axes[1, 1].set_ylabel('Dim 2')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('paper/viz_overview.png', dpi=150)
    plt.close()
    print('Saved: viz_overview.png')
    
    print('\nFinal error:', round(errors[-1], 4))
    print('All visualizations saved!')


if __name__ == "__main__":
    plot_all()
