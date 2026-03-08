"""
FCRS可视化
"""

import os
os.makedirs('paper', exist_ok=True)

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fcrs import FCRS


def visualize():
    """生成可视化"""
    from core import EnvironmentLoop
    env = EnvironmentLoop(input_dim=10)
    
    # 运行
    fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)
    
    errors = []
    for _ in range(500):
        x = env.generate_input()
        fcrs.step(x)
        errors.append(fcrs.get_avg_error())
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Average Error')
    plt.title('FCRS Learning Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('paper/fcrs_learning_curve.png', dpi=150)
    plt.close()
    
    print('Saved: paper/fcrs_learning_curve.png')
    print('Final error:', round(errors[-1], 4))


if __name__ == "__main__":
    visualize()
