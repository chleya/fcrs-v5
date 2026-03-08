"""
复杂环境测试
测试在不同环境复杂度下的系统表现
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem, EnvironmentLoop
import numpy as np


class ComplexEnv(EnvironmentLoop):
    """复杂环境 - 多模式混合"""
    
    def __init__(self, input_dim=10, n_classes=5):
        super().__init__(input_dim)
        self.n_classes = n_classes
        # 更多类别中心
        self.class_centers = {
            i: np.random.randn(input_dim) * (1 + i*0.5) 
            for i in range(n_classes)
        }


class NoisyEnv(EnvironmentLoop):
    """噪声环境"""
    
    def __init__(self, input_dim=10, noise_level=0.5):
        super().__init__(input_dim)
        self.noise_level = noise_level
    
    def generate_input(self):
        x = super().generate_input()
        # 添加噪声
        x = x + np.random.randn(self.input_dim) * self.noise_level
        return x


def test_complex_env():
    print('='*60)
    print('复杂环境测试')
    print('='*60)
    
    # 简单环境
    print('\n[简单环境: 3类]')
    simple = FCRSystem(pool_capacity=5, vector_dim=10)
    simple.engine.spawn_reuse_threshold = 2
    simple.engine.min_compression_gain = 0.00001
    
    for i in range(500):
        simple.step()
    
    print('  最终维度: ' + str(simple.pool.get_total_dims()))
    print('  新维度诞生: ' + str(len(simple.engine.new_dim_history)))
    
    # 复杂环境
    print('\n[复杂环境: 5类]')
    complex_sys = FCRSystem(pool_capacity=5, vector_dim=10)
    complex_sys.env = ComplexEnv(input_dim=10, n_classes=5)
    complex_sys.engine.spawn_reuse_threshold = 2
    complex_sys.engine.min_compression_gain = 0.00001
    
    for i in range(500):
        complex_sys.step()
    
    print('  最终维度: ' + str(complex_sys.pool.get_total_dims()))
    print('  新维度诞生: ' + str(len(complex_sys.engine.new_dim_history)))
    
    # 噪声环境
    print('\n[噪声环境: 噪声=0.5]')
    noisy = FCRSystem(pool_capacity=5, vector_dim=10)
    noisy.env = NoisyEnv(input_dim=10, noise_level=0.5)
    noisy.engine.spawn_reuse_threshold = 2
    noisy.engine.min_compression_gain = 0.00001
    
    for i in range(500):
        noisy.step()
    
    print('  最终维度: ' + str(noisy.pool.get_total_dims()))
    print('  新维度诞生: ' + str(len(noisy.engine.new_dim_history)))
    
    # 高噪声环境
    print('\n[高噪声环境: 噪声=1.0]')
    high_noise = FCRSystem(pool_capacity=5, vector_dim=10)
    high_noise.env = NoisyEnv(input_dim=10, noise_level=1.0)
    high_noise.engine.spawn_reuse_threshold = 2
    high_noise.engine.min_compression_gain = 0.00001
    
    for i in range(500):
        high_noise.step()
    
    print('  最终维度: ' + str(high_noise.pool.get_total_dims()))
    print('  新维度诞生: ' + str(len(high_noise.engine.new_dim_history)))
    
    print('\n' + '='*60)
    print('对比结果')
    print('='*60)
    print('简单环境: ' + str(simple.pool.get_total_dims()))
    print('复杂环境: ' + str(complex_sys.pool.get_total_dims()))
    print('低噪声: ' + str(noisy.pool.get_total_dims()))
    print('高噪声: ' + str(high_noise.pool.get_total_dims()))


if __name__ == "__main__":
    test_complex_env()
