"""
多组环境实验 + 参数敏感性分析
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem, EnvironmentLoop
import numpy as np


class MultiModeEnv(EnvironmentLoop):
    """多模式环境"""
    def __init__(self, input_dim=10, n_modes=5):
        super().__init__(input_dim)
        self.n_modes = n_modes
        self.class_centers = {
            i: np.random.randn(input_dim) * (1 + i*0.3) 
            for i in range(n_modes)
        }


class NoisyEnv(EnvironmentLoop):
    """噪声环境"""
    def __init__(self, input_dim=10, noise=0.5):
        super().__init__(input_dim)
        self.noise = noise
    
    def generate_input(self):
        x = super().generate_input()
        return x + np.random.randn(self.input_dim) * self.noise


def test_environments():
    """多组环境实验"""
    print('='*60)
    print('多组环境实验')
    print('='*60)
    
    configs = [
        ('简单(3类)', EnvironmentLoop(input_dim=10)),
        ('中等(5类)', MultiModeEnv(input_dim=10, n_modes=5)),
        ('复杂(10类)', MultiModeEnv(input_dim=10, n_modes=10)),
        ('低噪声(0.3)', NoisyEnv(input_dim=10, noise=0.3)),
        ('高噪声(0.8)', NoisyEnv(input_dim=10, noise=0.8)),
    ]
    
    results = {}
    
    for name, env in configs:
        print('\n[' + name + ']')
        
        system = FCRSystem(pool_capacity=5, vector_dim=10)
        system.env = env
        system.engine.spawn_reuse_threshold = 2
        system.engine.min_compression_gain = 0.00001
        
        for i in range(500):
            system.step()
        
        dims = system.pool.get_total_dims()
        new_dims = len(system.engine.new_dim_history)
        
        print('  维度: ' + str(dims) + ', 新维度: ' + str(new_dims))
        
        results[name] = {'dims': dims, 'new_dims': new_dims}
    
    print('\n' + '='*60)
    print('环境对比结果')
    print('='*60)
    for name, r in results.items():
        print(name + ': ' + str(r['dims']) + ' 维, ' + str(r['new_dims']) + ' 新维度')
    
    return results


def test_parameter_sensitivity():
    """参数敏感性分析"""
    print('\n' + '='*60)
    print('参数敏感性分析')
    print('='*60)
    
    # 扫描 spawn_reuse_threshold
    print('\n[spawn_reuse_threshold 扫描]')
    thresholds = [1, 2, 3, 5, 10]
    
    threshold_results = {}
    for th in thresholds:
        system = FCRSystem(pool_capacity=5, vector_dim=10)
        system.engine.spawn_reuse_threshold = th
        system.engine.min_compression_gain = 0.00001
        
        for i in range(500):
            system.step()
        
        dims = system.pool.get_total_dims()
        new_dims = len(system.engine.new_dim_history)
        
        print('  threshold=' + str(th) + ': ' + str(dims) + ' 维, ' + str(new_dims) + ' 新维度')
        threshold_results[th] = {'dims': dims, 'new_dims': new_dims}
    
    # 扫描 min_compression_gain
    print('\n[min_compression_gain 扫描]')
    gains = [0.0001, 0.001, 0.01, 0.05, 0.1]
    
    gain_results = {}
    for g in gains:
        system = FCRSystem(pool_capacity=5, vector_dim=10)
        system.engine.spawn_reuse_threshold = 2
        system.engine.min_compression_gain = g
        
        for i in range(500):
            system.step()
        
        dims = system.pool.get_total_dims()
        new_dims = len(system.engine.new_dim_history)
        
        print('  gain=' + str(g) + ': ' + str(dims) + ' 维, ' + str(new_dims) + ' 新维度')
        gain_results[g] = {'dims': dims, 'new_dims': new_dims}
    
    return threshold_results, gain_results


if __name__ == "__main__":
    # 环境实验
    env_results = test_environments()
    
    # 参数敏感性
    th_results, gain_results = test_parameter_sensitivity()
    
    print('\n' + '='*60)
    print('实验完成')
    print('='*60)
