"""
FCRS综合基准测试
对比不同配置的系统性能
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem
import numpy as np


class Benchmark:
    """基准测试"""
    
    def __init__(self, name, config):
        self.name = name
        self.config = config
        
    def run(self, steps=500):
        """运行测试"""
        system = FCRSystem(
            pool_capacity=self.config.get('pool_capacity', 5),
            vector_dim=self.config.get('vector_dim', 10)
        )
        
        # 设置参数
        if 'threshold' in self.config:
            system.engine.spawn_reuse_threshold = self.config['threshold']
        if 'gain' in self.config:
            system.engine.min_compression_gain = self.config['gain']
        
        # 运行
        for i in range(steps):
            system.step()
        
        return {
            'name': self.name,
            'dims': system.pool.get_total_dims(),
            'new_dims': len(system.engine.new_dim_history),
            'pool_size': len(system.pool)
        }


def run_benchmark():
    """运行综合基准"""
    print('='*60)
    print('FCRS综合基准测试')
    print('='*60)
    
    configs = [
        # 基线配置
        ('基线(th=2,g=0.001)', {'threshold': 2, 'gain': 0.001}),
        ('高阈值(th=5)', {'threshold': 5, 'gain': 0.001}),
        ('高增益(g=0.01)', {'threshold': 2, 'gain': 0.01}),
        
        # 容量配置
        ('小池(3)', {'pool_capacity': 3, 'threshold': 2, 'gain': 0.001}),
        ('大池(10)', {'pool_capacity': 10, 'threshold': 2, 'gain': 0.001}),
        
        # 维度配置
        ('小初始(5)', {'vector_dim': 5, 'threshold': 2, 'gain': 0.001}),
        ('大初始(20)', {'vector_dim': 20, 'threshold': 2, 'gain': 0.001}),
    ]
    
    results = []
    
    for name, config in configs:
        print('\n[' + name + ']')
        b = Benchmark(name, config)
        r = b.run(500)
        
        print('  维度: ' + str(r['dims']))
        print('  新维度: ' + str(r['new_dims']))
        
        results.append(r)
    
    # 汇总
    print('\n' + '='*60)
    print('基准汇总')
    print('='*60)
    
    print('\n排名(按新维度):')
    sorted_results = sorted(results, key=lambda x: x['new_dims'], reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(str(i) + '. ' + r['name'] + ': ' + str(r['new_dims']) + ' 新维度, ' + str(r['dims']) + ' 维')
    
    # 最佳配置
    best = sorted_results[0]
    print('\n最佳配置: ' + best['name'])
    print('  新维度: ' + str(best['new_dims']))
    print('  总维度: ' + str(best['dims']))


if __name__ == "__main__":
    run_benchmark()
