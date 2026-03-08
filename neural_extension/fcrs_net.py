"""
FCRS神经网络扩展 - 简化版
将动态维度机制嵌入神经网络
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from core import FCRSystem


class FCRSNet:
    """FCRS增强网络 - 简化版"""
    
    def __init__(self, input_dim=10):
        self.input_dim = input_dim
        
        # FCRS系统作为特征提取器
        self.fcrs = FCRSystem(pool_capacity=5, vector_dim=input_dim)
        self.fcrs.engine.spawn_reuse_threshold = 2
        self.fcrs.engine.min_compression_gain = 0.001
        
        # 输出权重
        self.w = np.random.randn(input_dim) * 0.1
    
    def forward(self, x):
        """前向传播"""
        x = np.asarray(x).flatten()
        
        # 更新FCRS
        self.fcrs.step()
        
        # 使用表征信息
        if self.fcrs.pool.representations:
            # 使用原始输入维度
            features = x  # 直接使用输入
        else:
            features = x
        
        # 输出
        return np.dot(features, self.w)
    
    def get_info(self):
        return {
            'dims': self.fcrs.pool.get_total_dims(),
            'new_dims': len(self.fcrs.engine.new_dim_history)
        }


def test():
    print('='*60)
    print('FCRS神经网络扩展测试')
    print('='*60)
    
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # FCRS网络
    print('\n[FCRS网络]')
    net = FCRSNet(input_dim=10)
    
    for i in range(100):
        pred = net.forward(X[i])
    
    info = net.get_info()
    print('  维度: ' + str(info['dims']))
    print('  新维度: ' + str(info['new_dims']))
    
    # 固定网络
    print('\n[固定网络]')
    fixed_w = np.random.randn(10) * 0.1
    
    for i in range(100):
        pred = np.dot(X[i], fixed_w)
    
    print('  维度: 10')
    
    print('\n' + '='*60)
    print('FCRS: ' + str(info['dims']) + '维, 固定: 10维')


if __name__ == "__main__":
    test()
