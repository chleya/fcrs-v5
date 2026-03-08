"""
FCRS测试
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from fcrs import FCRS
import numpy as np


def test_basic():
    """基础测试"""
    fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)
    
    for _ in range(100):
        x = np.random.randn(10)
        fcrs.step(x)
    
    assert fcrs.get_avg_error() > 0
    print('test_basic: PASS')


def test_params():
    """参数测试"""
    for lr in [0.001, 0.01, 0.1]:
        fcrs = FCRS(pool_capacity=5, input_dim=10, lr=lr)
        
        for _ in range(50):
            x = np.random.randn(10)
            fcrs.step(x)
    
    print('test_params: PASS')


def test_stats():
    """统计测试"""
    fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)
    
    for _ in range(100):
        x = np.random.randn(10)
        fcrs.step(x)
    
    stats = fcrs.get_stats()
    assert 'n_reps' in stats
    assert 'avg_error' in stats
    
    print('test_stats: PASS')


if __name__ == "__main__":
    test_basic()
    test_params()
    test_stats()
    print('\nAll tests passed!')
