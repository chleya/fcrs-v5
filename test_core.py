"""
FCRS-v5 快速测试
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

from core import FCRSystem


def test_basic():
    """基础测试"""
    system = FCRSystem(pool_capacity=5, vector_dim=10)
    system.engine.spawn_reuse_threshold = 2
    system.engine.min_compression_gain = 0.001
    
    for i in range(100):
        system.step()
    
    assert system.pool.get_total_dims() > 0
    print('test_basic: PASS')


def test_parameters():
    """参数测试"""
    for th in [1, 2, 3]:
        for gain in [0.001, 0.01]:
            system = FCRSystem(pool_capacity=5, vector_dim=10)
            system.engine.spawn_reuse_threshold = th
            system.engine.min_compression_gain = gain
            
            for i in range(50):
                system.step()
    
    print('test_parameters: PASS')


def test_statistics():
    """统计测试"""
    system = FCRSystem(pool_capacity=5, vector_dim=10)
    
    for i in range(100):
        system.step()
    
    stats = system.get_statistics()
    assert 'total_dims' in stats
    assert 'pool_size' in stats
    
    print('test_statistics: PASS')


if __name__ == "__main__":
    test_basic()
    test_parameters()
    test_statistics()
    print('\nAll tests passed!')
