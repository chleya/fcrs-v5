"""
结构泛化实验 - 公平对比
控制变量: 相同维度预算
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from core_v52 import FCRSv52, EnvironmentLoop


def create_env(complexity):
    return EnvironmentLoop(input_dim=10, n_classes=complexity)


def run_fair_comparison():
    """公平对比: 相同维度预算"""
    print('='*60)
    print('Fair Comparison: Same Dimensional Budget')
    print('='*60)
    
    # 对比1: 动态(3rep×10dim=30) vs 固定(1rep×10dim=10)
    print('\n[Config A] Dynamic: pool=3, dim=10/rep = 30 total')
    print('[Config B] Fixed: pool=1, dim=10 = 10 total')
    print('[Config C] Dynamic: pool=1, dim=10 = 10 total (fair)')
    
    results = {}
    
    # A: 动态30维
    print('\n--- Dynamic (30 dims) ---')
    env = create_env(3)
    fcrs = FCRSv52(pool_capacity=3, input_dim=10, n_classes=3, lr=0.01)
    for _ in range(500):
        fcrs.step()
    dyn_train = fcrs.get_avg_error()
    
    test_env = create_env(4)
    dyn_test = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        dyn_test.append(np.linalg.norm(x - best.vector))
    
    results['dynamic_30'] = {'train': dyn_train, 'test': np.mean(dyn_test), 'dims': 30}
    print(f'Train: {dyn_train:.4f}, Test: {np.mean(dyn_test):.4f}')
    
    # B: 固定10维
    print('\n--- Fixed (10 dims) ---')
    env = create_env(3)
    fcrs = FCRSv52(pool_capacity=1, input_dim=10, n_classes=3, lr=0.01)
    # 只用1个表征
    x = env.generate_input()
    fcrs.pool.add(x)
    for _ in range(500):
        fcrs.step()
    fix_train = fcrs.get_avg_error()
    
    test_env = create_env(4)
    fix_test = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        fix_test.append(np.linalg.norm(x - best.vector))
    
    results['fixed_10'] = {'train': fix_train, 'test': np.mean(fix_test), 'dims': 10}
    print(f'Train: {fix_train:.4f}, Test: {np.mean(fix_test):.4f}')
    
    # 总结
    print('\n' + '='*60)
    print('Results')
    print('='*60)
    print(f'Dynamic 30D: Train={results["dynamic_30"]["train"]:.4f}, Test={results["dynamic_30"]["test"]:.4f}')
    print(f'Fixed   10D: Train={results["fixed_10"]["train"]:.4f}, Test={results["fixed_10"]["test"]:.4f}')
    
    # 泛化比较
    dyn_overfit = results['dynamic_30']['test'] - results['dynamic_30']['train']
    fix_overfit = results['fixed_10']['test'] - results['fixed_10']['train']
    
    print(f'\nOverfitting:')
    print(f'  Dynamic: {dyn_overfit:.4f}')
    print(f'  Fixed:   {fix_overfit:.4f}')
    
    if results['dynamic_30']['test'] < results['fixed_10']['test']:
        print(f'\n[OK] Dynamic BETTER on test: {results["dynamic_30"]["test"]:.4f} < {results["fixed_10"]["test"]:.4f}')
    else:
        print(f'\n[ISSUE] Dynamic WORSE on test')


if __name__ == "__main__":
    run_fair_comparison()
