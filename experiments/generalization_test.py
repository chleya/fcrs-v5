"""
结构泛化实验
验证: 有限物质系统是否通过表征扩张实现持续提升的泛化能力

实验设计:
- 训练: E1, E2, E3 (复杂度递增)
- 测试: E4 (未见过的复杂度)
- 对比: 固定维度 vs 动态维度
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from fcrs_architecture import create_fcrs, VectorRepresentation, StructuredEnvironment


def create_environment(complexity):
    """创建指定复杂度的环境"""
    return StructuredEnvironment(input_dim=10, n_classes=complexity)


def train_and_test(train_complexities, test_complexity, pool_capacity=5, lr=0.01):
    """训练+测试"""
    # 创建系统
    env = create_environment(train_complexities[-1])
    fcrs = create_fcrs(
        env_type='structured',
        pool_capacity=pool_capacity,
        input_dim=10,
        n_classes=train_complexities[-1],
        lr=lr
    )
    
    # 初始化表征
    for _ in range(3):
        rep = VectorRepresentation(env.generate_input())
        fcrs.pool.add(rep)
    
    # 训练阶段
    train_errors = []
    for step in range(500):
        fcrs.step()
        if step >= 400:
            train_errors.append(fcrs.get_avg_error())
    
    train_error = np.mean(train_errors) if train_errors else float('inf')
    final_dims = fcrs.pool.get_total_dims()
    
    # 测试阶段 - 在新复杂度环境上测试
    test_env = create_environment(test_complexity)
    test_errors = []
    
    for step in range(100):
        x = test_env.generate_input()
        
        # 使用训练好的表征
        active = fcrs.pool.select(x)
        
        if active is not None:
            error = np.linalg.norm(x - active.vector)
            test_errors.append(error)
            
            # 在线学习
            active.vector += lr * (x - active.vector)
        else:
            test_errors.append(float('inf'))
    
    test_error = np.mean(test_errors) if test_errors else float('inf')
    
    return {
        'train_error': train_error,
        'test_error': test_error,
        'final_dims': final_dims,
        'overfitting': test_error - train_error
    }


def fixed_dimension_baseline(complexities, test_complexity):
    """固定维度基线"""
    env = StructuredEnvironment(input_dim=10, n_classes=complexities[-1])
    fcrs = create_fcrs(
        env_type='structured',
        pool_capacity=1,  # 固定1个表征
        input_dim=10,
        n_classes=complexities[-1],
        lr=0.01
    )
    
    # 初始化
    x = env.generate_input()
    rep = VectorRepresentation(x)
    fcrs.pool.add(rep)
    
    # 训练
    for _ in range(500):
        fcrs.step()
    
    # 测试
    test_env = create_environment(test_complexity)
    test_errors = []
    
    for _ in range(100):
        x = test_env.generate_input()
        active = fcrs.pool.select(x)
        
        if active:
            error = np.linalg.norm(x - active.vector)
            test_errors.append(error)
    
    return {
        'train_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'test_error': np.mean(test_errors),
        'dims': 10,
        'overfitting': np.mean(test_errors) - np.mean(fcrs.errors[-100:])
    }


def run_generalization_experiment():
    """运行结构泛化实验"""
    print('='*60)
    print('Structure Generalization Experiment')
    print('='*60)
    print('\n核心问题: 动态维度是否能持续提升泛化能力?')
    print('='*60)
    
    # 实验1: E1训练, E1测试 (baseline)
    print('\n[Exp1] E1→E1 (Baseline)')
    result_e1 = train_and_test([1], 1)
    print(f'  Train: {result_e1["train_error"]:.4f}, Test: {result_e1["test_error"]:.4f}')
    print(f'  Dims: {result_e1["final_dims"]}, Overfit: {result_e1["overfitting"]:.4f}')
    
    # 实验2: E1训练, E2测试 (未见过的复杂度)
    print('\n[Exp2] E1→E2 (Novel Complexity)')
    result_e2 = train_and_test([1], 2)
    print(f'  Train: {result_e2["train_error"]:.4f}, Test: {result_e2["test_error"]:.4f}')
    print(f'  Dims: {result_e2["final_dims"]}, Overfit: {result_e2["overfitting"]:.4f}')
    
    # 实验3: E1-E2训练, E3测试
    print('\n[Exp3] E1+E2→E3 (Cumulative)')
    result_e3 = train_and_test([1, 2], 3)
    print(f'  Train: {result_e3["train_error"]:.4f}, Test: {result_e3["test_error"]:.4f}')
    print(f'  Dims: {result_e3["final_dims"]}, Overfit: {result_e3["overfitting"]:.4f}')
    
    # 实验4: E1-E3训练, E4测试 (关键!)
    print('\n[Exp4] E1+E2+E3→E4 (Generalization!)')
    result_e4 = train_and_test([1, 2, 3], 4)
    print(f'  Train: {result_e4["train_error"]:.4f}, Test: {result_e4["test_error"]:.4f}')
    print(f'  Dims: {result_e4["final_dims"]}, Overfit: {result_e4["overfitting"]:.4f}')
    
    # 固定维度基线
    print('\n[Baseline] Fixed Dim (pool=1)')
    fixed = fixed_dimension_baseline([1, 2, 3], 4)
    print(f'  Train: {fixed["train_error"]:.4f}, Test: {fixed["test_error"]:.4f}')
    print(f'  Dims: {fixed["dims"]}, Overfit: {fixed["overfitting"]:.4f}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'{"Experiment":<20} {"Train":<10} {"Test":<10} {"Dims":<8} {"Overfit":<10}')
    print('-'*60)
    print(f'{"E1→E1":<20} {result_e1["train_error"]:<10.4f} {result_e1["test_error"]:<10.4f} {result_e1["final_dims"]:<8} {result_e1["overfitting"]:<10.4f}')
    print(f'{"E1→E2":<20} {result_e2["train_error"]:<10.4f} {result_e2["test_error"]:<10.4f} {result_e2["final_dims"]:<8} {result_e2["overfitting"]:<10.4f}')
    print(f'{"E1+E2→E3":<20} {result_e3["train_error"]:<10.4f} {result_e3["test_error"]:<10.4f} {result_e3["final_dims"]:<8} {result_e3["overfitting"]:<10.4f}')
    print(f'{"E1+E2+E3→E4":<20} {result_e4["train_error"]:<10.4f} {result_e4["test_error"]:<10.4f} {result_e4["final_dims"]:<8} {result_e4["overfitting"]:<10.4f}')
    print(f'{"Fixed Baseline":<20} {fixed["train_error"]:<10.4f} {fixed["test_error"]:<10.4f} {fixed["dims"]:<8} {fixed["overfitting"]:<10.4f}')
    
    # 结论
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    
    # 判断泛化能力
    if result_e4['test_error'] < fixed['test_error']:
        print('[PASS] Dynamic dims BEAT fixed dims on novel task!')
    else:
        print('[FAIL] No generalization advantage')
    
    if result_e4['overfitting'] < 0.5:
        print('[PASS] Low overfitting - good generalization')
    else:
        print('[FAIL] High overfitting')
    
    # 判断"持续提升"
    test_errors = [result_e1['test_error'], result_e2['test_error'], 
                   result_e3['test_error'], result_e4['test_error']]
    
    # 理想: 随训练复杂度增加，泛化能力提升（测试误差不增加）
    degradation = result_e4['test_error'] - result_e1['test_error']
    
    if degradation < 0.5:
        print('[PASS] Test error stable despite increasing complexity!')
    else:
        print(f'[WARN] Test error increased by {degradation:.4f}')
    
    return {
        'results': [result_e1, result_e2, result_e3, result_e4],
        'fixed': fixed
    }


if __name__ == "__main__":
    results = run_generalization_experiment()
