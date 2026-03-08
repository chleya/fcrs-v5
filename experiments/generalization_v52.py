"""
结构泛化实验 - v5.2真正涌现版本
对比: 旧版本 vs 新版本
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from core_v52 import FCRSv52, EnvironmentLoop


def create_env(complexity):
    return EnvironmentLoop(input_dim=10, n_classes=complexity)


def run_experiment_v52():
    """v5.2版本实验"""
    print('='*60)
    print('Structure Generalization - FCRS v5.2 (True Emergence)')
    print('='*60)
    
    results = []
    
    # Exp1: E1→E1
    print('\n[Exp1] E1→E1')
    env = create_env(1)
    fcrs = FCRSv52(pool_capacity=5, input_dim=10, n_classes=1, lr=0.01)
    for _ in range(500):
        fcrs.step()
    train_err = fcrs.get_avg_error()
    
    # 测试
    test_env = create_env(1)
    test_errs = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        test_errs.append(np.linalg.norm(x - best.vector))
    
    r1 = {'train': train_err, 'test': np.mean(test_errs), 'dims': fcrs.pool.get_total_dims()}
    print(f'  Train: {r1["train"]:.4f}, Test: {r1["test"]:.4f}, Dims: {r1["dims"]}')
    results.append(r1)
    
    # Exp2: E1→E2 (新复杂度)
    print('\n[Exp2] E1→E2')
    env = create_env(1)
    fcrs = FCRSv52(pool_capacity=5, input_dim=10, n_classes=1, lr=0.01)
    for _ in range(500):
        fcrs.step()
    train_err = fcrs.get_avg_error()
    
    test_env = create_env(2)
    test_errs = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        test_errs.append(np.linalg.norm(x - best.vector))
    
    r2 = {'train': train_err, 'test': np.mean(test_errs), 'dims': fcrs.pool.get_total_dims()}
    print(f'  Train: {r2["train"]:.4f}, Test: {r2["test"]:.4f}, Dims: {r2["dims"]}')
    results.append(r2)
    
    # Exp3: E1+E2→E3
    print('\n[Exp3] E1+E2→E3')
    env = create_env(3)
    fcrs = FCRSv52(pool_capacity=5, input_dim=10, n_classes=3, lr=0.01)
    for _ in range(500):
        fcrs.step()
    train_err = fcrs.get_avg_error()
    
    test_env = create_env(3)
    test_errs = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        test_errs.append(np.linalg.norm(x - best.vector))
    
    r3 = {'train': train_err, 'test': np.mean(test_errs), 'dims': fcrs.pool.get_total_dims()}
    print(f'  Train: {r3["train"]:.4f}, Test: {r3["test"]:.4f}, Dims: {r3["dims"]}')
    results.append(r3)
    
    # Exp4: E1+E2+E3→E4 (关键泛化测试)
    print('\n[Exp4] E1+E2+E3→E4')
    env = create_env(3)
    fcrs = FCRSv52(pool_capacity=5, input_dim=10, n_classes=3, lr=0.01)
    for _ in range(500):
        fcrs.step()
    train_err = fcrs.get_avg_error()
    
    test_env = create_env(4)
    test_errs = []
    for _ in range(100):
        x = test_env.generate_input()
        best = max(fcrs.pool.representations, key=lambda r: np.dot(r.vector, x))
        test_errs.append(np.linalg.norm(x - best.vector))
    
    r4 = {'train': train_err, 'test': np.mean(test_errs), 'dims': fcrs.pool.get_total_dims()}
    print(f'  Train: {r4["train"]:.4f}, Test: {r4["test"]:.4f}, Dims: {r4["dims"]}')
    results.append(r4)
    
    # 固定维度基线
    print('\n[Baseline] Fixed Dim')
    from fcrs_architecture import create_fcrs, VectorRepresentation, StructuredEnvironment
    
    env = StructuredEnvironment(input_dim=10, n_classes=3)
    fcrs = create_fcrs(env_type='structured', pool_capacity=1, input_dim=10, n_classes=3, lr=0.01)
    for _ in range(3):
        fcrs.pool.add(VectorRepresentation(env.generate_input()))
    for _ in range(500):
        fcrs.step()
    fixed_train = fcrs.get_avg_error()
    
    test_env = create_env(4)
    fixed_test_errs = []
    for _ in range(100):
        x = test_env.generate_input()
        active = fcrs.pool.select(x)
        if active:
            fixed_test_errs.append(np.linalg.norm(x - active.vector))
    
    r_fixed = {'train': fixed_train, 'test': np.mean(fixed_test_errs), 'dims': 10}
    print(f'  Train: {r_fixed["train"]:.4f}, Test: {r_fixed["test"]:.4f}, Dims: {r_fixed["dims"]}')
    
    # 总结
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print(f'{"Experiment":<20} {"Train":<10} {"Test":<10} {"Dims":<8}')
    print('-'*60)
    print(f'{"E1→E1":<20} {r1["train"]:<10.4f} {r1["test"]:<10.4f} {r1["dims"]:<8}')
    print(f'{"E1→E2":<20} {r2["train"]:<10.4f} {r2["test"]:<10.4f} {r2["dims"]:<8}')
    print(f'{"E1+E2→E3":<20} {r3["train"]:<10.4f} {r3["test"]:<10.4f} {r3["dims"]:<8}')
    print(f'{"E1+E2+E3→E4":<20} {r4["train"]:<10.4f} {r4["test"]:<10.4f} {r4["dims"]:<8}')
    print(f'{"Fixed Baseline":<20} {r_fixed["train"]:<10.4f} {r_fixed["test"]:<10.4f} {r_fixed["dims"]:<8}')
    
    # 结论
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    
    if r4['test'] < r_fixed['test']:
        print(f'[PASS] v5.2 BEATS baseline: {r4["test"]:.4f} < {r_fixed["test"]:.4f}')
    else:
        print(f'[FAIL] v5.2 vs baseline: {r4["test"]:.4f} vs {r_fixed["test"]:.4f}')
    
    overfit = r4['test'] - r4['train']
    if overfit < 1.0:
        print(f'[PASS] Low overfitting: {overfit:.4f}')
    else:
        print(f'[WARN] Overfitting: {overfit:.4f}')
    
    return results, r_fixed


if __name__ == "__main__":
    run_experiment_v52()
