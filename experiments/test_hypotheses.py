"""
验证表征达尔文主义假说
H1: 资源临界假说
H2: 涌现阈值假说  
H3: 多维度协同假说
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
from fcrs_architecture import create_fcrs, VectorRepresentation, StructuredEnvironment


def hypothesis_1_resource_critical():
    """H1: 资源临界假说"""
    print('='*60)
    print('Hypothesis 1: Resource Critical Point')
    print('='*60)
    
    pool_sizes = [1, 2, 3, 5, 10, 20]
    results = []
    
    for pool_size in pool_sizes:
        # 创建系统
        env = StructuredEnvironment(input_dim=10, n_classes=3)
        fcrs = create_fcrs(
            env_type='structured',
            pool_capacity=pool_size,
            input_dim=10,
            n_classes=3,
            lr=0.01
        )
        
        # 初始化表征
        for _ in range(3):
            rep = VectorRepresentation(env.generate_input())
            fcrs.pool.add(rep)
        
        # 运行
        errors = []
        for _ in range(500):
            fcrs.step()
            if len(fcrs.errors) >= 100:
                errors.append(fcrs.get_avg_error())
        
        avg_error = np.mean(errors[-20:]) if errors else float('inf')
        
        # 计算持久度(稳定性)
        stability = 1.0 / (np.std(errors[-20:]) + 0.1) if len(errors) >= 20 else 0
        
        results.append({
            'pool_size': pool_size,
            'error': avg_error,
            'stability': stability
        })
        
        print(f'pool={pool_size}: error={avg_error:.4f}, stability={stability:.2f}')
    
    return results


def hypothesis_2_emergence_threshold():
    """H2: 涌现阈值假说"""
    print('\n' + '='*60)
    print('Hypothesis 2: Emergence Threshold')
    print('='*60)
    
    complexities = [1, 2, 3, 5, 10, 20]
    results = []
    
    for complexity in complexities:
        env = StructuredEnvironment(input_dim=10, n_classes=complexity)
        fcrs = create_fcrs(
            env_type='structured',
            pool_capacity=5,
            input_dim=10,
            n_classes=complexity,
            lr=0.01
        )
        
        for _ in range(3):
            rep = VectorRepresentation(env.generate_input())
            fcrs.pool.add(rep)
        
        # 运行并记录波动率
        errors = []
        for _ in range(500):
            fcrs.step()
            if len(fcrs.errors) >= 50:
                errors.append(fcrs.get_avg_error())
        
        # 计算波动率
        volatility = np.std(errors) if len(errors) > 1 else 0
        
        results.append({
            'complexity': complexity,
            'volatility': volatility,
            'final_error': np.mean(errors[-20:]) if errors else 0
        })
        
        print(f'complexity={complexity}: volatility={volatility:.4f}, error={results[-1]["final_error"]:.4f}')
    
    return results


def hypothesis_3_multidimensional_synergy():
    """H3: 多维度协同假说"""
    print('\n' + '='*60)
    print('Hypothesis 3: Multidimensional Synergy')
    print('='*60)
    
    # 需要协同的任务：不同域的信息需要组合
    # 这里简化为多类环境
    
    env = StructuredEnvironment(input_dim=10, n_classes=3)
    fcrs = create_fcrs(
        env_type='structured',
        pool_capacity=5,
        input_dim=10,
        n_classes=3,
        lr=0.01
    )
    
    for _ in range(3):
        rep = VectorRepresentation(env.generate_input())
        fcrs.pool.add(rep)
    
    # 运行
    errors = []
    dim_usage = []
    
    for i in range(1000):
        fcrs.step()
        
        if i % 10 == 0:
            errors.append(fcrs.get_avg_error())
            
            # 计算各表征使用率
            if fcrs.pool.representations:
                usage = [r.activation_count / (i+1) for r in fcrs.pool.representations]
                dim_usage.append(np.std(usage))
    
    print(f'Final error: {errors[-1]:.4f}')
    print(f'Dimension usage std: {np.mean(dim_usage[-20:]):.4f}')
    
    return {
        'error': errors[-1],
        'dim_usage_std': np.mean(dim_usage[-20:])
    }


def main():
    """运行所有假说验证"""
    print('\n# Testing Representation Darwinism Hypotheses\n')
    
    # H1: 资源临界
    h1_results = hypothesis_1_resource_critical()
    
    # H2: 涌现阈值
    h2_results = hypothesis_2_emergence_threshold()
    
    # H3: 多维度协同
    h3_result = hypothesis_3_multidimensional_synergy()
    
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    print('H1: Resource critical point - see results above')
    print('H2: Emergence threshold - see results above')  
    print('H3: Multidimensional synergy - dim_usage_std={:.4f}'.format(h3_result['dim_usage_std']))


if __name__ == "__main__":
    main()
