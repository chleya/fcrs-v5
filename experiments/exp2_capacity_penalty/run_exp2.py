"""
Experiment 2: Capacity Cost Test
测试 H1: 缺乏容量成本 → 添加 λ * dimension penalty
"""

import sys
sys.path.insert(0, 'F:/fcrs-v5')

import numpy as np
import json
from core_v52 import FCRSv52, EnvironmentLoop


class FCRSv52WithPenalty:
    """带容量惩罚的FCRS - 简化版"""
    
    def __init__(self, pool_capacity=5, input_dim=10, n_classes=5, lr=0.01, lambda_penalty=0.0):
        from core_v52 import RepresentationPool, EmergenceEngine
        
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        
        self.pool = RepresentationPool(capacity=pool_capacity)
        self.env = EnvironmentLoop(input_dim, n_classes)
        self.engine = EmergenceEngine(self.pool, self.env)
        
        self.step_count = 0
        self.errors = []
        self.task_losses = []
        self.penalties = []
        
        # 初始化表征
        for _ in range(3):
            x = self.env.generate_input()
            self.pool.add(x)
    
    def get_total_loss(self, task_loss):
        """总损失 = 任务损失 + 容量惩罚"""
        active_dims = sum(1 for r in self.pool.representations 
                         if np.max(np.abs(r.vector)) > 0.01)
        penalty = self.lambda_penalty * active_dims
        return task_loss + penalty, penalty
    
    def step(self):
        """一步 - 记录额外指标"""
        self.step_count += 1
        
        x = self.env.generate_input()
        
        best_rep = None
        best_score = -float('inf')
        
        for rep in self.pool.representations:
            score = np.dot(rep.vector, x) / (np.linalg.norm(rep.vector) + 1e-8)
            if score > best_score:
                best_score = score
                best_rep = rep
        
        if best_rep is not None:
            error_vec = x - best_rep.vector
            error = np.linalg.norm(error_vec)
            task_loss = error ** 2
            
            # 计算总损失（含惩罚）
            total_loss, penalty = self.get_total_loss(task_loss)
            
            # 学习
            best_rep.vector += self.lr * error_vec
            
            best_rep.activation_count += 1
            best_rep.fitness_history.append(-error)
            best_rep.age += 1
            
            self.errors.append(total_loss)
            self.task_losses.append(task_loss)
            self.penalties.append(penalty)
            
            if len(self.pool) < self.pool.capacity:
                self.engine.try_emergence()
            else:
                self.engine.try_emergence()
                if len(self.pool) > self.pool.capacity:
                    self.engine.delete_worst()
        else:
            self.errors.append(float('inf'))
            self.task_losses.append(float('inf'))
            self.penalties.append(0)


def run_experiment(lambda_penalty, seed, steps=500):
    """运行单次实验"""
    np.random.seed(seed)
    
    fcrs = FCRSv52WithPenalty(
        pool_capacity=5,
        input_dim=10,
        n_classes=3,
        lr=0.01,
        lambda_penalty=lambda_penalty
    )
    
    # 记录
    task_losses = []
    penalties = []
    dimensions = []
    active_dims = []
    
    for _ in range(steps):
        fcrs.step()
        
        if _ >= steps - 100:
            task_losses.append(np.mean(fcrs.task_losses[-100:]) if fcrs.task_losses else 0)
            penalties.append(np.mean(fcrs.penalties[-100:]) if fcrs.penalties else 0)
            dimensions.append(fcrs.pool.get_total_dims())
            active_dims.append(sum(1 for r in fcrs.pool.representations 
                                  if np.max(np.abs(r.vector)) > 0.01))
    
    return {
        'lambda': lambda_penalty,
        'seed': seed,
        'avg_task_loss': np.mean(task_losses) if task_losses else float('inf'),
        'avg_penalty': np.mean(penalties) if penalties else 0,
        'avg_dimension': np.mean(dimensions) if dimensions else 0,
        'avg_active_dimension': np.mean(active_dims) if active_dims else 0,
    }


def run_exp2():
    """运行Exp2: Capacity Cost Test"""
    print('='*60)
    print('Experiment 2: Capacity Cost Test')
    print('='*60)
    
    lambdas = [0, 0.001, 0.01, 0.1]
    seeds = [0, 1, 2]
    
    results = []
    
    total_runs = len(lambdas) * len(seeds)
    run_id = 0
    
    for lam in lambdas:
        print(f'\n--- λ = {lam} ---')
        
        for seed in seeds:
            run_id += 1
            print(f'[{run_id}/{total_runs}] seed={seed}', end=' ')
            
            result = run_experiment(lam, seed)
            results.append(result)
            
            print(f"loss={result['avg_task_loss']:.4f}, dim={result['avg_dimension']:.1f}")
    
    # 汇总
    print('\n' + '='*60)
    print('Results Summary')
    print('='*60)
    
    # 按λ聚合
    summary = {}
    for lam in lambdas:
        lam_results = [r for r in results if r['lambda'] == lam]
        
        summary[lam] = {
            'task_loss': np.mean([r['avg_task_loss'] for r in lam_results]),
            'penalty': np.mean([r['avg_penalty'] for r in lam_results]),
            'dimension': np.mean([r['avg_dimension'] for r in lam_results]),
            'active_dimension': np.mean([r['avg_active_dimension'] for r in lam_results]),
        }
    
    # 打印表格
    print(f'{"λ":<10} {"task_loss":<12} {"penalty":<12} {"dimension":<12} {"active_dim":<12}')
    print('-'*60)
    
    for lam in lambdas:
        s = summary[lam]
        print(f'{lam:<10} {s["task_loss"]:<12.4f} {s["penalty"]:<12.4f} {s["dimension"]:<12.1f} {s["active_dimension"]:<12.1f}')
    
    # 保存结果
    with open('F:/fcrs-v5/experiments/exp2_capacity_penalty/results.json', 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    
    print('\nResults saved to experiments/exp2_capacity_penalty/results.json')
    
    # 分析
    print('\n' + '='*60)
    print('Analysis')
    print('='*60)
    
    dims = [summary[lam]['dimension'] for lam in lambdas]
    losses = [summary[lam]['task_loss'] for lam in lambdas]
    
    # 维度是否随λ减少？
    if dims[0] > dims[-1]:
        print('[OK] dimension decreases with λ')
    else:
        print('[FAIL] dimension does not decrease')
    
    # 是否出现U形曲线？
    if losses[1] < losses[0] and losses[2] < losses[3]:
        print('[OK] U-curve pattern detected')
    elif min(losses) == losses[1] or min(losses) == losses[2]:
        print('[OK] optimal λ exists')
    else:
        print('[WARN] no clear optimal')
    
    return summary


if __name__ == "__main__":
    run_exp2()
