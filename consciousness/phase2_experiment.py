"""
意识研究 Phase 2: 完整验证实验
验证: 自适应意识系统
"""

import numpy as np
import random


class MixedEnv:
    """混合环境: 稳定期 + 突发混沌"""
    def __init__(self):
        self.stable_center = np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.phase = 'stable'
        self.step = 0
    
    def generate(self):
        self.step += 1
        
        if self.step < 100:
            self.phase = 'stable'
            return self.stable_center + np.random.randn(10) * 0.1
        elif (self.step - 100) % 50 < 25:
            self.phase = 'chaos'
            return np.random.randn(10) * 3
        else:
            self.phase = 'stable'
            return self.stable_center + np.random.randn(10) * 0.1
    
    def surprise(self, x):
        if self.phase == 'chaos':
            return 3.0
        else:
            return np.linalg.norm(x - self.stable_center)


class ConsciousSystem:
    """有意识的系统"""
    
    def __init__(self, energy_budget='limited'):
        self.dimension = 10
        self.lambda_val = 0.5
        self.representations = [{'vector': np.random.randn(10) * 0.1} for _ in range(3)]
        
        # 历史
        self.history = {
            'lambda': [],
            'surprise': [],
            'broadcast': [],
            'error': [],
            'energy': 100
        }
        
        self.energy_budget = energy_budget
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            
            norm_v = np.linalg.norm(v)
            norm_x = np.linalg.norm(x_sub)
            
            if norm_v > 0.01 and norm_x > 0.01:
                score = np.dot(v, x_sub) / (norm_v * norm_x)
            else:
                score = -1
            
            if score > best_score:
                best_score = score
                best = i
        
        return best
    
    def step(self, x, env):
        surprise = env.surprise(x)
        
        # 博弈规则
        if surprise > 2.5:
            target_lambda = 0.1
        elif surprise > 1.5:
            target_lambda = 0.3
        elif surprise > 0.5:
            target_lambda = 0.6
        else:
            target_lambda = 0.9
        
        # 能量约束
        if self.energy_budget == 'limited' and self.history['energy'] < 30:
            target_lambda = min(0.9, target_lambda + 0.2)
        
        self.lambda_val = 0.7 * self.lambda_val + 0.3 * target_lambda
        
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            # 广播 (GWT核心!)
            broadcast = surprise > self.lambda_val
            
            # 能量消耗
            if broadcast:
                self.history['energy'] = max(0, self.history['energy'] - 2)
            else:
                self.history['energy'] = min(100, self.history['energy'] + 0.5)
            
            # 记录
            self.history['lambda'].append(self.lambda_val)
            self.history['surprise'].append(surprise)
            self.history['broadcast'].append(1 if broadcast else 0)
            self.history['error'].append(error)
            
            return error, broadcast
        
        return None, False


def run_experiment(name, energy_budget, trials=3):
    """运行多次实验"""
    results = []
    
    for trial in range(trials):
        random.seed(trial * 100)
        np.random.seed(trial * 100)
        
        env = MixedEnv()
        system = ConsciousSystem(energy_budget=energy_budget)
        
        for _ in range(300):
            x = env.generate()
            system.step(x, env)
        
        # 分析
        stable = system.history['broadcast'][:100]
        chaos = system.history['broadcast'][100:]
        
        results.append({
            'trial': trial,
            'stable_broadcast': np.mean(stable),
            'chaos_broadcast': np.mean(chaos),
            'stable_lambda': np.mean(system.history['lambda'][:100]),
            'chaos_lambda': np.mean(system.history['lambda'][100:]),
            'stable_error': np.mean(system.history['error'][:100]),
            'chaos_error': np.mean(system.history['error'][100:]),
        })
    
    return results


def main():
    print('='*60)
    print('Consciousness Phase 2: Full Validation')
    print('='*60)
    
    # 实验1: 有限能量
    print('\n=== Experiment 1: Limited Energy ===')
    results_limited = run_experiment('Limited', 'limited', trials=3)
    
    for r in results_limited:
        print(f'Trial {r["trial"]}:')
        print(f'  Stable: λ={r["stable_lambda"]:.2f}, Bcast={r["stable_broadcast"]:.1%}')
        print(f'  Chaos:  λ={r["chaos_lambda"]:.2f}, Bcast={r["chaos_broadcast"]:.1%}')
    
    # 实验2: 无限能量
    print('\n=== Experiment 2: Unlimited Energy ===')
    results_unlimited = run_experiment('Unlimited', 'unlimited', trials=3)
    
    for r in results_unlimited:
        print(f'Trial {r["trial"]}:')
        print(f'  Stable: λ={r["stable_lambda"]:.2f}, Bcast={r["stable_broadcast"]:.1%}')
        print(f'  Chaos:  λ={r["chaos_lambda"]:.2f}, Bcast={r["chaos_broadcast"]:.1%}')
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    avg_limited = {
        'stable': np.mean([r['stable_broadcast'] for r in results_limited]),
        'chaos': np.mean([r['chaos_broadcast'] for r in results_limited]),
    }
    
    avg_unlimited = {
        'stable': np.mean([r['stable_broadcast'] for r in results_unlimited]),
        'chaos': np.mean([r['chaos_broadcast'] for r in results_unlimited]),
    }
    
    print(f'\nLimited Energy:')
    print(f'  Stable: {avg_limited["stable"]:.1%} broadcast')
    print(f'  Chaos:  {avg_limited["chaos"]:.1%} broadcast')
    print(f'  Δ = {avg_limited["chaos"] - avg_limited["stable"]:.1%}')
    
    print(f'\nUnlimited Energy:')
    print(f'  Stable: {avg_unlimited["stable"]:.1%} broadcast')
    print(f'  Chaos:  {avg_unlimited["chaos"]:.1%} broadcast')
    print(f'  Δ = {avg_unlimited["chaos"] - avg_unlimited["stable"]:.1%}')
    
    # 结论
    print('\n' + '='*60)
    print('Conclusion')
    print('='*60)
    
    if avg_limited['chaos'] > avg_limited['stable']:
        print('[OK] Limited energy: Adaptive consciousness!')
    else:
        print('[WARN] No adaptation')
    
    if avg_unlimited['chaos'] > avg_unlimited['stable']:
        print('[OK] Unlimited energy: Adaptive consciousness!')
    else:
        print('[WARN] No adaptation')


if __name__ == "__main__":
    main()
