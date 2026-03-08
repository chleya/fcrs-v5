"""
Experiment 3: Competition Strength Test
H3: 淘汰率对表征稳定性的影响
"""

import numpy as np
import json
import random


class SimpleRep:
    def __init__(self, vector):
        self.vector = vector.copy()
        self.fitness_history = []
        self.activation_count = 0


class SimpleEnv:
    def __init__(self, input_dim=10, n_classes=3):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class CompetitionFCRS:
    """带竞争强度的FCRS"""
    
    def __init__(self, pool_capacity=5, input_dim=10, n_classes=3, lr=0.01, 
                 lambda_penalty=0.1, elimination_rate=0.3):
        self.input_dim = input_dim
        self.lr = lr
        self.lambda_penalty = lambda_penalty
        self.elimination_rate = elimination_rate
        
        self.representations = []
        self.env = SimpleEnv(input_dim, n_classes)
        
        self.errors = []
        self.dimension = input_dim
        self.spawn_count = 0
        self.elim_count = 0
        
        # 初始化
        for _ in range(3):
            x = self.env.generate_input()
            self.representations.append(SimpleRep(x))
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            v = rep.vector[:self.dimension]
            score = np.dot(v, x) / (np.linalg.norm(v) + 1e-8)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def step(self):
        x = self.env.generate_input()
        
        active = self.select(x)
        
        if active is not None:
            error_vec = x - active.vector[:self.dimension]
            error = np.linalg.norm(error_vec)
            
            active.vector[:self.dimension] += self.lr * error_vec
            
            active.activation_count += 1
            active.fitness_history.append(-error)
            
            self.errors.append(error)
            
            # 结构决策
            self._structural_decision()
        else:
            self.errors.append(float('inf'))
    
    def _structural_decision(self):
        """结构决策 - 带竞争"""
        
        # 边际收益
        gain = random.random() * 0.3
        
        # Spawn: if gain > λ
        if len(self.representations) < 5 and gain > self.lambda_penalty:
            self.dimension += 1
            self.spawn_count += 1
        
        # Competition: 淘汰最低适应度的表征
        if len(self.representations) > 1:
            # 计算适应度
            fitnesses = []
            for r in self.representations:
                if r.fitness_history:
                    f = np.mean(r.fitness_history[-10:])
                else:
                    f = 0
                fitnesses.append(f)
            
            # 按适应度排序
            indices = np.argsort(fitnesses)
            
            # 淘汰最低的 elimination_rate%
            n_elim = max(1, int(len(self.representations) * self.elimination_rate))
            
            # 保留最好的
            keep_indices = indices[n_elim:]
            
            # 更新表征列表
            new_reps = []
            for i, rep in enumerate(self.representations):
                if i in keep_indices:
                    new_reps.append(rep)
                else:
                    self.elim_count += 1
            
            self.representations = new_reps
            
            # 补充新表征（如果需要）
            while len(self.representations) < 3:
                x = self.env.generate_input()
                self.representations.append(SimpleRep(x))


def run_exp3(elimination_rate, seed, steps=500):
    random.seed(seed)
    np.random.seed(seed)
    
    fcrs = CompetitionFCRS(
        pool_capacity=5,
        input_dim=10,
        n_classes=3,
        lr=0.01,
        lambda_penalty=0.1,
        elimination_rate=elimination_rate
    )
    
    for _ in range(steps):
        fcrs.step()
    
    return {
        'elimination_rate': elimination_rate,
        'seed': seed,
        'avg_error': np.mean(fcrs.errors[-100:]) if fcrs.errors else float('inf'),
        'final_dimension': fcrs.dimension,
        'spawn': fcrs.spawn_count,
        'eliminate': fcrs.elim_count,
    }


def main():
    print('='*60)
    print('Experiment 3: Competition Strength')
    print('='*60)
    
    elim_rates = [0.1, 0.3, 0.5, 0.9]
    seeds = [0, 1, 2]
    
    results = []
    
    for rate in elim_rates:
        print(f'\n--- elimination_rate = {rate} ---')
        
        for seed in seeds:
            result = run_exp3(rate, seed)
            results.append(result)
            print(f"seed={seed}: error={result['avg_error']:.4f}, dim={result['final_dimension']}, spawn={result['spawn']}, elim={result['eliminate']}")
    
    # 汇总
    print('\n' + '='*60)
    print('Summary')
    print('='*60)
    
    summary = {}
    for rate in elim_rates:
        rate_results = [r for r in results if r['elimination_rate'] == rate]
        
        summary[rate] = {
            'error': np.mean([r['avg_error'] for r in rate_results]),
            'dimension': np.mean([r['final_dimension'] for r in rate_results]),
            'spawn': np.mean([r['spawn'] for r in rate_results]),
            'eliminate': np.mean([r['eliminate'] for r in rate_results]),
        }
    
    print(f'{"rate":<12} {"error":<12} {"dimension":<12} {"spawn":<10} {"elim":<10}')
    print('-'*60)
    
    for rate in elim_rates:
        s = summary[rate]
        print(f'{rate:<12} {s["error"]:<12.4f} {s["dimension"]:<12.1f} {s["spawn"]:<10.1f} {s["eliminate"]:<10.1f}')
    
    # 保存
    with open('F:/fcrs-v5/experiments/exp3_competition/results.json', 'w') as f:
        json.dump({'summary': summary, 'raw': results}, f, indent=2)
    
    print('\nSaved to results.json')


if __name__ == "__main__":
    main()
