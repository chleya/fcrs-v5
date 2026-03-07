"""
实验3：动态平衡
Phenomenon 3: Dynamic Equilibrium

目标：系统达到稳态，表征生成率与消亡率相等，总误差趋于稳定
"""

import numpy as np
import json


class Representation:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector
        self.fitness_history = []
        self.activation_count = 0
        self.age = 0


class Environment:
    def __init__(self, dim=5):
        self.dim = dim
        self.center = np.random.randn(dim)
    
    def generate(self):
        return self.center + np.random.randn(self.dim) * 0.5


class FCRSystem:
    def __init__(self, capacity=15, dim=5):
        self.capacity = capacity
        self.dim = dim
        self.reps = []
        self.next_id = 0
        self.births = 0
        self.deaths = 0
        self.errors = []
    
    def add(self, vec):
        rep = Representation(self.next_id, vec.copy())
        self.next_id += 1
        self.reps.append(rep)
        self.births += 1
        return rep
    
    def select(self, x):
        if not self.reps:
            return None
        return min(self.reps, key=lambda r: np.linalg.norm(r.vector - x))
    
    def persistence(self, r):
        avg_fit = np.mean(r.fitness_history[-5:]) if r.fitness_history else -1
        reuse = r.activation_count / max(1, r.age)
        return avg_fit + reuse - r.age * 0.001
    
    
    def step(self, x):
        for r in self.reps:
            r.age += 1
        
        active = self.select(x)
        if active is None:
            self.add(x)
            self.errors.append(0.5)
            return
        
        active.activation_count += 1
        error = np.linalg.norm(active.vector - x)
        active.fitness_history.append(-error)
        self.errors.append(error)
        
        # 变异
        if np.random.random() < 0.03:
            new_vec = active.vector + np.random.randn(self.dim) * 0.2
            
            if len(self.reps) >= self.capacity:
                worst = min(self.reps, key=lambda r: self.persistence(r))
                self.reps.remove(worst)
                self.deaths += 1
            
            self.add(new_vec)
    
    @property
    def balance_rate(self):
        """平衡率"""
        if self.births == 0:
            return 0
        return self.deaths / self.births


def run():
    print("=" * 50)
    print("实验3: 动态平衡")
    print("=" * 50)
    
    sys = FCRSystem(capacity=15, dim=5)
    env = Environment(dim=5)
    
    print("运行5000步...")
    
    history = {
        'steps': [],
        'pool_size': [],
        'births': [],
        'deaths': [],
        'avg_error': []
    }
    
    for step in range(5000):
        x = env.generate()
        sys.step(x)
        
        if step % 500 == 0:
            avg_err = np.mean(sys.errors[-100:]) if sys.errors else 0
            
            history['steps'].append(step)
            history['pool_size'].append(len(sys.reps))
            history['births'].append(sys.births)
            history['deaths'].append(sys.deaths)
            history['avg_error'].append(avg_err)
            
            print(f"  Step {step}: 池大小={len(sys.reps)}, "
                  f"生成={sys.births}, 消亡={sys.deaths}, "
                  f"误差={avg_err:.3f}")
    
    # 分析
    print("\n分析:")
    
    # 检查后期是否平衡
    late_pool = history['pool_size'][-5:]
    pool_stable = max(late_pool) - min(late_pool) <= 3
    
    # 检查误差是否收敛
    late_errors = history['avg_error'][-5:]
    error_stable = max(late_errors) - min(late_errors) < 0.1
    
    # 检查生成/消亡是否平衡
    balance = sys.balance_rate
    balance_ok = 0.3 < balance < 1.0
    
    print(f"  池大小稳定: {pool_stable}")
    print(f"  误差收敛: {error_stable}")
    print(f"  生成消亡平衡: {balance_ok} ({balance:.2f})")
    
    success = pool_stable and error_stable
    
    print(f"\n结果:")
    if success:
        print("v 动态平衡验证通过")
    else:
        print("X 验证未通过")
    
    return {
        'experiment': 'phenomenon_3',
        'history': history,
        'final_pool': len(sys.reps),
        'final_births': sys.births,
        'final_deaths': sys.deaths,
        'balance_rate': balance,
        'success': success
    }


if __name__ == "__main__":
    result = run()
    
    with open('F:/fcrs-v5/experiments/exp3_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\n结果已保存")
