"""
验证实验 - 调整参数后的完整版本
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class Rep:
    id: int
    vector: np.ndarray
    fitness: float = 0.0
    origin: str = 'initial'
    birth: int = 0


class EmergentFCRS:
    """涌现驱动系统"""
    
    def __init__(self, capacity=10, alpha=0.4, beta=0.3, gamma=0.3):
        self.capacity = capacity
        self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
        
        # 临界状态
        self.state_history = []
        self.error_history = []
        
        # 统计
        self.births = 0
        self.birth_times = []
        self.crit_history = []
        
        self.step = 0
    
    def criticality(self):
        if len(self.state_history) < 10:
            return 0.5
        fluc = np.std(self.state_history[-50:])
        return min(1.0, fluc * 0.5)
    
    def step(self, x):
        self.step += 1
        
        # 选择
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        
        # 误差
        error = np.linalg.norm(best.vector - x)
        
        # 学习
        best.vector += 0.1 * (x - best.vector)
        best.fitness = -error
        
        # 记录
        self.state_history.append(np.linalg.norm(x))
        self.error_history.append(error)
        if len(self.state_history) > 50:
            self.state_history.pop(0)
        
        # 临界状态
        crit = self.criticality()
        self.crit_history.append(crit)
        
        # 涌现产生 (调整条件)
        if crit > 0.3 and np.random.random() < 0.2:
            new_vec = best.vector + np.random.randn(10) * 0.2 * crit
            self.reps.append(Rep(len(self.reps), new_vec, -error, 'perturbation', self.step))
            self.births += 1
            self.birth_times.append(self.step)
        
        # 容量限制
        if len(self.reps) > self.capacity:
            # 淘汰最弱的
            self.reps.sort(key=lambda r: r.fitness)
            self.reps.pop(0)
        
        # 重新编号
        for i, r in enumerate(self.reps):
            r.id = i
        
        return error


class OptimizedFCRS:
    """优化驱动系统"""
    
    def __init__(self, capacity=10, threshold=0.5):
        self.capacity = capacity
        self.threshold = threshold
        self.gain = 0
        self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
        self.step = 0
        self.birth_times = []
    
    def step(self, x):
        self.step += 1
        
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        error = np.linalg.norm(best.vector - x)
        best.vector += 0.1 * (x - best.vector)
        
        # 阈值判断
        self.gain += 0.01
        if self.gain > self.threshold:
            new_vec = best.vector + np.random.randn(10) * 0.1
            self.reps.append(Rep(len(self.reps), new_vec, -error, 'threshold', self.step))
            self.birth_times.append(self.step)
            self.gain = 0
        
        if len(self.reps) > self.capacity:
            self.reps.pop(0)
        
        for i, r in enumerate(self.reps):
            r.id = i
        
        return error


class Env:
    def __init__(self, n_classes=5):
        self.centers = {i: np.random.randn(10)*2 for i in range(n_classes)}
    
    def generate(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(10)*0.3


# ==================== 实验1 ====================
def exp1():
    print("="*60)
    print("Exp1: Emergent vs Optimized")
    print("="*60)
    
    for run in range(10):
        np.random.seed(run*100)
        
        # Emergent
        env = Env(5)
        sys1 = EmergentFCRS(10)
        
        for _ in range(3000):
            sys1.step(env.generate())
        
        # Optimized
        np.random.seed(run*100)
        env = Env(5)
        sys2 = OptimizedFCRS(10, 0.3)
        
        for _ in range(3000):
            sys2.step(env.generate())
        
        print(f"Run {run}: Emergent={sys1.births}, Optimized={len(sys2.birth_times)}")
    
    return {'emergent': sys1.births, 'optimized': len(sys2.birth_times)}


# ==================== 实验2 ====================
def exp2():
    print("\n" + "="*60)
    print("Exp2: Criticality Correlation")
    print("="*60)
    
    np.random.seed(42)
    env = Env(5)
    sys = EmergentFCRS(10)
    
    for _ in range(5000):
        sys.step(env.generate())
    
    print(f"Total births: {sys.births}")
    print(f"Mean criticality: {np.mean(sys.crit_history):.3f}")
    
    if sys.births > 0:
        print(f"Birth times: {sys.birth_times[:10]}")
    
    return {'births': sys.births, 'crit': np.mean(sys.crit_history)}


# ==================== 实验3 ====================
def exp3():
    print("\n" + "="*60)
    print("Exp3: Multi-dim Fitness")
    print("="*60)
    
    configs = [
        {'a': 0.6, 'b': 0.2, 'c': 0.2, 'name': 'High-Pred'},
        {'a': 0.2, 'b': 0.6, 'c': 0.2, 'name': 'High-Comp'},
        {'a': 0.2, 'b': 0.2, 'c': 0.6, 'name': 'High-Beh'},
    ]
    
    results = {}
    
    for cfg in configs:
        np.random.seed(42)
        env = Env(5)
        sys = EmergentFCRS(10, cfg['a'], cfg['b'], cfg['c'])
        
        for _ in range(3000):
            sys.step(env.generate())
        
        results[cfg['name']] = sys.births
        print(f"{cfg['name']}: births={sys.births}")
    
    return results


# ==================== 实验4 ====================
def exp4():
    print("\n" + "="*60)
    print("Exp4: Probation Screening")
    print("="*60)
    
    np.random.seed(42)
    env = Env(5)
    sys = EmergentFCRS(10)
    
    for _ in range(5000):
        sys.step(env.generate())
    
    origins = [r.origin for r in sys.reps]
    
    print(f"Births: {sys.births}")
    print(f"Final origins: {origins}")
    
    return {'births': sys.births, 'origins': origins}


# ==================== 实验5 ====================
def exp5():
    print("\n" + "="*60)
    print("Exp5: Resource Constraint")
    print("="*60)
    
    for cap in [15, 10, 5, 3]:
        np.random.seed(42)
        env = Env(5)
        sys = EmergentFCRS(cap)
        
        for _ in range(2000):
            sys.step(env.generate())
        
        print(f"Capacity {cap}: births={sys.births}, final={len(sys.reps)}")


# ==================== Main ====================
def main():
    print("FCRS-v5.2 Strict Validation\n")
    
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    
    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
