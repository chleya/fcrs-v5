"""
FCRS-v5.3 验证实验
压缩-预测驱动系统 vs 旧系统
"""

import numpy as np
import sys
sys.path.insert(0, 'F:/fcrs-v5')

from fcrs_predictive import PredictiveFCRS, RandomFCRS


class OptimizedFCRS:
    """旧版本: 优化驱动"""
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.reps = []
        self.step = 0
        self.births = 0
        self.birth_times = []
        self.gain = 0
    
    def update(self, x):
        self.step += 1
        
        if not self.reps:
            self.reps = [np.random.randn(10)*0.5 for _ in range(3)]
        
        best = max(self.reps, key=lambda v: np.dot(v, x))
        
        # 阈值驱动
        self.gain += 0.01
        if self.gain > 0.5:
            new_vec = best + np.random.randn(10) * 0.1
            self.reps.append(new_vec)
            self.births += 1
            self.birth_times.append(self.step)
            self.gain = 0
        
        if len(self.reps) > self.capacity:
            self.reps.pop(0)
        
        return np.linalg.norm(best - x)


class Env:
    def __init__(self, n=5):
        self.centers = {i: np.random.randn(10)*2 for i in range(n)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(10)*0.3


# ==================== 实验1: Emergent vs Optimized ====================
def exp1():
    print("="*60)
    print("Exp1: Predictive vs Optimized")
    print("="*60)
    
    results = {'predictive': [], 'optimized': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        # Predictive
        env = Env(5)
        sys1 = PredictiveFCRS(10, 10, 5)
        
        for _ in range(2000):
            sys1.update(env.gen())
        
        results['predictive'].append(len(sys1.representations))
        
        # Optimized
        np.random.seed(run * 100)
        env = Env(5)
        sys2 = OptimizedFCRS(10)
        
        for _ in range(2000):
            sys2.update(env.gen())
        
        results['optimized'].append(sys2.births)
    
    print(f"Predictive reps: {np.mean(results['predictive']):.1f}")
    print(f"Optimized births: {np.mean(results['optimized']):.1f}")


# ==================== 实验2: Criticality ====================
def exp2():
    print("\n" + "="*60)
    print("Exp2: Compression-Prediction Correlation")
    print("="*60)
    
    np.random.seed(42)
    env = Env(5)
    sys = PredictiveFCRS(10, 10, 5)
    
    errors = []
    rep_counts = []
    
    for _ in range(3000):
        x = env.gen()
        err = sys.update(x)
        errors.append(err)
        rep_counts.append(len(sys.representations))
    
    print(f"Mean error: {np.mean(errors):.3f}")
    print(f"Mean reps: {np.mean(rep_counts):.1f}")
    
    # 相关性
    if len(errors) > 100:
        corr = np.corrcoef(errors[-500:], rep_counts[-500:])[0,1]
        print(f"Error-Rep correlation: {corr:.3f}")


# ==================== 实验3: Multi-dim Fitness ====================
def exp3():
    print("\n" + "="*60)
    print("Exp3: Different Compression Dimensions")
    print("="*60)
    
    for dim in [3, 5, 8]:
        np.random.seed(42)
        env = Env(5)
        sys = PredictiveFCRS(10, 10, dim)
        
        errs = []
        for _ in range(2000):
            errs.append(sys.update(env.gen()))
        
        print(f"Dim {dim}: error={np.mean(errs):.3f}, reps={len(sys.representations)}")


# ==================== 实验4: Probation/Screening ====================
def exp4():
    print("\n" + "="*60)
    print("Exp4: Screening Effectiveness")
    print("="*60)
    
    np.random.seed(42)
    env = Env(5)
    sys = PredictiveFCRS(10, 10, 5)
    
    for _ in range(3000):
        sys.update(env.gen())
    
    # 分析表征来源
    print(f"Final reps: {len(sys.representations)}")
    print(f"Compression active: {sys.global_compression.vector[:3]}")


# ==================== 实验5: Resource Constraint ====================
def exp5():
    print("\n" + "="*60)
    print("Exp5: Resource Constraint")
    print("="*60)
    
    for cap in [15, 10, 5]:
        np.random.seed(42)
        env = Env(5)
        sys = PredictiveFCRS(10, cap, 5)
        
        errs = []
        for _ in range(2000):
            errs.append(sys.update(env.gen()))
        
        print(f"Capacity {cap}: error={np.mean(errs):.3f}, reps={len(sys.representations)}")


# ==================== Main ====================
print("FCRS-v5.3 Validation (Compression-Prediction)\n")

exp1()
exp2()
exp3()
exp4()
exp5()

print("\n" + "="*60)
print("All Experiments Complete!")
print("="*60)
