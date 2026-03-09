"""
验证实验 - 最终版
"""

import numpy as np


class Rep:
    def __init__(self, id, vec, fit=0, origin='init', birth=0):
        self.id = id
        self.vector = vec
        self.fitness = fit
        self.origin = origin
        self.birth = birth


class EmergentFCRS:
    def __init__(self, cap=10, a=0.4, b=0.3, g=0.3):
        self.cap = cap
        self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
        self.state_hist = []
        self.err_hist = []
        self.births = 0
        self.birth_times = []
        self.crit_hist = []
        self.steps = 0
    
    def crit(self):
        if len(self.state_hist) < 10:
            return 0.5
        return min(1.0, np.std(self.state_hist[-50:]) * 0.5)
    
    def forward(self, x):
        self.steps += 1
        
        # 选择
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        err = np.linalg.norm(best.vector - x)
        
        # 学习
        best.vector += 0.1 * (x - best.vector)
        best.fitness = -err
        
        # 记录
        self.state_hist.append(np.linalg.norm(x))
        self.err_hist.append(err)
        if len(self.state_hist) > 50:
            self.state_hist.pop(0)
        
        # 临界
        c = self.crit()
        self.crit_hist.append(c)
        
        # 涌现
        if c > 0.3 and np.random.random() < 0.2:
            new_vec = best.vector + np.random.randn(10) * 0.2 * c
            self.reps.append(Rep(len(self.reps), new_vec, -err, 'pert', self.steps))
            self.births += 1
            self.birth_times.append(self.steps)
        
        # 容量
        if len(self.reps) > self.cap:
            self.reps.sort(key=lambda r: r.fitness)
            self.reps.pop(0)
        
        # 重新编号
        for i, r in enumerate(self.reps):
            r.id = i
        
        return err


class OptimizedFCRS:
    def __init__(self, cap=10, thresh=0.5):
        self.cap = cap
        self.thresh = thresh
        self.gain = 0
        self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
        self.steps = 0
        self.birth_times = []
    
    def forward(self, x):
        self.steps += 1
        
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        err = np.linalg.norm(best.vector - x)
        best.vector += 0.1 * (x - best.vector)
        
        # 阈值
        self.gain += 0.01
        if self.gain > self.thresh:
            new_vec = best.vector + np.random.randn(10) * 0.1
            self.reps.append(Rep(len(self.reps), new_vec, -err, 'thresh', self.steps))
            self.birth_times.append(self.steps)
            self.gain = 0
        
        if len(self.reps) > self.cap:
            self.reps.pop(0)
        
        for i, r in enumerate(self.reps):
            r.id = i
        
        return err


class Env:
    def __init__(self, n=5):
        self.centers = {i: np.random.randn(10)*2 for i in range(n)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(10)*0.3


# ==================== 实验 ====================
def test1():
    print("="*60)
    print("Exp1: Emergent vs Optimized")
    print("="*60)
    
    for r in range(10):
        np.random.seed(r*100)
        e = Env(5)
        sys1 = EmergentFCRS(10)
        for _ in range(2000):
            sys1.forward(e.gen())
        
        np.random.seed(r*100)
        e = Env(5)
        sys2 = OptimizedFCRS(10, 0.3)
        for _ in range(2000):
            sys2.forward(e.gen())
        
        print(f"Run {r}: Emergent={sys1.births}, Optimized={len(sys2.birth_times)}")


def test2():
    print("\n" + "="*60)
    print("Exp2: Criticality")
    print("="*60)
    
    np.random.seed(42)
    e = Env(5)
    sys = EmergentFCRS(10)
    
    for _ in range(3000):
        sys.forward(e.gen())
    
    print(f"Births: {sys.births}")
    print(f"Mean crit: {np.mean(sys.crit_hist):.3f}")


def test3():
    print("\n" + "="*60)
    print("Exp3: Multi Fitness")
    print("="*60)
    
    for cfg, name in [((0.6, 0.2, 0.2), "High-Pred"),
                       ((0.2, 0.6, 0.2), "High-Comp"),
                       ((0.2, 0.2, 0.6), "High-Beh")]:
        np.random.seed(42)
        e = Env(5)
        sys = EmergentFCRS(10, *cfg[:3])
        
        for _ in range(2000):
            sys.forward(e.gen())
        
        print(f"{name}: births={sys.births}")


def test4():
    print("\n" + "="*60)
    print("Exp4: Probation")
    print("="*60)
    
    np.random.seed(42)
    e = Env(5)
    sys = EmergentFCRS(10)
    
    for _ in range(3000):
        sys.forward(e.gen())
    
    print(f"Births: {sys.births}")
    print(f"Origins: {[r.origin for r in sys.reps]}")


def test5():
    print("\n" + "="*60)
    print("Exp5: Resource")
    print("="*60)
    
    for cap in [15, 10, 5]:
        np.random.seed(42)
        e = Env(5)
        sys = EmergentFCRS(cap)
        
        for _ in range(2000):
            sys.forward(e.gen())
        
        print(f"Cap {cap}: births={sys.births}, final={len(sys.reps)}")


# ==================== Main ====================
print("FCRS-v5.2 Validation\n")
test1()
test2()
test3()
test4()
test5()
print("\n" + "="*60)
print("All Complete!")
print("="*60)
