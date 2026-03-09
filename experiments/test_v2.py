"""
验证实验 - 简化版
"""

import numpy as np


class Rep:
    def __init__(self, id, vector, origin='initial'):
        self.id = id
        self.vector = vector
        self.origin = origin


class SimpleFCRS:
    """简化FCRS"""
    
    def __init__(self, emergent=True, capacity=10):
        self.emergent = emergent
        self.capacity = capacity
        self.reps = [Rep(i, np.random.randn(10) * 0.5) for i in range(3)]
        self.births = 0
        self.birth_times = []
        self.step_count = 0
    
    def forward(self, x):
        self.step_count += 1
        
        # 选择最佳
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        error = np.linalg.norm(best.vector - x)
        
        # 学习
        best.vector += 0.1 * (x - best.vector)
        
        # 涌现产生
        if self.emergent:
            if np.random.random() < 0.3:
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'perturbation'))
                self.births += 1
                self.birth_times.append(self.step_count)
        else:
            # 优化模式
            if self.step_count % 200 == 0:
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'threshold'))
                self.births += 1
                self.birth_times.append(self.step_count)
        
        # 容量限制
        if len(self.reps) > self.capacity:
            self.reps.pop(0)
        
        return error


class Env:
    def __init__(self, n_classes=5):
        self.centers = {i: np.random.randn(10) * 2 for i in range(n_classes)}
    
    def generate(self):
        cls = np.random.randint(0, len(self.centers))
        return self.centers[cls] + np.random.randn(10) * 0.3


# ==================== 实验 ====================
def test1():
    print("="*60)
    print("Test 1: Emergent vs Optimized")
    print("="*60)
    
    results = {'emergent': 0, 'optimized': 0}
    
    for mode in ['emergent', 'optimized']:
        for run in range(10):
            np.random.seed(run * 100)
            env = Env(5)
            fcrs = SimpleFCRS(emergent=(mode == 'emergent'))
            
            for _ in range(1000):
                x = env.generate()
                fcrs.forward(x)
            
            results[mode] += fcrs.births
    
    print(f"Emergent births: {results['emergent']/10:.1f}")
    print(f"Optimized births: {results['optimized']/10:.1f}")


def test2():
    print("\n" + "="*60)
    print("Test 2: Criticality Correlation")
    print("="*60)
    
    np.random.seed(42)
    env = Env(5)
    fcrs = SimpleFCRS(emergent=True)
    
    for _ in range(1000):
        x = env.generate()
        fcrs.forward(x)
    
    print(f"Total births: {fcrs.births}")
    print(f"Origins: {[r.origin for r in fcrs.reps]}")


def test3():
    print("\n" + "="*60)
    print("Test 3: Resource Constraint")
    print("="*60)
    
    for cap in [15, 10, 5]:
        np.random.seed(42)
        env = Env(5)
        fcrs = SimpleFCRS(emergent=True, capacity=cap)
        
        for _ in range(1000):
            x = env.generate()
            fcrs.forward(x)
        
        print(f"Capacity {cap}: births={fcrs.births}, final={len(fcrs.reps)}")


# ==================== Main ====================
def main():
    test1()
    test2()
    test3()
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
