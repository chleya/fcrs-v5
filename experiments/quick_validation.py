"""
验证实验 - 直接测试
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Rep:
    id: int
    vector: np.ndarray
    origin: str = 'initial'


class SimpleFCRS:
    """简化FCRS"""
    
    def __init__(self, emergent=True):
        self.emergent = emergent
        self.reps = [Rep(i, np.random.randn(10) * 0.5) for i in range(3)]
        self.births = 0
        self.birth_times = []
        self.step = 0
    
    def step(self, x):
        self.step += 1
        
        # 选择
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        error = np.linalg.norm(best.vector - x)
        
        # 更新
        best.vector += 0.1 * (x - best.vector)
        
        if self.emergent:
            # 涌现模式 - 概率产生
            if np.random.random() < 0.3:  # 30%概率
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'perturbation'))
                self.births += 1
                self.birth_times.append(self.step)
        else:
            # 优化模式 - 固定阈值
            if self.step % 200 == 0:
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'threshold'))
                self.births += 1
                self.birth_times.append(self.step)
        
        # 容量限制
        if len(self.reps) > 10:
            self.reps.pop(0)
        
        return error


class Env:
    def __init__(self, n_classes=5):
        self.centers = {i: np.random.randn(10) * 2 for i in range(n_classes)}
    
    def generate(self):
        cls = np.random.randint(0, len(self.centers))
        return self.centers[cls] + np.random.randn(10) * 0.3


def test_emergent_vs_optimized():
    print("="*60)
    print("Test 1: Emergent vs Optimized")
    print("="*60)
    
    results = {'emergent': [], 'optimized': []}
    
    for mode in ['emergent', 'optimized']:
        for run in range(10):
            np.random.seed(run * 100)
            env = Env(5)
            fcrs = SimpleFCRS(emergent=(mode == 'emergent'))
            
            for _ in range(1000):
                x = env.generate()
                fcrs.step(x)
            
            results[mode].append(fcrs.births)
    
    print(f"Emergent: {np.mean(results['emergent']):.1f} births")
    print(f"Optimized: {np.mean(results['optimized']):.1f} births")


def test_criticality():
    print("\n" + "="*60)
    print("Test 2: Criticality Correlation")
    print("="*60)
    
    env = Env(5)
    fcrs = SimpleFCRS(emergent=True)
    
    crit_history = []
    
    for step in range(1000):
        x = env.generate()
        
        # 模拟临界度
        crit = 0.5 + 0.3 * np.sin(step * 0.01) + np.random.randn() * 0.1
        crit_history.append(crit)
        
        fcrs.step(x)
    
    print(f"Total births: {fcrs.births}")
    print(f"Mean criticality: {np.mean(crit_history):.3f}")
    
    if fcrs.births > 0:
        birth_crit = [crit_history[min(t, len(crit_history)-1)] for t in fcrs.birth_times]
        print(f"Mean criticality at birth: {np.mean(birth_crit):.3f}")


def test_probation():
    print("\n" + "="*60)
    print("Test 3: Probation Screening")
    print("="*60)
    
    env = Env(5)
    fcrs = SimpleFCRS(emergent=True)
    
    for _ in range(2000):
        x = env.generate()
        fcrs.step(x)
    
    origins = [r.origin for r in fcrs.reps]
    print(f"Final reps: {len(fcrs.reps)}")
    print(f"Origins: {origins}")
    print(f"Total births: {fcrs.births}")


def test_resource():
    print("\n" + "="*60)
    print("Test 4: Resource Constraint")
    print("="*60)
    
    for capacity in [15, 10, 5]:
        np.random.seed(42)
        env = Env(5)
        
        class LimitedFCRS(SimpleFCRS):
            def __init__(self):
                super().__init__(emergent=True)
                self.capacity = capacity
            
            def step(self, x):
                error = super().step(x)
                if len(self.reps) > self.capacity:
                    self.reps.pop(0)
                return error
        
        fcrs = LimitedFCRS()
        
        for _ in range(1000):
            x = env.generate()
            fcrs.step(x)
        
        print(f"Capacity {capacity}: births={fcrs.births}, final={len(fcrs.reps)}")


def main():
    test_emergent_vs_optimized()
    test_criticality()
    test_probation()
    test_resource()
    
    print("\n" + "="*60)
    print("All Tests Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
