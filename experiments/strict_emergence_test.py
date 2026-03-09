"""
真涌现验证实验
目的：验证维度产生是否真正由系统内在动力学驱动
而非参数设置
"""

import numpy as np


class Rep:
    def __init__(self, id, vec, origin='init'):
        self.id = id
        self.vector = vec
        self.origin = origin


class TestSystem:
    """测试系统"""
    def __init__(self, mode='emergent', threshold=0.5, prob=0.3):
        self.mode = mode
        self.threshold = threshold
        self.prob = prob
        self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
        self.births = 0
        self.step = 0
        self.state_hist = []
    
    def forward(self, x):
        self.step += 1
        
        best = max(self.reps, key=lambda r: np.dot(r.vector, x))
        err = np.linalg.norm(best.vector - x)
        best.vector += 0.1 * (x - best.vector)
        
        self.state_hist.append(np.linalg.norm(x))
        
        if self.mode == 'emergent':
            # 临界度驱动
            crit = np.std(self.state_hist[-20:]) if len(self.state_hist) > 20 else 0
            if crit > 0.3 and np.random.random() < self.prob:
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'pert'))
                self.births += 1
        else:
            # 阈值驱动
            if self.step % 100 == 0:
                new_vec = best.vector + np.random.randn(10) * 0.1
                self.reps.append(Rep(len(self.reps), new_vec, 'thresh'))
                self.births += 1
        
        if len(self.reps) > 10:
            self.reps.pop(0)
        
        for i, r in enumerate(self.reps):
            r.id = i
        
        return err


# ==================== 关键实验 ====================

def test_param_sensitivity():
    """
    实验0: 参数敏感性测试
    目的：验证结果差异是否来自参数
    """
    print("="*60)
    print("Test 0: Param Sensitivity")
    print("="*60)
    
    # 同样的"涌现"模式，不同参数
    for prob in [0.1, 0.3, 0.5]:
        np.random.seed(42)
        env_centers = {i: np.random.randn(10)*2 for i in range(5)}
        
        def gen():
            c = env_centers[np.random.randint(0, 5)]
            return c + np.random.randn(10)*0.3
        
        sys = TestSystem(mode='emergent', prob=prob)
        
        for _ in range(2000):
            sys.forward(gen())
        
        print(f"Prob {prob}: births={sys.births}")
    
    # 同样的"阈值"模式，不同间隔
    for interval in [50, 100, 200]:
        np.random.seed(42)
        
        class IntervalSystem(TestSystem):
            def __init__(self, interval):
                super().__init__(mode='threshold')
                self.interval = interval
        
        sys = IntervalSystem(interval)
        
        for _ in range(2000):
            sys.forward(gen())
        
        print(f"Interval {interval}: births={sys.births}")


def test_no_params():
    """
    实验1: 无参数涌现测试
    目的：完全移除人为参数，看系统是否还能"涌现"
    """
    print("\n" + "="*60)
    print("Test 1: No-Parameter Emergence")
    print("="*60)
    
    np.random.seed(42)
    
    class PureEmergent:
        """没有任何参数的纯粹涌现"""
        def __init__(self):
            self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
            self.births = 0
            self.step = 0
            self.errors = []
        
        def forward(self, x):
            self.step += 1
            
            # 简单的选择
            best = max(self.reps, key=lambda r: np.dot(r.vector, x))
            err = np.linalg.norm(best.vector - x)
            self.errors.append(err)
            
            # 学习
            best.vector += 0.1 * (x - best.vector)
            
            # 完全随机生成 - 无任何参数控制
            # 每个样本都有小概率产生新表征
            if np.random.random() < 0.01:  # 1% 固定概率，不是"涌现"
                new_vec = best.vector + np.random.randn(10) * 0.2
                self.reps.append(Rep(len(self.reps), new_vec, 'random'))
                self.births += 1
            
            if len(self.reps) > 10:
                self.reps.pop(0)
            
            for i, r in enumerate(self.reps):
                r.id = i
            
            return err
    
    env_centers = {i: np.random.randn(10)*2 for i in range(5)}
    def gen():
        c = env_centers[np.random.randint(0, 5)]
        return c + np.random.randn(10)*0.3
    
    sys = PureEmergent()
    
    for _ in range(5000):
        sys.forward(gen())
    
    print(f"Pure random (1%): births={sys.births}")
    print(f"Mean error: {np.mean(sys.errors):.3f}")


def test_emergent_vs_random():
    """
    实验2: 涌现 vs 纯随机
    目的：验证我们的"涌现"是否优于纯随机
    """
    print("\n" + "="*60)
    print("Test 2: Emergent vs Random")
    print("="*60)
    
    results = {'emergent': [], 'random': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        # Emergent
        env_centers = {i: np.random.randn(10)*2 for i in range(5)}
        def gen():
            c = env_centers[np.random.randint(0, 5)]
            return c + np.random.randn(10)*0.3
        
        sys1 = TestSystem(mode='emergent', prob=0.3)
        
        for _ in range(2000):
            sys1.forward(gen())
        
        results['emergent'].append(sys1.births)
        
        # Random (相同概率)
        np.random.seed(run * 100)
        env_centers = {i: np.random.randn(10)*2 for i in range(5)}
        
        class RandomSys:
            def __init__(self):
                self.reps = [Rep(i, np.random.randn(10)*0.5) for i in range(3)]
                self.births = 0
            
            def forward(self, x):
                best = max(self.reps, key=lambda r: np.dot(r.vector, x))
                best.vector += 0.1 * (x - best.vector)
                
                # 纯随机，不看任何状态
                if np.random.random() < 0.3:
                    new_vec = best.vector + np.random.randn(10) * 0.2
                    self.reps.append(Rep(len(self.reps), new_vec, 'random'))
                    self.births += 1
                
                if len(self.reps) > 10:
                    self.reps.pop(0)
                
                for i, r in enumerate(self.reps):
                    r.id = i
        
        sys2 = RandomSys()
        
        for _ in range(2000):
            sys2.forward(gen())
        
        results['random'].append(sys2.births)
    
    print(f"Emergent: {np.mean(results['emergent']):.1f} ± {np.std(results['emergent']):.1f}")
    print(f"Random: {np.mean(results['random']):.1f} ± {np.std(results['random']):.1f}")
    
    if np.abs(np.mean(results['emergent']) - np.mean(results['random'])) < 50:
        print("[WARNING] Emergent ≈ Random! No real difference!")
    else:
        print("[OK] Significant difference detected")


def test_adaptation():
    """
    实验3: 适应性测试
    目的：验证系统是否能适应环境变化
    """
    print("\n" + "="*60)
    print("Test 3: Adaptation")
    print("="*60)
    
    np.random.seed(42)
    
    # 初始环境: 5类
    env = {i: np.random.randn(10)*2 for i in range(5)}
    
    def generate():
        c = env[np.random.randint(0, len(env))]
        return c + np.random.randn(10)*0.3
    
    sys = TestSystem(mode='emergent', prob=0.3)
    
    # Phase 1: 5类
    for _ in range(1000):
        sys.forward(generate())
    
    births_p1 = sys.births
    print(f"Phase 1 (5 classes): {births_p1} births")
    
    # Phase 2: 变化为10类
    env = {i: np.random.randn(10)*2 for i in range(10)}
    
    for _ in range(1000):
        sys.forward(generate())
    
    births_p2 = sys.births - births_p1
    print(f"Phase 2 (10 classes): {births_p2} births")
    
    # Phase 3: 回到5类
    env = {i: np.random.randn(10)*2 for i in range(5)}
    
    for _ in range(1000):
        sys.forward(generate())
    
    births_p3 = sys.births - births_p1 - births_p2
    print(f"Phase 3 (5 classes): {births_p3} births")
    
    # 分析
    print(f"\nAdaptation: P1={births_p1}, P2={births_p2}, P3={births_p3}")


# ==================== Main ====================
print("Strict Emergence Validation Tests\n")

test_param_sensitivity()
test_no_params()
test_emergent_vs_random()
test_adaptation()

print("\n" + "="*60)
print("All Tests Complete")
print("="*60)
