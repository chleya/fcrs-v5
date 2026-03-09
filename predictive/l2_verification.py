"""
L2验证实验: 预测外推能力
测试系统的泛化能力
"""

import numpy as np
from direction_A import PredictiveFCRS_A
from direction_B import PredictiveFCRS_B


class Env:
    """环境"""
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.n_classes = n_classes
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3
    
    def change_classes(self, new_n_classes):
        """改变环境类别"""
        self.n_classes = new_n_classes
        self.centers = {i: np.random.randn(self.dim) * 2 for i in range(new_n_classes)}


class L2Env:
    """L2专用环境：训练/测试分离"""
    
    def __init__(self, dim=10, train_classes=[0,1,2], test_classes=[3,4]):
        self.dim = dim
        self.train_classes = train_classes
        self.test_classes = test_classes
        
        self.train_centers = {i: np.random.randn(dim) * 2 for i in train_classes}
        self.test_centers = {i: np.random.randn(dim) * 2 for i in test_classes}
    
    def generate_train(self):
        c = self.train_centers[np.random.choice(self.train_classes)]
        return c + np.random.randn(self.dim) * 0.3
    
    def generate_test(self):
        c = self.test_centers[np.random.choice(self.test_classes)]
        return c + np.random.randn(self.dim) * 0.3


def test_L2_basic():
    """L2基础测试: 训练集 vs 测试集"""
    print("=" * 60)
    print("L2 Test 1: Train vs Test Generalization")
    print("=" * 60)
    
    results = {'train': [], 'test': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        env = L2Env(10, [0,1,2], [3,4])
        
        sys = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        
        for _ in range(1000):
            x = env.generate_train()
            sys.forward(x)
        
        train_error = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
        results['train'].append(train_error)
        
        test_errors = []
        for _ in range(200):
            x = env.generate_test()
            f = sys.compress(x)
            idx = sys.select_by_prediction(x, f)
            rep = sys.reps[idx]
            err = np.linalg.norm(rep - x)
            test_errors.append(err)
        
        results['test'].append(np.mean(test_errors))
    
    print(f"\n训练集误差: {np.mean(results['train']):.4f}")
    print(f"测试集误差: {np.mean(results['test']):.4f}")
    
    gap = (np.mean(results['test']) - np.mean(results['train'])) / np.mean(results['train']) * 100
    print(f"泛化差距: {gap:+.1f}%")
    
    return results


def test_L2_change():
    """L2测试: 环境变化"""
    print("\n" + "=" * 60)
    print("L2 Test 2: Environment Change")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
    
    # Phase 1
    for _ in range(500):
        sys.forward(env.gen())
    p1 = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
    print(f"Phase 1 (5类): {p1:.4f}")
    
    # Phase 2
    env.change_classes(10)
    for _ in range(500):
        sys.forward(env.gen())
    p2 = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
    print(f"Phase 2 (10类): {p2:.4f}")
    
    # Phase 3
    env.change_classes(5)
    for _ in range(500):
        sys.forward(env.gen())
    p3 = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
    print(f"Phase 3 (5类): {p3:.4f}")
    
    print(f"\n适应: P1→P2: {(p2-p1)/p1*100:+.1f}%, P2→P3: {(p3-p2)/p2*100:+.1f}%")
    
    return {'p1': p1, 'p2': p2, 'p3': p3}


def test_L2_long_term():
    """L2测试: 长期预测"""
    print("\n" + "=" * 60)
    print("L2 Test 3: Long-term Prediction")
    print("=" * 60)
    
    results = []
    
    for steps in [100, 300, 500]:
        np.random.seed(42)
        env = Env(10, 5)
        sys = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        
        for _ in range(steps):
            sys.forward(env.gen())
        
        err = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
        results.append((steps, err))
        print(f"训练{steps}步: 误差={err:.4f}")
    
    return results


def test_L2_AB_compare():
    """L2测试: 方向A vs 方向B"""
    print("\n" + "=" * 60)
    print("L2 Test 4: Direction A vs B")
    print("=" * 60)
    
    results = {'A': [], 'B': []}
    
    for run in range(5):
        # A
        np.random.seed(run * 100)
        env = L2Env(10, [0,1,2], [3,4])
        
        sys_A = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        for _ in range(500):
            sys_A.forward(env.generate_train())
        
        errs_A = []
        for _ in range(100):
            x = env.generate_test()
            f = sys_A.compress(x)
            idx = sys_A.select_by_prediction(x, f)
            errs_A.append(np.linalg.norm(sys_A.reps[idx] - x))
        results['A'].append(np.mean(errs_A))
        
        # B
        np.random.seed(run * 100)
        env = L2Env(10, [0,1,2], [3,4])
        
        sys_B = PredictiveFCRS_B(10, 3, 3, 10, 0.01, 0.1)
        for _ in range(500):
            x = env.generate_train()
            sys_B.forward_with_action(x)
        
        errs_B = []
        for _ in range(100):
            x = env.generate_test()
            f = sys_B.compress(x)
            idx = sys_B.select_rep(x, f)
            errs_B.append(np.linalg.norm(sys_B.reps[idx] - x))
        results['B'].append(np.mean(errs_B))
    
    print(f"\n方向A 测试误差: {np.mean(results['A']):.4f}")
    print(f"方向B 测试误差: {np.mean(results['B']):.4f}")
    
    return results


# ==================== Main ====================
if __name__ == "__main__":
    print("L2 Verification: Prediction Extrapolation\n")
    
    test_L2_basic()
    test_L2_change()
    test_L2_long_term()
    test_L2_AB_compare()
    
    print("\n" + "=" * 60)
    print("L2 Complete!")
    print("=" * 60)
