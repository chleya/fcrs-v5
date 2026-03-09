"""
FCRS-v5 Predictive: 整合方向A和B
完整系统：每个表征有自己的预测器 + 状态转移模型
"""

import numpy as np


class PerRepPredictor:
    """每个表征的预测器"""
    
    def __init__(self, compress_dim, lr=0.01):
        self.compress_dim = compress_dim
        self.lr = lr
        self.W = np.eye(compress_dim) * 0.5
        self.b = np.zeros(compress_dim)
        self.errors = []
    
    def predict(self, f):
        return f @ self.W + self.b
    
    def update(self, f_curr, f_next):
        pred = self.predict(f_curr)
        error = f_next - pred
        self.errors.append(np.linalg.norm(error))
        self.W += self.lr * np.outer(f_curr, error)
        self.b += self.lr * error
        return np.linalg.norm(error)
    
    def get_error(self):
        return np.mean(self.errors[-5:]) if len(self.errors) >= 3 else 1.0


class StateTransitionModel:
    """状态转移模型 P(s'|s,a)"""
    
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.W = np.random.randn(state_dim + action_dim, state_dim) * 0.1
        self.b = np.zeros(state_dim)
        self.errors = []
    
    def predict(self, state, action):
        sa = np.concatenate([state, action])
        next_state = sa @ self.W + self.b
        next_state = np.maximum(0, next_state)
        if np.linalg.norm(next_state) > 1e-6:
            next_state = next_state / np.linalg.norm(next_state)
        return next_state
    
    def update(self, state, action, actual_next):
        predicted = self.predict(state, action)
        error = actual_next - predicted
        self.errors.append(np.linalg.norm(error))
        sa = np.concatenate([state, action])
        self.W += self.lr * np.outer(sa, error)
        self.b += self.lr * error
        return np.linalg.norm(error)
    
    def get_error(self):
        return np.mean(self.errors[-10:]) if len(self.errors) >= 3 else 1.0
    
    def imagine(self, state, actions):
        states = [state]
        for action in actions:
            states.append(self.predict(states[-1], action))
        return states


class IntegratedSystem:
    """
    整合系统：方向A + 方向B
    
    - 每个表征有自己的预测器 (方向A)
    - 状态转移模型 P(s'|s,a) (方向B)
    - 心理模拟能力
    """
    
    def __init__(self, input_dim=10, compress_dim=5, action_dim=3, 
                 capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        # 编码器
        self.W_enc = np.random.randn(input_dim, compress_dim) * 0.1
        self.b_enc = np.zeros(compress_dim)
        
        # 表征池 + 每个表征的预测器
        self.reps = []
        self.rep_predictors = {}
        
        for i in range(3):
            rep = np.random.randn(input_dim)
            rep = rep / (np.linalg.norm(rep) + 1e-8)
            self.reps.append(rep)
            self.rep_predictors[i] = PerRepPredictor(compress_dim, lr)
        
        # 状态转移模型
        self.transition = StateTransitionModel(compress_dim, action_dim, lr)
        
        # 历史
        self.step = 0
        self.prev_f = None
        self.prev_action = None
        
        self.pred_errors = []
        self.transition_errors = []
        self.recon_errors = []
    
    def compress(self, x):
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def generate_action(self):
        return np.random.randn(self.action_dim)
    
    def select_rep_by_prediction(self, x, current_f):
        """方向A: 基于每个表征的预测误差选择"""
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        scores = {}
        for i in range(len(self.reps)):
            error = self.rep_predictors[i].get_error()
            scores[i] = -error
        
        return max(scores, key=scores.get)
    
    def select_action_by_transition(self, state):
        """方向B: 基于状态转移选择行动"""
        if np.random.random() < self.explore:
            return self.generate_action()
        
        # 选择导致最大状态变化的动作
        possible_actions = [self.generate_action() for _ in range(5)]
        scores = []
        
        for action in possible_actions:
            next_state = self.transition.predict(state, action)
            score = np.linalg.norm(next_state - state)
            scores.append(score)
        
        return possible_actions[np.argmax(scores)]
    
    def forward(self, x, action=None):
        """执行一步"""
        self.step += 1
        
        # 压缩
        f = self.compress(x)
        
        # 更新预测器 (方向A)
        if self.prev_f is not None:
            for i in range(len(self.reps)):
                self.rep_predictors[i].update(self.prev_f, f)
            
            # 更新状态转移模型 (方向B)
            if self.prev_action is not None:
                t_err = self.transition.update(self.prev_f, self.prev_action, f)
                self.transition_errors.append(t_err)
        
        # 选择行动 (方向B)
        if action is None:
            action = self.select_action_by_transition(f)
        
        # 选择表征 (方向A)
        idx = self.select_rep_by_prediction(x, f)
        rep = self.reps[idx]
        
        # 重构误差
        recon_err = np.linalg.norm(rep - x)
        self.recon_errors.append(recon_err)
        
        # 更新表征
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        
        # 新表征加入预测器
        if len(self.reps) < self.capacity:
            self.reps.append(new_rep)
            new_idx = len(self.reps) - 1
            self.rep_predictors[new_idx] = PerRepPredictor(self.compress_dim, self.lr)
        else:
            self.reps[idx] = new_rep
        
        # 记录
        self.prev_f = f
        self.prev_action = action
        
        return {
            'recon_error': recon_err,
            'transition_error': self.transition.get_error(),
            'mean_pred_error': np.mean([p.get_error() for p in self.rep_predictors.values()])
        }
    
    def imagine_plan(self, state, actions):
        """心理模拟"""
        return self.transition.imagine(state, actions)
    
    def run(self, env, steps):
        for _ in range(steps):
            self.forward(env.gen())
        return self.get_stats()
    
    def get_stats(self):
        return {
            'recon_error': np.mean(self.recon_errors[-100:]) if self.recon_errors else 0,
            'transition_error': self.transition.get_error(),
            'mean_pred_error': np.mean([p.get_error() for p in self.rep_predictors.values()]),
            'pool_size': len(self.reps)
        }


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 测试 ====================
def test_integrated():
    """测试整合系统"""
    print("=" * 60)
    print("Test: Integrated System (A + B)")
    print("=" * 60)
    
    results = {'integrated': [], 'A_only': [], 'B_only': []}
    
    for run in range(10):
        # 整合系统
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = IntegratedSystem(10, 3, 3, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        results['integrated'].append(r['recon_error'])
        
        # 方向A单独
        np.random.seed(run * 100)
        env = Env(10, 5)
        from direction_A import PredictiveFCRS_A
        sys_A = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        r_A = sys_A.run(env, 1000)
        results['A_only'].append(r_A['recon_error'])
        
        # 方向B单独
        np.random.seed(run * 100)
        env = Env(10, 5)
        from direction_B import PredictiveFCRS_B
        sys_B = PredictiveFCRS_B(10, 3, 3, 10, 0.01, 0.1)
        sys_B.run(env, 1000)
        results['B_only'].append(sys_B.get_stats()['transition_error'])
    
    print(f"\n整合系统: {np.mean(results['integrated']):.4f}")
    print(f"方向A单独: {np.mean(results['A_only']):.4f}")
    print(f"方向B单独: {np.mean(results['B_only']):.4f}")
    
    # 改进
    imp_A = (np.mean(results['A_only']) - np.mean(results['integrated'])) / np.mean(results['A_only']) * 100
    imp_B = (np.mean(results['B_only']) - np.mean(results['integrated'])) / np.mean(results['B_only']) * 100
    
    print(f"\n整合 vs A: {imp_A:+.1f}%")
    print(f"整合 vs B: {imp_B:+.1f}%")
    
    return results


def test_components():
    """测试各组件"""
    print("\n" + "=" * 60)
    print("Test: Component Analysis")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = IntegratedSystem(10, 3, 3, 10, 0.01, 0.1)
    
    for _ in range(1000):
        sys.forward(env.gen())
    
    stats = sys.get_stats()
    
    print(f"\n重构误差: {stats['recon_error']:.4f}")
    print(f"转移误差: {stats['transition_error']:.4f}")
    print(f"平均预测误差: {stats['mean_pred_error']:.4f}")
    print(f"表征池大小: {stats['pool_size']}")
    
    # 心理模拟测试
    test_state = np.random.randn(3)
    test_actions = [np.random.randn(3) for _ in range(3)]
    imagined = sys.imagine_plan(test_state, test_actions)
    
    print(f"\n心理模拟:")
    print(f"  初始状态: {np.linalg.norm(test_state):.3f}")
    print(f"  想象状态数: {len(imagined)}")
    print(f"  最终状态: {np.linalg.norm(imagined[-1]):.3f}")


def test_L2():
    """L2测试"""
    print("\n" + "=" * 60)
    print("Test: L2 Generalization")
    print("=" * 60)
    
    class L2Env:
        def __init__(self):
            self.train_c = {i: np.random.randn(10) * 2 for i in [0,1,2]}
            self.test_c = {i: np.random.randn(10) * 2 for i in [3,4]}
        
        def gen(self, test=False):
            c = self.test_c[np.random.choice([3,4])] if test else self.train_c[np.random.choice([0,1,2])]
            return c + np.random.randn(10) * 0.3
    
    results = {'train': [], 'test': []}
    
    for run in range(5):
        np.random.seed(run * 100)
        env = L2Env()
        
        sys = IntegratedSystem(10, 3, 3, 10, 0.01, 0.1)
        
        # 训练
        for _ in range(500):
            sys.forward(env.gen(test=False))
        
        train_err = np.mean(sys.recon_errors[-100:]) if sys.recon_errors else 0
        results['train'].append(train_err)
        
        # 测试
        test_errs = []
        for _ in range(100):
            err = sys.forward(env.gen(test=True))['recon_error']
            test_errs.append(err)
        results['test'].append(np.mean(test_errs))
    
    print(f"\n训练集: {np.mean(results['train']):.4f}")
    print(f"测试集: {np.mean(results['test']):.4f}")
    gap = (np.mean(results['test']) - np.mean(results['train'])) / np.mean(results['train']) * 100
    print(f"泛化差距: {gap:+.1f}%")


# ==================== Main ====================
if __name__ == "__main__":
    test_integrated()
    test_components()
    test_L2()
    print("\nDone!")
