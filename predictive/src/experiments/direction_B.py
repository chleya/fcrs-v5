"""
FCRS-v5 Predictive: 方向B
状态转移模型 P(s'|s,a)

从"被动观察" → "主动干预"
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
    """
    状态转移模型 P(s'|s,a)
    
    核心：学习"行动的后果"
    """
    
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 转移模型: s + a -> s'
        self.W = np.random.randn(state_dim + action_dim, state_dim) * 0.1
        self.b = np.zeros(state_dim)
        
        self.errors = []
    
    def predict(self, state, action):
        """预测下一状态"""
        sa = np.concatenate([state, action])
        next_state = sa @ self.W + self.b
        next_state = np.maximum(0, next_state)  # ReLU
        if np.linalg.norm(next_state) > 1e-6:
            next_state = next_state / np.linalg.norm(next_state)
        return next_state
    
    def update(self, state, action, actual_next):
        """更新模型"""
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
        """心理模拟: 想象执行一系列行动后的状态"""
        states = [state]
        for action in actions:
            next_state = self.predict(states[-1], action)
            states.append(next_state)
        return states


class PredictiveFCRS_B:
    """
    方向B: 带状态转移模型的预测系统
    
    新增：
    - 状态转移模型 P(s'|s,a)
    - 行动空间
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
        
        # 表征池
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
    
    def compress(self, x):
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def generate_action(self):
        """生成随机行动"""
        return np.random.randn(self.action_dim)
    
    def select_action(self, state, possible_actions):
        """选择行动: 使用状态转移模型"""
        # 探索
        if np.random.random() < self.explore:
            return possible_actions[np.random.randint(len(possible_actions))]
        
        # 利用: 选择导致"最有趣"状态的动作
        # "有趣" = 状态变化大
        scores = []
        for action in possible_actions:
            next_state = self.transition.predict(state, action)
            # 评分: 状态变化
            score = np.linalg.norm(next_state - state)
            scores.append(score)
        
        return possible_actions[np.argmax(scores)]
    
    def select_rep(self, x, current_f):
        """选择表征"""
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        # 选择预测误差最小的
        scores = {}
        for i in range(len(self.reps)):
            error = self.rep_predictors[i].get_error()
            scores[i] = -error
        
        return max(scores, key=scores.get)
    
    def forward_with_action(self, x, action=None):
        """执行一步（带行动）"""
        self.step += 1
        
        # 压缩
        f = self.compress(x)
        
        # 更新预测器
        if self.prev_f is not None:
            for i in range(len(self.reps)):
                self.rep_predictors[i].update(self.prev_f, f)
            
            # 更新状态转移模型
            if self.prev_action is not None:
                t_err = self.transition.update(self.prev_f, self.prev_action, f)
                self.transition_errors.append(t_err)
        
        # 选择行动
        if action is None:
            action = self.generate_action()
        
        # 选择表征
        idx = self.select_rep(x, f)
        rep = self.reps[idx]
        
        # 更新表征
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        self.reps[idx] = new_rep
        
        # 记录
        self.prev_f = f
        self.prev_action = action
        
        return {
            'state': f,
            'action': action,
            'rep_idx': idx,
            'transition_error': self.transition.get_error()
        }
    
    def forward(self, x):
        """简化版：不带行动"""
        return self.forward_with_action(x, None)
    
    def imagine_plan(self, initial_state, actions):
        """心理模拟"""
        return self.transition.imagine(initial_state, actions)
    
    def run(self, env, steps, with_actions=True):
        for _ in range(steps):
            x = env.gen()
            action = self.generate_action() if with_actions else None
            self.forward_with_action(x, action)
        
        return self.get_stats()
    
    def get_stats(self):
        pred_errs = [p.get_error() for p in self.rep_predictors.values()]
        return {
            'mean_pred_error': np.mean(pred_errs),
            'transition_error': self.transition.get_error(),
            'pool_size': len(self.reps)
        }


class Env:
    """环境"""
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 实验 ====================
def test_transition_learning():
    """测试状态转移学习"""
    print("=" * 60)
    print("Test B1: State Transition Learning")
    print("=" * 60)
    
    results = {'with_transition': [], 'without': []}
    
    for run in range(10):
        # 有状态转移
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = PredictiveFCRS_B(10, 3, 3, 10, 0.01, 0.1)
        r = sys.run(env, 1000, with_actions=True)
        results['with_transition'].append(r['transition_error'])
        
        # 无状态转移（只有预测器）
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        class SimpleSys:
            def __init__(self):
                self.reps = [np.random.randn(10) for _ in range(3)]
                self.predictors = {i: PerRepPredictor(3, 0.01) for i in range(3)}
                self.prev_f = None
            
            def forward(self, x):
                f = x @ np.random.randn(10, 3) @ np.eye(3) * 0.1
                if self.prev_f is not None:
                    for p in self.predictors.values():
                        p.update(self.prev_f, f)
                self.prev_f = f
            
            def run(self, env, steps):
                for _ in range(steps):
                    self.forward(env.gen())
                return {'transition_error': np.mean([p.get_error() for p in self.predictors.values()])}
        
        sys2 = SimpleSys()
        results['without'].append(sys2.run(env, 1000)['transition_error'])
    
    print(f"\nWith Transition: {np.mean(results['with_transition']):.4f}")
    print(f"Without: {np.mean(results['without']):.4f}")
    
    diff = (np.mean(results['without']) - np.mean(results['with_transition'])) / np.mean(results['without']) * 100
    print(f"改进: {diff:+.1f}%")


def test_action_selection():
    """测试行动选择"""
    print("\n" + "=" * 60)
    print("Test B2: Action Selection")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = PredictiveFCRS_B(10, 3, 3, 10, 0.01, 0.1)
    
    # 运行
    for _ in range(1000):
        x = env.gen()
        sys.forward_with_action(x)
    
    # 心理模拟测试
    test_state = np.random.randn(3)
    test_actions = [np.random.randn(3) for _ in range(5)]
    
    imagined = sys.imagine_plan(test_state, test_actions)
    
    print(f"Initial state norm: {np.linalg.norm(test_state):.3f}")
    print(f"Imagined states: {len(imagined)}")
    print(f"Final state norm: {np.linalg.norm(imagined[-1]):.3f}")
    print(f"Transition error: {sys.transition.get_error():.4f}")


def test_comparison():
    """完整对比"""
    print("\n" + "=" * 60)
    print("Test B3: Full Comparison (方向A vs 方向B)")
    print("=" * 60)
    
    # 方向A
    from direction_A import PredictiveFCRS_A, Env as EnvA
    
    results = {'A': [], 'B': []}
    
    for run in range(5):
        np.random.seed(run * 100)
        env = EnvA(10, 5)
        sys_A = PredictiveFCRS_A(10, 3, 10, 0.01, 0.1)
        sys_A.run(env, 500)
        results['A'].append(sys_A.run(env, 0)['pred_error'])
        
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys_B = PredictiveFCRS_B(10, 3, 3, 10, 0.01, 0.1)
        sys_B.run(env, 500)
        results['B'].append(sys_B.get_stats()['transition_error'])
    
    print(f"方向A (per-rep预测): {np.mean(results['A']):.4f}")
    print(f"方向B (状态转移): {np.mean(results['B']):.4f}")


# ==================== Main ====================
if __name__ == "__main__":
    test_transition_learning()
    test_action_selection()
    test_comparison()
    print("\nDone!")
