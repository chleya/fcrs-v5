"""
FCRS-v5 Predictive: 优化 + L3因果推理
方向2: 系统优化
方向3: 因果推理能力
"""

import numpy as np


class PerRepPredictor:
    def __init__(self, compress_dim, lr=0.01):
        self.compress_dim = compress_dim
        self.lr = lr
        self.W = np.eye(compress_dim) * 0.5
        self.b = np.zeros(compress_dim)
        self.errors = []
        self.momentum = np.zeros_like(self.W)
    
    def predict(self, f):
        return f @ self.W + self.b
    
    def update(self, f_curr, f_next):
        pred = self.predict(f_curr)
        error = f_next - pred
        self.errors.append(np.linalg.norm(error))
        
        # 动量更新
        grad = np.outer(f_curr, error)
        self.momentum = 0.9 * self.momentum + self.lr * grad
        self.W += self.momentum
        self.b += self.lr * error
        
        return np.linalg.norm(error)
    
    def get_error(self):
        return np.mean(self.errors[-5:]) if len(self.errors) >= 3 else 1.0


class StateTransitionModel:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.W = np.random.randn(state_dim + action_dim, state_dim) * 0.1
        self.b = np.zeros(state_dim)
        self.errors = []
        self.causal_graph = {}  # 因果图
    
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
        
        # 记录因果关系
        self._update_causal_graph(state, action, actual_next)
        
        return np.linalg.norm(error)
    
    def _update_causal_graph(self, state, action, next_state):
        """更新因果图"""
        # 简化：记录action对next_state的影响
        action_effect = np.dot(action, next_state)
        key = tuple(np.round(action, 2))
        if key not in self.causal_graph:
            self.causal_graph[key] = []
        self.causal_graph[key].append(action_effect)
    
    def get_causal_effect(self, action):
        """获取行动的因果效应"""
        key = tuple(np.round(action, 2))
        if key in self.causal_graph and self.causal_graph[key]:
            return np.mean(self.causal_graph[key])
        return 0.0
    
    def imagine(self, state, actions):
        states = [state]
        for action in actions:
            states.append(self.predict(states[-1], action))
        return states
    
    def get_error(self):
        return np.mean(self.errors[-10:]) if len(self.errors) >= 3 else 1.0


class OptimizedSystem:
    """优化版系统"""
    
    def __init__(self, input_dim=10, compress_dim=5, action_dim=3, 
                 capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        # 编码器 (带BatchNorm风格)
        self.W_enc = np.random.randn(input_dim, compress_dim) * 0.1
        self.b_enc = np.zeros(compress_dim)
        self.running_mean = np.zeros(compress_dim)
        self.running_var = np.ones(compress_dim)
        
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
        self.recon_errors = []
        
        # 因果记录
        self.causal_records = []
    
    def compress(self, x, training=True):
        """带归一化的压缩"""
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)  # ReLU
        
        if training:
            # 更新运行统计
            self.running_mean = 0.9 * self.running_mean + 0.1 * np.mean(f)
            self.running_var = 0.9 * self.running_var + 0.1 * np.var(f)
        
        # 归一化
        f = (f - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)
        
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def generate_action(self):
        return np.random.randn(self.action_dim)
    
    def select_rep_by_prediction(self, x, current_f):
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        scores = {}
        for i in range(len(self.reps)):
            error = self.rep_predictors[i].get_error()
            scores[i] = -error
        
        return max(scores, key=scores.get)
    
    def select_action_by_transition(self, state):
        if np.random.random() < self.explore:
            return self.generate_action()
        
        possible_actions = [self.generate_action() for _ in range(5)]
        scores = []
        
        for action in possible_actions:
            next_state = self.transition.predict(state, action)
            score = np.linalg.norm(next_state - state)
            # 加入因果效应
            causal_effect = self.transition.get_causal_effect(action)
            score += 0.1 * causal_effect
            scores.append(score)
        
        return possible_actions[np.argmax(scores)]
    
    def forward(self, x, action=None):
        self.step += 1
        
        f = self.compress(x)
        
        if self.prev_f is not None:
            for i in range(len(self.reps)):
                self.rep_predictors[i].update(self.prev_f, f)
            
            if self.prev_action is not None:
                t_err = self.transition.update(self.prev_f, self.prev_action, f)
                self.transition_errors.append(t_err)
        
        if action is None:
            action = self.select_action_by_transition(f)
        
        idx = self.select_rep_by_prediction(x, f)
        rep = self.reps[idx]
        
        recon_err = np.linalg.norm(rep - x)
        self.recon_errors.append(recon_err)
        
        new_rep = rep + self.lr * (x - rep)
        new_rep = new_rep / (np.linalg.norm(new_rep) + 1e-8)
        
        if len(self.reps) < self.capacity:
            self.reps.append(new_rep)
            new_idx = len(self.reps) - 1
            self.rep_predictors[new_idx] = PerRepPredictor(self.compress_dim, self.lr)
        else:
            self.reps[idx] = new_rep
        
        self.prev_f = f
        self.prev_action = action
        
        # 记录因果
        self.causal_records.append({
            'state': f.copy(),
            'action': action.copy(),
            'next_state': None  # 下一时刻记录
        })
        
        return {
            'recon_error': recon_err,
            'transition_error': self.transition.get_error(),
            'mean_pred_error': np.mean([p.get_error() for p in self.rep_predictors.values()])
        }
    
    def imagine_plan(self, state, actions):
        return self.transition.imagine(state, actions)
    
    def intervene(self, state, action):
        """干预：强制执行某个行动，观察结果"""
        # 记录干预前的预测
        predicted = self.transition.predict(state, action)
        
        # 实际执行（在真实环境中）
        # 这里返回预测结果作为"假设"
        return predicted
    
    def counterfactual(self, state, action_a, action_b):
        """反事实：如果执行action_b而非action_a，结果会怎样？"""
        # 预测action_a的结果
        result_a = self.transition.predict(state, action_a)
        
        # 预测action_b的结果  
        result_b = self.transition.predict(state, action_b)
        
        return {
            'if_did_a': result_a,
            'if_did_b': result_b,
            'difference': result_b - result_a
        }
    
    def why(self, state, action, outcome):
        """回答"为什么"：解释为什么这个行动导致这个结果"""
        # 找到最相似的历史状态
        similarities = []
        for record in self.causal_records:
            if record['next_state'] is not None:
                sim = np.dot(state, record['state'])
                similarities.append((sim, record))
        
        similarities.sort(reverse=True)
        
        # 返回最相似的原因
        if similarities:
            cause = similarities[0][1]
            return {
                'similar_state': cause['state'],
                'action_taken': cause['action'],
                'outcome': cause['next_state'],
                'explanation': '因为在相似情况下，采取相同行动导致了类似结果'
            }
        
        return {'explanation': ' insufficient data'}
    
    def run(self, env, steps):
        for _ in range(steps):
            self.forward(env.gen())
        return self.get_stats()
    
    def get_stats(self):
        return {
            'recon_error': np.mean(self.recon_errors[-100:]) if self.recon_errors else 0,
            'transition_error': self.transition.get_error(),
            'mean_pred_error': np.mean([p.get_error() for p in self.rep_predictors.values()]),
            'pool_size': len(self.reps),
            'causal_links': len(self.transition.causal_graph)
        }


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 测试 ====================
def test_optimization():
    """测试优化版"""
    print("=" * 60)
    print("Test: Optimized System")
    print("=" * 60)
    
    results = {'optimized': [], 'basic': []}
    
    for run in range(10):
        # Optimized
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys = OptimizedSystem(10, 3, 3, 10, 0.01, 0.1)
        r = sys.run(env, 1000)
        results['optimized'].append(r['recon_error'])
        
        # Basic
        np.random.seed(run * 100)
        env = Env(10, 5)
        from integrated import IntegratedSystem
        sys2 = IntegratedSystem(10, 3, 3, 10, 0.01, 0.1)
        r2 = sys2.run(env, 1000)
        results['basic'].append(r2['recon_error'])
    
    print(f"\nOptimized: {np.mean(results['optimized']):.4f}")
    print(f"Basic: {np.mean(results['basic']):.4f}")
    
    imp = (np.mean(results['basic']) - np.mean(results['optimized'])) / np.mean(results['basic']) * 100
    print(f"改进: {imp:+.1f}%")


def test_L3_causal():
    """测试L3因果推理"""
    print("\n" + "=" * 60)
    print("Test: L3 Causal Reasoning")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = OptimizedSystem(10, 3, 3, 10, 0.01, 0.1)
    
    # 训练
    for _ in range(500):
        sys.forward(env.gen())
    
    # 测试干预能力
    test_state = np.random.randn(3)
    test_action = np.random.randn(3)
    
    intervened = sys.intervene(test_state, test_action)
    print(f"\n干预结果: {np.linalg.norm(intervened):.3f}")
    
    # 测试反事实
    action_a = np.array([1.0, 0.0, 0.0])
    action_b = np.array([0.0, 1.0, 0.0])
    
    cf = sys.counterfactual(test_state, action_a, action_b)
    print(f"\n反事实差异: {np.linalg.norm(cf['difference']):.3f}")
    
    # 测试为什么
    why_result = sys.why(test_state, test_action, intervened)
    print(f"\n解释: {why_result.get('explanation', 'N/A')}")
    
    # 因果图
    print(f"\n因果链路数: {sys.transition.causal_graph}")


def test_L2_generalization():
    """L2泛化测试"""
    print("\n" + "=" * 60)
    print("Test: L2 Generalization (Optimized)")
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
        
        sys = OptimizedSystem(10, 3, 3, 10, 0.01, 0.1)
        
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
    test_optimization()
    test_L3_causal()
    test_L2_generalization()
    print("\nDone!")
