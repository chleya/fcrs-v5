"""
FCRS-v5 Predictive: L3因果推理 - 最终版
修复检索bug
"""

import numpy as np


class CausalModel:
    """因果模型"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.effects = []  # list of (action, effect)
    
    def record(self, action, effect):
        """记录: action -> effect"""
        self.effects.append((action.copy(), effect.copy()))
    
    def intervene(self, action):
        """干预: 强制执行action"""
        if len(self.effects) < 10:
            return np.zeros(self.state_dim)
        
        # 找相似action的效果
        weights = []
        results = []
        
        for a, e in self.effects[-100:]:  # 最近100条
            # 相似度
            norm_a = np.linalg.norm(action)
            norm_a2 = np.linalg.norm(a)
            if norm_a < 1e-8 or norm_a2 < 1e-8:
                sim = 0
            else:
                sim = np.dot(action, a) / (norm_a * norm_a2)
            
            if sim > 0.3:  # 相似度阈值
                weights.append(sim)
                results.append(e)
        
        if results:
            weights = np.array(weights)
            results = np.array(results)
            return np.average(results, weights=weights, axis=0)
        
        # 默认返回平均效果
        return np.mean([e for _, e in self.effects], axis=0)
    
    def counterfactual(self, action_a, action_b, state):
        """反事实: 如果做B而不是A"""
        effect_a = self.intervene(action_a)
        effect_b = self.intervene(action_b)
        
        return {
            'actual': state + effect_a,
            'hypothetical': state + effect_b,
            'difference': effect_b - effect_a
        }
    
    def why(self, action, outcome):
        """为什么: 解释action导致outcome"""
        if len(self.effects) < 10:
            return "数据不足，无法解释"
        
        # 找相似action
        explanations = []
        
        for a, e in self.effects[-50:]:
            sim = np.dot(action, a) / (np.linalg.norm(action) * np.linalg.norm(a) + 1e-8)
            if sim > 0.5:
                # 这个action导致了e
                if np.linalg.norm(e) > 0.1:
                    explanations.append((sim, e))
        
        if not explanations:
            return "未发现明显因果关系"
        
        # 返回最强的因果效应
        best = max(explanations, key=lambda x: x[0] * np.linalg.norm(x[1]))
        sim, effect = best
        
        reason = []
        for i, val in enumerate(effect):
            if abs(val) > 0.05:
                reason.append(f"维度{i}:{val:+.2f}")
        
        return f"相似行动历史显示 → {', '.join(reason[:3])}"


class PredictiveSystem:
    """完整预测系统"""
    
    def __init__(self, input_dim=10, compress_dim=5, action_dim=3, capacity=10, lr=0.01, explore=0.1):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.capacity = capacity
        self.explore = explore
        self.lr = lr
        
        # 编码器
        self.W_enc = np.random.randn(input_dim, compress_dim) * 0.1
        self.b_enc = np.zeros(compress_dim)
        
        # 表征池
        self.reps = [np.random.randn(input_dim) for _ in range(3)]
        
        # 转移模型
        self.W_trans = np.random.randn(compress_dim + action_dim, compress_dim) * 0.1
        self.b_trans = np.zeros(compress_dim)
        
        # 因果模型
        self.causal = CausalModel(compress_dim, action_dim)
        
        # 历史
        self.step = 0
        self.prev_state = None
        self.prev_action = None
    
    def compress(self, x):
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def predict_next(self, state, action):
        sa = np.concatenate([state, action])
        return np.maximum(0, sa @ self.W_trans + self.b_trans)
    
    def forward(self, x, action=None):
        self.step += 1
        state = self.compress(x)
        
        if action is None:
            action = np.random.randn(self.compress_dim)
        
        # 更新
        if self.prev_state is not None and self.prev_action is not None:
            pred = self.predict_next(self.prev_state, self.prev_action)
            error = state - pred
            
            # 转移模型
            sa = np.concatenate([self.prev_state, self.prev_action])
            self.W_trans += self.lr * np.outer(sa, error)
            self.b_trans += self.lr * error
            
            # 因果记录
            effect = state - self.prev_state
            self.causal.record(self.prev_action, effect)
        
        # 选择表征
        if np.random.random() < self.explore:
            idx = np.random.randint(len(self.reps))
        else:
            idx = np.argmax([-np.linalg.norm(r - x) for r in self.reps])
        
        # 更新表征
        self.reps[idx] = self.reps[idx] + self.lr * (x - self.reps[idx])
        self.reps[idx] = self.reps[idx] / (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        self.prev_state = state
        self.prev_action = action
        
        return state
    
    # L3能力
    def do(self, action):
        """干预"""
        return self.causal.intervene(action)
    
    def what_if(self, action_a, action_b, state):
        """反事实"""
        return self.causal.counterfactual(action_a, action_b, state)
    
    def why(self, action, outcome):
        """为什么"""
        return self.causal.why(action, outcome)


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 测试 ====================
def test_l3():
    """L3测试"""
    print("=" * 60)
    print("Test: L3 Causal Reasoning")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = PredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
    
    # 训练
    for _ in range(500):
        sys.forward(env.gen())
    
    # 测试
    print(f"\n因果记录数: {len(sys.causal.effects)}")
    
    # 干预
    action = np.array([1.0, 0.5, -0.5])
    intervention = sys.do(action)
    print(f"\n干预 do({action[:3]}):")
    print(f"  效果: {intervention}")
    
    # 反事实
    state = np.random.randn(3)
    cf = sys.what_if(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), state)
    print(f"\n反事实 what-if:")
    print(f"  差异: {cf['difference']}")
    
    # 为什么
    outcome = np.random.randn(3)
    why = sys.why(action, outcome)
    print(f"\n为什么? {why}")


def test_l3_improvement():
    """L3改进测试"""
    print("\n" + "=" * 60)
    print("Test: L3 Improvement")
    print("=" * 60)
    
    results = {'basic': [], 'causal': []}
    
    for run in range(10):
        # 基础系统
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys1 = PredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
        
        errs1 = []
        for _ in range(500):
            err = np.linalg.norm(sys1.forward(env.gen()))
            errs1.append(err)
        results['basic'].append(np.mean(errs1))
        
        # 因果系统
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys2 = PredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
        
        errs2 = []
        for _ in range(500):
            # 用因果选择更好行动
            action = np.random.randn(3)
            
            # 评估几个候选
            effects = []
            for a in [action, action * 0.5, action * 1.5, -action]:
                e = sys2.do(a)
                effects.append(np.linalg.norm(e))
            
            best_idx = np.argmax(effects)
            best_actions = [action, action * 0.5, action * 1.5, -action]
            
            err = np.linalg.norm(sys2.forward(env.gen(), best_actions[best_idx]))
            errs2.append(err)
        
        results['causal'].append(np.mean(errs2))
    
    print(f"\n基础系统: {np.mean(results['basic']):.4f}")
    print(f"因果系统: {np.mean(results['causal']):.4f}")
    
    imp = (np.mean(results['basic']) - np.mean(results['causal'])) / np.mean(results['basic']) * 100
    print(f"改进: {imp:+.1f}%")


# ==================== Main ====================
test_l3()
test_l3_improvement()
print("\nDone!")
