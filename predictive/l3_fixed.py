"""
FCRS-v5 Predictive: 修复版L3因果推理
"""

import numpy as np
from collections import defaultdict


class CausalModel:
    """因果模型"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.causal_effects = []  # (action, next_state - current_state)
    
    def record(self, state, action, next_state):
        effect = next_state - state
        self.causal_effects.append((action.copy(), effect.copy()))
    
    def intervene(self, action):
        """给定行动，返回预期状态变化"""
        if not self.causal_effects:
            return np.zeros(self.state_dim)
        
        # 找相似行动的效果
        similar_effects = []
        for a, e in self.causal_effects:
            # 行动相似度
            sim = np.dot(action, a) / (np.linalg.norm(action) * np.linalg.norm(a) + 1e-8)
            if sim > 0.5:  # 相似行动
                similar_effects.append(e)
        
        if similar_effects:
            return np.mean(similar_effects, axis=0)
        
        # 返回平均效果
        return np.mean([e for _, e in self.causal_effects], axis=0)
    
    def counterfactual(self, actual_action, hypothetical_action, current_state):
        """反事实"""
        actual_effect = self.intervene(actual_action)
        hypothetical_effect = self.intervene(hypothetical_action)
        
        return {
            'actual': current_state + actual_effect,
            'hypothetical': current_state + hypothetical_effect,
            'difference': hypothetical_effect - actual_effect
        }
    
    def why(self, action, outcome):
        """解释"""
        if len(self.causal_effects) < 10:
            return "数据不足"
        
        # 找最相似的历史
        best_sim = -1
        best_effect = None
        
        for a, e in self.causal_effects[-100:]:  # 最近100条
            sim = np.dot(action, a) / (np.linalg.norm(action) * np.linalg.norm(a) + 1e-8)
            if sim > best_sim:
                best_sim = sim
                best_effect = e
        
        if best_effect is not None and best_sim > 0.3:
            reasons = []
            for i, e in enumerate(best_effect):
                if abs(e) > 0.1:
                    reasons.append(f"d{i}={'+' if e > 0 else ''}{e:.2f}")
            return f"因为过去相似行动导致: {', '.join(reasons)}"
        
        return "模式不明显"


class PredictiveWithCausal:
    """预测系统 + 因果推理"""
    
    def __init__(self, input_dim=10, compress_dim=5, action_dim=3, capacity=10, lr=0.01, explore=0.1):
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
        self.reps = [np.random.randn(input_dim) for _ in range(3)]
        
        # 转移模型
        self.W_trans = np.random.randn(compress_dim + action_dim, compress_dim) * 0.1
        self.b_trans = np.zeros(compress_dim)
        
        # 因果模型
        self.causal = CausalModel(compress_dim, action_dim)
        
        self.step = 0
        self.prev_f = None
        self.prev_action = None
    
    def compress(self, x):
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def forward(self, x, action=None):
        self.step += 1
        f = self.compress(x)
        
        if action is None:
            action = np.random.randn(self.action_dim)
        
        # 更新转移模型和因果
        if self.prev_f is not None and self.prev_action is not None:
            pred = self.predict_next(self.prev_f, self.prev_action)
            error = f - pred
            
            sa = np.concatenate([self.prev_f, self.prev_action])
            self.W_trans += self.lr * np.outer(sa, error)
            self.b_trans += self.lr * error
            
            # 记录因果
            self.causal.record(self.prev_f, self.prev_action, f)
        
        # 选择表征
        idx = np.random.randint(len(self.reps)) if np.random.random() < self.explore else \
              np.argmax([-np.linalg.norm(r - x) for r in self.reps])
        
        # 更新表征
        self.reps[idx] = self.reps[idx] + self.lr * (x - self.reps[idx])
        self.reps[idx] = self.reps[idx] / (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        self.prev_f = f
        self.prev_action = action
        
        return f
    
    def predict_next(self, f, action):
        sa = np.concatenate([f, action])
        return np.maximum(0, sa @ self.W_trans + self.b_trans)
    
    def do(self, action):
        return self.causal.intervene(action)
    
    def what_if(self, actual, hypothetical, state):
        return self.causal.counterfactual(actual, hypothetical, state)
    
    def why_because(self, action, outcome):
        return self.causal.why(action, outcome)


class Env:
    def __init__(self, dim=10, n_classes=5):
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def gen(self):
        c = self.centers[np.random.randint(0, len(self.centers))]
        return c + np.random.randn(self.dim) * 0.3


# ==================== 测试 ====================
def test_causal_fixed():
    """修复后的因果测试"""
    print("=" * 60)
    print("Test: Fixed Causal Model")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = PredictiveWithCausal(10, 3, 3, 10, 0.01, 0.1)
    
    # 训练
    for _ in range(500):
        sys.forward(env.gen())
    
    print(f"\n因果记录数: {len(sys.causal.causal_effects)}")
    
    # 测试干预
    test_action = np.array([1.0, 0.5, -0.5])
    intervention = sys.do(test_action)
    print(f"干预结果: {intervention}")
    
    # 测试反事实
    cf = sys.what_if(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.random.randn(3)
    )
    print(f"反事实差异: {cf['difference']}")
    
    # 测试为什么
    why = sys.why_because(test_action, np.random.randn(3))
    print(f"解释: {why}")


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
        sys1 = PredictiveWithCausal(10, 3, 3, 10, 0.01, 0.1)
        
        errs1 = []
        for _ in range(500):
            err = np.linalg.norm(sys1.forward(env.gen()))
            errs1.append(err)
        results['basic'].append(np.mean(errs1))
        
        # 因果系统（使用因果选择）
        np.random.seed(run * 100)
        env = Env(10, 5)
        sys2 = PredictiveWithCausal(10, 3, 3, 10, 0.01, 0.1)
        
        errs2 = []
        for _ in range(500):
            # 用因果选择更好的行动
            action = np.random.randn(3)
            
            # 评估几个候选行动
            effects = []
            for a in [action, action * 0.5, action * 1.5, -action]:
                e = sys2.do(a)
                effects.append(np.linalg.norm(e))
            
            best_idx = np.argmax(effects)
            best_actions = [action, action * 0.5, action * 1.5, -action]
            best_action = best_actions[best_idx]
            
            err = np.linalg.norm(sys2.forward(env.gen(), best_action))
            errs2.append(err)
        
        results['causal'].append(np.mean(errs2))
    
    print(f"\n基础系统: {np.mean(results['basic']):.4f}")
    print(f"因果系统: {np.mean(results['causal']):.4f}")
    
    imp = (np.mean(results['basic']) - np.mean(results['causal'])) / np.mean(results['basic']) * 100
    print(f"改进: {imp:+.1f}%")


# ==================== Main ====================
test_causal_fixed()
test_l3_improvement()
print("\nDone!")
