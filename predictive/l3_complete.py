"""
FCRS-v5 Predictive: 完整L3因果推理系统
更严格的实现
"""

import numpy as np
from collections import defaultdict


class CausalReasoningSystem:
    """
    L3因果推理系统
    
    核心能力:
    1. 干预 (Intervention): do(X) = x
    2. 反事实 (Counterfactual): 如果当初...会怎样
    3. 为什么 (Why): 解释原因
    """
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 因果模型: P(next_state | do(action))
        self.causal_model = defaultdict(list)
        
        # 历史记录
        self.history = []
        
        # 因果图
        self.causal_graph = defaultdict(lambda: defaultdict(list))
    
    def record(self, state, action, next_state):
        """记录因果关系"""
        # 记录 (action -> next_state) 的因果效应
        key = tuple(np.round(action, 1))
        effect = next_state - state  # 状态变化
        self.causal_model[key].append(effect)
        
        # 更新因果图
        for i, e in enumerate(effect):
            self.causal_graph[i][key[i]].append(e)
        
        # 记录历史
        self.history.append({
            'state': state,
            'action': action,
            'next_state': next_state
        })
    
    def intervene(self, action):
        """干预: 强制执行某个行动"""
        # 返回干预后的预期结果
        key = tuple(np.round(action, 1))
        
        if key in self.causal_model:
            effects = self.causal_model[key]
            expected_effect = np.mean(effects, axis=0)
            return expected_effect
        else:
            # 无数据，返回随机
            return np.zeros(self.state_dim)
    
    def counterfactual(self, actual_action, hypothetical_action, current_state):
        """反事实推理"""
        # 实际情况
        actual_effect = self.intervene(actual_action)
        
        # 假设情况
        hypothetical_effect = self.intervene(hypothetical_action)
        
        # 反事实结果
        cf_result = current_state + hypothetical_effect - actual_effect
        
        return {
            'actual': current_state + actual_effect,
            'hypothetical': cf_result,
            'difference': hypothetical_effect - actual_effect
        }
    
    def why(self, action, outcome):
        """回答为什么"""
        key = tuple(np.round(action, 1))
        
        if key in self.causal_model:
            effects = self.causal_model[key]
            mean_effect = np.mean(effects, axis=0)
            
            # 找到最相关的因果因素
            relevant_dims = np.argsort(np.abs(mean_effect))[-2:]
            
            explanation = []
            for dim in relevant_dims:
                if mean_effect[dim] > 0:
                    explanation.append(f"维度{dim}增加{mean_effect[dim]:.2f}")
                else:
                    explanation.append(f"维度{dim}减少{abs(mean_effect[dim]):.2f}")
            
            return {
                'explanation': "因为" + "，".join(explanation),
                'confidence': len(effects) / 100,  # 基于样本数
                'mean_effect': mean_effect
            }
        
        return {'explanation': '数据不足', 'confidence': 0}


class CompletePredictiveSystem:
    """完整预测系统 + L3因果推理"""
    
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
        self.reps = []
        self.rep_predictors = {}
        
        for i in range(3):
            rep = np.random.randn(input_dim)
            rep = rep / (np.linalg.norm(rep) + 1e-8)
            self.reps.append(rep)
            self.rep_predictors[i] = self._make_predictor()
        
        # 状态转移
        self.W_trans = np.random.randn(compress_dim + action_dim, compress_dim) * 0.1
        self.b_trans = np.zeros(compress_dim)
        
        # 因果推理
        self.causal = CausalReasoningSystem(compress_dim, action_dim)
        
        self.step = 0
        self.prev_f = None
        self.prev_action = None
    
    def _make_predictor(self):
        return {
            'W': np.eye(self.compress_dim) * 0.5,
            'b': np.zeros(self.compress_dim),
            'errors': []
        }
    
    def compress(self, x):
        f = x @ self.W_enc + self.b_enc
        f = np.maximum(0, f)
        if np.linalg.norm(f) > 1e-6:
            f = f / np.linalg.norm(f)
        return f
    
    def predict_next(self, f, action):
        """预测下一状态"""
        sa = np.concatenate([f, action])
        next_f = sa @ self.W_trans + self.b_trans
        return np.maximum(0, next_f)
    
    def update(self, f_curr, action, f_next):
        """更新模型"""
        # 预测
        pred = self.predict_next(f_curr, action)
        error = f_next - pred
        
        # 更新转移模型
        sa = np.concatenate([f_curr, action])
        self.W_trans += self.lr * np.outer(sa, error)
        self.b_trans += self.lr * error
        
        # 记录因果
        self.causal.record(f_curr, action, f_next)
        
        return np.linalg.norm(error)
    
    def select_rep(self, x, f):
        if np.random.random() < self.explore:
            return np.random.randint(len(self.reps))
        
        scores = []
        for rep in self.reps:
            # 简单评分
            scores.append(-np.linalg.norm(rep - x))
        
        return np.argmax(scores)
    
    def forward(self, x, action=None):
        self.step += 1
        
        f = self.compress(x)
        
        if action is None:
            action = np.random.randn(self.action_dim)
        
        # 更新
        if self.prev_f is not None and self.prev_action is not None:
            self.update(self.prev_f, self.prev_action, f)
        
        # 选择表征
        idx = self.select_rep(x, f)
        
        # 更新表征
        self.reps[idx] = self.reps[idx] + self.lr * (x - self.reps[idx])
        self.reps[idx] = self.reps[idx] / (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        self.prev_f = f
        self.prev_action = action
        
        return f
    
    def do(self, action):
        """干预"""
        return self.causal.intervene(action)
    
    def what_if(self, actual, hypothetical, state):
        """反事实"""
        return self.causal.counterfactual(actual, hypothetical, state)
    
    def why_because(self, action, outcome):
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
def test_l3_full():
    """完整L3测试"""
    print("=" * 60)
    print("Complete L3 Test")
    print("=" * 60)
    
    np.random.seed(42)
    env = Env(10, 5)
    sys = CompletePredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
    
    # 训练
    for _ in range(1000):
        sys.forward(env.gen())
    
    # 测试干预
    test_action = np.array([1.0, 0.5, -0.5])
    intervention = sys.do(test_action)
    print(f"\n干预结果: {intervention[:3]}")
    
    # 测试反事实
    actual = np.array([1.0, 0.0, 0.0])
    hypothetical = np.array([0.0, 1.0, 0.0])
    test_state = np.random.randn(3)
    
    cf = sys.what_if(actual, hypothetical, test_state)
    print(f"\n反事实差异: {cf['difference'][:3]}")
    
    # 测试为什么
    outcome = np.random.randn(3)
    why = sys.why_because(test_action, outcome)
    print(f"\n解释: {why['explanation']}")
    print(f"置信度: {why['confidence']:.2f}")
    
    # 因果图大小
    print(f"\n因果链路数: {len(sys.causal.causal_model)}")


def test_comparison():
    """对比测试"""
    print("\n" + "=" * 60)
    print("Comparison: L1-L2 vs L3")
    print("=" * 60)
    
    results = {'L1_L2': [], 'L3': []}
    
    for run in range(10):
        # L1_L2系统
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        sys = CompletePredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
        
        errors = []
        for _ in range(500):
            err = np.linalg.norm(sys.forward(env.gen()))
            errors.append(err)
        
        results['L1_L2'].append(np.mean(errors))
        
        # L3系统（有因果推理）
        np.random.seed(run * 100)
        env = Env(10, 5)
        
        sys3 = CompletePredictiveSystem(10, 3, 3, 10, 0.01, 0.1)
        
        # 使用因果推理选择
        errors3 = []
        for _ in range(500):
            action = np.random.randn(3)
            
            # 用因果模型选择更好的行动
            best_action = action
            best_effect = 0
            
            for a in [action, action * 0.5, action * 1.5]:
                effect = np.linalg.norm(sys3.do(a))
                if effect > best_effect:
                    best_effect = effect
                    best_action = a
            
            err = np.linalg.norm(sys3.forward(env.gen(), best_action))
            errors3.append(err)
        
        results['L3'].append(np.mean(errors3))
    
    print(f"\nL1_L2: {np.mean(results['L1_L2']):.4f}")
    print(f"L3: {np.mean(results['L3']):.4f}")
    
    imp = (np.mean(results['L1_L2']) - np.mean(results['L3'])) / np.mean(results['L1_L2']) * 100
    print(f"改进: {imp:+.1f}%")


# ==================== Main ====================
test_l3_full()
test_comparison()
print("\nDone!")
