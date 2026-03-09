"""
最低限度涌现系统
核心：去掉所有人为设计，让环境自然选择
"""

import numpy as np


class MinimalAgent:
    """最小智能体：只有2条规则"""
    
    def __init__(self):
        # 状态 = 任意向量
        self.state = np.random.randn(10)
        # 连接强度
        self.connections = np.random.randn(10, 10) * 0.1
    
    def perceive(self, env):
        """感知：无差别记录"""
        return env.copy()
    
    def act(self, perception):
        """行动：基于当前状态"""
        # 简单：状态 = 状态 + 连接 @ 感知
        self.state += self.connections @ perception * 0.1
        return self.state
    
    def learn(self, feedback):
        """学习：无适应度函数"""
        # Hebbian: 如果同时激活，加强连接
        # feedback > 0 表示"surprise" or "error reduction"
        self.connections += feedback * 0.01


class MinimalEcosystem:
    """最小生态系统"""
    
    def __init__(self, n_agents=10):
        self.agents = [MinimalAgent() for _ in range(n_agents)]
        # 环境状态
        self.env = np.random.randn(10)
        self.env_history = [self.env.copy()]
    
    def step(self):
        # 环境自然变化
        self.env += np.random.randn(10) * 0.1
        self.env_history.append(self.env.copy())
        
        # 每个agent独立感知和行动
        for agent in self.agents:
            p = agent.perceive(self.env)
            output = agent.act(p)
            
            # 反馈 = 环境与输出的差异（负=好）
            # 这就是"预测误差"
            feedback = -(output - self.env)
            agent.learn(feedback)
        
        # 自然选择：随机淘汰弱的
        # 这里的"弱" = 长时间无变化
        if len(self.agents) > 5:
            # 随机淘汰一个
            idx = np.random.randint(0, len(self.agents))
            self.agents.pop(idx)
            
            # 补充新的（通过变异现有）
            parent = np.random.choice(self.agents)
            child = MinimalAgent()
            child.state = parent.state + np.random.randn(10) * 0.1
            child.connections = parent.connections + np.random.randn(10, 10) * 0.05
            self.agents.append(child)


def test_minimal():
    """测试最低限度系统"""
    print("="*60)
    print("Minimal Emergence Test")
    print("="*60)
    
    np.random.seed(42)
    eco = MinimalEcosystem(10)
    
    # 跟踪
    env_changes = []
    agent_states = []
    
    for step in range(1000):
        eco.step()
        
        if step % 100 == 0:
            # 记录环境变化
            env_changes.append(np.linalg.norm(eco.env))
            
            # 记录agent状态差异
            states = [np.linalg.norm(a.state) for a in eco.agents]
            agent_states.append(np.std(states))
            
            print(f"Step {step}: env={env_changes[-1]:.2f}, agent_std={agent_states[-1]:.2f}")
    
    # 分析
    print("\n--- Analysis ---")
    print(f"Final env norm: {np.linalg.norm(eco.env):.2f}")
    print(f"Agent state diversity: {np.mean(agent_states):.3f}")
    
    # 检查是否有涌现行为
    # 标准：agent行为与环境相关，而非随机
    correlations = []
    for i in range(len(eco.agents)):
        corr = np.corrcoef(eco.agents[i].state, eco.env)[0, 1]
        correlations.append(corr)
    
    print(f"Agent-Env correlation: {np.mean(correlations):.3f}")
    
    if np.abs(np.mean(correlations)) > 0.3:
        print("[OK] Emergence: Agents adapted to environment!")
    else:
        print("[?] No clear emergence")


def test_vs_random():
    """对比：是否优于随机"""
    print("\n" + "="*60)
    print("Comparison: Adaptive vs Random")
    print("="*60)
    
    results = {'adaptive': [], 'random': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        # Adaptive
        eco1 = MinimalEcosystem(10)
        for _ in range(500):
            eco1.step()
        
        # 最终误差
        errors1 = [np.linalg.norm(a.state - eco1.env) for a in eco1.agents]
        results['adaptive'].append(np.mean(errors1))
        
        # Random (no learning)
        np.random.seed(run * 100)
        eco2 = MinimalEcosystem(10)
        
        class NoLearnAgent(MinimalAgent):
            def learn(self, feedback):
                pass  # 不学习
        
        eco2.agents = [NoLearnAgent() for _ in range(10)]
        
        for _ in range(500):
            eco2.step()
        
        errors2 = [np.linalg.norm(a.state - eco2.env) for a in eco2.agents]
        results['random'].append(np.mean(errors2))
    
    print(f"Adaptive: {np.mean(results['adaptive']):.3f}")
    print(f"Random: {np.mean(results['random']):.3f}")
    
    if np.mean(results['adaptive']) < np.mean(results['random']):
        print("[OK] Learning helps!")
    else:
        print("[?] No clear advantage")


# ==================== Main ====================
test_minimal()
test_vs_random()

print("\n" + "="*60)
print("Minimal Design Principles:")
print("1. Multiple candidates (diversity)")
print("2. Environmental feedback (signal)")
print("3. Keep what works (selection)")
print("4. No preset goal")
print("="*60)
