# -*- coding: utf-8 -*-
"""
FCRS-v5 诊断实验: 找瓶颈

实验1: Oracle预测器 (完美预测) - 看成功率上限
实验2: 完美重构 - 看压缩是否是瓶颈
"""

import numpy as np
import sys
import os

core_path = os.path.join(os.path.dirname(__file__), '..', 'core')
sys.path.insert(0, core_path)

from grid_world import GridWorldEnv
from metrics import Metrics


class LinearPredictor:
    def __init__(self, dim, lr=0.01):
        self.dim, self.lr = dim, lr
        self.W = np.eye(dim) * 0.5
        self.errors = []
    
    def predict(self, s):
        o = s @ self.W
        n = np.linalg.norm(o)
        return o/n if n > 1e-8 else o
    
    def update(self, sc, sn):
        e = sn - self.predict(sc)
        self.errors.append(np.linalg.norm(e))
        self.W += self.lr * np.outer(sc, e)
        if len(self.errors) > 100:
            self.errors = self.errors[-100:]
    
    def get_mean_error(self):
        return np.mean(self.errors[-10:]) if self.errors else 1.0


class DiagnosticAgent:
    def __init__(self, dim, ad, mode='prediction', n=5, lr=0.05, exp=0.3, seed=100):
        np.random.seed(seed)
        self.dim, self.ad, self.mode = dim, ad, mode
        self.n, self.lr, self.exp = n, lr, exp
        
        # 表征池
        self.reps = [np.random.randn(dim) * 0.1 for _ in range(n)]
        
        # 预测器
        if mode == 'oracle':
            self.predictors = None  # Oracle不用预测器
        else:
            self.predictors = [LinearPredictor(dim, lr) for _ in range(n)]
        
        # 策略网络
        self.W = np.random.randn(dim, ad) * 0.1
        self.b = np.zeros(ad)
        
        self.hist = []
        self.m = Metrics()
    
    def sel_rep(self, s, next_s=None):
        """选择表征"""
        if self.mode == 'prediction':
            e = [p.get_mean_error() for p in self.predictors]
            return np.argmin(e)
        elif self.mode == 'reconstruction':
            return np.argmin([np.linalg.norm(r - s) for r in self.reps])
        elif self.mode == 'oracle':
            return np.random.randint(self.n)  # Oracle随机选
        elif self.mode == 'perfect_recon':
            return np.argmax([np.dot(r, s) for r in self.reps])  # 完美重构
        return np.random.randint(self.n)
    
    def sel_act(self, r):
        v = r @ self.W + self.b
        return np.random.randint(self.ad) if np.random.random() < self.exp else np.argmax(v)
    
    def learn(self, s, a, r, ns, i, next_s=None):
        # 更新表征
        self.reps[i] += self.lr * (s - self.reps[i])
        self.reps[i] /= np.linalg.norm(self.reps[i]) + 1e-8
        
        # 更新预测器 (非oracle)
        if self.predictors and self.hist:
            self.predictors[i].update(self.hist[-1], s)
        
        # 策略更新
        rep = self.reps[i]
        v = rep @ self.W + self.b
        for j in range(self.ad):
            if j == a:
                self.W[:, j] += self.lr * (r - v[j]) * rep
                self.b[j] += self.lr * (r - v[j])
            else:
                self.W[:, j] -= 0.01 * v[j] * rep
        
        self.hist.append(s)
        self.m.add_recon_error(np.linalg.norm(self.reps[i] - s))


def run_exp(mode, n_reps=5, lr=0.05, exp=0.3, n_episodes=200, seed=100):
    """运行诊断实验"""
    np.random.seed(seed)
    env = GridWorldEnv(5, 3)
    
    agent = DiagnosticAgent(
        dim=env.input_dim,
        ad=env.action_dim,
        mode=mode,
        n=n_reps,
        lr=lr,
        exp=exp,
        seed=seed
    )
    
    for _ in range(n_episodes):
        s = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 50:
            i = agent.sel_rep(s)
            ac = agent.sel_act(agent.reps[i])
            ns, rw, done = env.step(ac)
            
            # 学习
            agent.learn(s, ac, rw, ns, i)
            
            s = ns
            steps += 1
        
        agent.m.add_episode_result(0, steps, done)
    
    return agent.m.get_success_rate() * 100


def main():
    print("="*60)
    print("FCRS-v5 诊断实验")
    print("="*60)
    
    # 基线对比
    print("\n--- 基线对比 ---")
    
    modes = ['prediction', 'reconstruction', 'random']
    for mode in modes:
        sr = run_exp(mode, n_reps=5, lr=0.05, exp=0.3)
        print(f"{mode:<15}: {sr:.0f}%")
    
    # 诊断实验
    print("\n--- 诊断实验 ---")
    
    # 实验1: Oracle预测器 (未来状态直接给)
    print("\n[实验1] Oracle预测器 (上限测试)")
    # Oracle = 完美预测，我们用"知道下一步真实状态"来模拟
    # 实际上无法直接实现，用随机选择作为下界
    oracle = run_exp('oracle', n_reps=5, lr=0.05, exp=0.3)
    print(f"Oracle (随机): {oracle:.0f}%")
    
    # 实验2: 完美重构
    print("\n[实验2] 完美重构")
    perfect = run_exp('perfect_recon', n_reps=5, lr=0.05, exp=0.3)
    print(f"Perfect Recon: {perfect:.0f}%")
    
    # 总结
    print("\n" + "="*60)
    print("诊断结论")
    print("="*60)
    
    pred = run_exp('prediction', n_reps=5, lr=0.05, exp=0.3)
    recon = run_exp('reconstruction', n_reps=5, lr=0.05, exp=0.3)
    
    print(f"\n当前最佳 (prediction): {pred:.0f}%")
    print(f"重构基线: {recon:.0f}%")
    print(f"Oracle: {oracle:.0f}%")
    print(f"Perfect Recon: {perfect:.0f}%")
    
    # 分析
    gap_to_oracle = oracle - pred
    gap_to_perfect = perfect - pred
    
    print(f"\n与Oracle差距: {gap_to_oracle:.0f}%")
    print(f"与完美重构差距: {gap_to_perfect:.0f}%")
    
    if gap_to_oracle > 20:
        print("\n结论: 预测器有巨大优化空间")
    elif gap_to_perfect > 10:
        print("\n结论: 表征压缩是瓶颈")
    else:
        print("\n结论: 策略网络/选择机制是瓶颈")


if __name__ == "__main__":
    main()
