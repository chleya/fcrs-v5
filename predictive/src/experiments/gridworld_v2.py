# -*- coding: utf-8 -*-
"""
FCRS-v5: 网格世界多步决策实验 v2
核心修改: 表征选择直接影响动作决策

决策链路:
环境输入 → 压缩为多组表征 → 表征选择(预测/重构/随机) → 
仅用选中表征生成动作 → 动作执行 → 进入下一循环
"""

import numpy as np
from typing import Dict, List
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from grid_world import GridWorldEnv
from metrics import Metrics


# ==================== 基于表征的Agent ====================

class RepresentationAgent:
    """
    表征驱动的Agent
    
    核心特点: 动作生成只依赖选中的表征
    """
    
    def __init__(self, state_dim: int, action_dim: int,
                 selection_mode: str = 'prediction',  # 'prediction', 'reconstruction', 'random'
                 n_reps: int = 3, lr: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selection_mode = selection_mode
        self.n_reps = n_reps
        self.lr = lr
        
        # ===== 表征池 =====
        # 初始表征 (state_dim维)
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        
        # 每个表征的预测器
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        
        # ===== 动作策略网络 (基于表征) =====
        # 输入: 表征维度 (不是原始state_dim!)
        # 输出: action_dim维Q值
        self.W_policy = np.random.randn(state_dim, action_dim) * 0.1
        self.b_policy = np.zeros(action_dim)
        
        # ===== 历史 =====
        self.prev_state = None
        self.state_history = []
        self.rep_history = []
        self.selection_history = []
        
        # ===== 指标 =====
        self.metrics = Metrics()
    
    def select_representation(self, state: np.ndarray, explore: float = 0.1) -> int:
        """选择表征 (核心决策点)"""
        # ε-贪心探索
        if np.random.random() < explore:
            idx = np.random.randint(self.n_reps)
            self.selection_history.append(idx)
            return idx
        
        if self.selection_mode == 'prediction':
            # ===== 预测导向: 选择历史预测误差最小的表征 =====
            scores = []
            for i in range(self.n_reps):
                # 用该表征的预测器，预测下一状态
                if len(self.state_history) > 0:
                    prev = self.state_history[-1]
                    pred = prev @ self.predictors[i]
                    # 预测误差 (越小越好)
                    error = np.linalg.norm(pred - state)
                else:
                    error = 1.0  # 无历史时默认
                
                scores.append(-error)  # 负误差 = 得分
            
            idx = np.argmax(scores)
        
        elif self.selection_mode == 'reconstruction':
            # ===== 重构导向: 选择与当前状态最相似的表征 =====
            scores = [np.linalg.norm(r - state) for r in self.reps]
            idx = np.argmin(scores)  # 误差最小
        
        else:  # random
            idx = np.random.randint(self.n_reps)
        
        self.selection_history.append(idx)
        return idx
    
    def select_action(self, representation: np.ndarray, explore: float = 0.1) -> int:
        """
        基于表征选择动作 (核心修改!)
        
        之前: 基于完整原始状态
        现在: 仅基于选中的表征
        """
        # 策略网络: 表征 -> Q值
        q_values = representation @ self.W_policy + self.b_policy
        
        # ε-贪心
        if np.random.random() < explore:
            return np.random.randint(self.action_dim)
        
        # 选择最大Q值的动作
        return np.argmax(q_values)
    
    def learn(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, rep_idx: int):
        """学习"""
        # ===== 1. 表征更新 =====
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        # ===== 2. 预测器更新 =====
        if self.prev_state is not None:
            pred = self.prev_state @ self.predictors[rep_idx]
            error = state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.prev_state, error)
        
        # ===== 3. 策略网络更新 (基于选中的表征) =====
        selected_rep = self.reps[rep_idx]
        
        # 简单策略梯度
        q_values = selected_rep @ self.W_policy + self.b_policy
        target_q = reward  # 简化为即时奖励
        
        # 更新策略网络
        for a in range(self.action_dim):
            if a == action:
                self.W_policy[:, a] += self.lr * (target_q - q_values[a]) * selected_rep
                self.b_policy[a] += self.lr * (target_q - q_values[a])
            else:
                # 惩罚其他动作
                self.W_policy[:, a] -= 0.01 * q_values[a] * selected_rep
        
        # ===== 4. 记录历史 =====
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        self.prev_state = state.copy()
        
        # ===== 5. 记录指标 =====
        recon_err = np.linalg.norm(self.reps[rep_idx] - state)
        self.metrics.add_recon_error(recon_err)
        
        # 多步预测误差
        if len(self.state_history) >= 3:
            # 用预测器预测未来3步
            total_error = 0
            curr = self.state_history[-1]
            for _ in range(3):
                curr = curr @ self.predictors[rep_idx]
                total_error += np.linalg.norm(curr - state)
            self.metrics.add_multi_step_error(total_error / 3)


class SimpleAgent:
    """简单Agent (基线)"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_table = np.zeros((state_dim, action_dim))
        self.metrics = Metrics()
    
    def select_action(self, state: np.ndarray, epsilon: float = 1.0) -> int:
        return np.random.randint(self.action_dim)
    
    def learn(self, state, action, reward, next_state, rep_idx):
        pass


# ==================== 实验 ====================

def run_experiment(selection_mode: str, n_episodes: int = 100,
                   n_runs: int = 10) -> Dict:
    """运行实验"""
    results = {
        'success_rates': [],
        'mean_rewards': [],
        'mean_lengths': [],
        'mean_recon_errors': [],
        'mean_multi_step_errors': []
    }
    
    for run in range(n_runs):
        np.random.seed(run * 100)
        
        # 环境
        env = GridWorldEnv(size=5, n_obstacles=3)
        
        # Agent
        agent = RepresentationAgent(
            state_dim=env.input_dim,
            action_dim=env.action_dim,
            selection_mode=selection_mode,
            n_reps=3,
            lr=0.01
        )
        
        # 训练
        for ep in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done and steps < 50:
                # ===== 核心: 先选择表征，再用表征生成动作 =====
                rep_idx = agent.select_representation(state, explore=0.2)
                
                # 用选中的表征生成动作
                action = agent.select_action(agent.reps[rep_idx], explore=0.2)
                
                # 执行动作
                next_state, reward, done = env.step(action)
                
                # 学习
                agent.learn(state, action, reward, next_state, rep_idx)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            agent.metrics.add_episode_result(total_reward, steps, done)
        
        # 记录结果
        results['success_rates'].append(agent.metrics.get_success_rate())
        results['mean_rewards'].append(agent.metrics.get_mean_reward())
        results['mean_lengths'].append(agent.metrics.get_mean_length())
        results['mean_recon_errors'].append(agent.metrics.get_mean_recon_error())
        results['mean_multi_step_errors'].append(agent.metrics.get_mean_multi_step_error())
    
    return {
        'success_rate': np.mean(results['success_rates']),
        'std_success': np.std(results['success_rates']),
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_length': np.mean(results['mean_lengths']),
        'mean_recon_error': np.mean(results['mean_recon_errors']),
        'mean_multi_step_error': np.mean(results['mean_multi_step_errors'])
    }


def main():
    """主实验"""
    print("="*60)
    print("GridWorld v2: Representation-Driven Decision")
    print("="*60)
    print("\n核心修改: 表征选择 -> 直接影响动作决策")
    
    # 实验
    modes = ['prediction', 'reconstruction', 'random']
    all_results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        all_results[mode] = run_experiment(mode, n_episodes=100, n_runs=10)
    
    # 结果对比
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Mode':<20} {'Success':<10} {'Reward':<10} {'Steps':<10} {'Recon Err':<12} {'Multi-Step':<12}")
    print("-"*80)
    
    for mode in modes:
        r = all_results[mode]
        print(f"{mode:<20} {r['success_rate']*100:<9.1f}% {r['mean_reward']:<10.2f} {r['mean_length']:<10.1f} {r['mean_recon_error']:<12.4f} {r['mean_multi_step_error']:<12.4f}")
    
    # 核心对比
    print("\n" + "="*60)
    print("PREDICTION vs RECONSTRUCTION")
    print("="*60)
    
    pred = all_results['prediction']
    recon = all_results['reconstruction']
    
    success_diff = (pred['success_rate'] - recon['success_rate']) * 100
    reward_diff = pred['mean_reward'] - recon['mean_reward']
    recon_err_diff = pred['mean_recon_error'] - recon['mean_recon_error']
    multi_step_diff = pred['mean_multi_step_error'] - recon['mean_multi_step_error']
    
    print(f"\nSuccess Rate: {success_diff:+.1f}%")
    print(f"Mean Reward: {reward_diff:+.2f}")
    print(f"Recon Error: {recon_err_diff:+.4f}")
    print(f"Multi-Step Error: {multi_step_diff:+.4f}")
    
    # 结论
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if pred['success_rate'] > recon['success_rate']:
        print("\n[OK] Prediction-based selection OUTPERFORMS!")
    elif pred['success_rate'] == recon['success_rate']:
        print("\n[=] No difference - need further investigation")
    else:
        print("\n[X] Reconstruction still wins on success rate")


if __name__ == "__main__":
    main()
