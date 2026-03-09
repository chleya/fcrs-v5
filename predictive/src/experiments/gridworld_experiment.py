# -*- coding: utf-8 -*-
"""
FCRS-v5: 网格世界多步决策实验
核心目标: 验证预测选择的前瞻规划价值
"""

import numpy as np
from typing import Dict, List
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

# 导入模块
from grid_world import GridWorldEnv, SimpleAgent
from metrics import Metrics, compare_methods, print_comparison


# ==================== Agent with Representation ====================

class RepAgent:
    """
    带表征的Agent
    - 有表征池
    - 可以选择预测导向或重构导向
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 selection_mode: str = 'prediction',  # 'prediction' or 'reconstruction'
                 n_reps: int = 3, lr: float = 0.01):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.selection_mode = selection_mode
        self.n_reps = n_reps
        self.lr = lr
        
        # 表征池 (state_dim维)
        self.reps = [np.random.randn(state_dim) * 0.1 for _ in range(n_reps)]
        
        # 每个表征的预测器
        self.predictors = [np.eye(state_dim) * 0.5 for _ in range(n_reps)]
        
        # Q表
        self.q_table = np.random.randn(state_dim, action_dim) * 0.01
        
        # 历史
        self.prev_state = None
        self.state_history = []
        self.selection_history = []
        
        # 指标
        self.metrics = Metrics()
    
    def select_representation(self, state: np.ndarray, explore: float = 0.1) -> int:
        """选择表征"""
        # ε-贪心
        if np.random.random() < explore:
            idx = np.random.randint(self.n_reps)
            self.selection_history.append(idx)
            return idx
        
        if self.selection_mode == 'prediction':
            # 预测导向: 选择历史预测误差最小的表征
            scores = []
            for i in range(self.n_reps):
                # 预测误差
                if len(self.state_history) > 0:
                    prev = self.state_history[-1]
                    pred = prev @ self.predictors[i]
                    error = np.linalg.norm(pred - state)
                else:
                    error = 1.0
                
                scores.append(-error)
            
            idx = np.argmax(scores)
        
        else:  # reconstruction
            # 重构导向: 选择与当前状态最相似的表征
            scores = [np.linalg.norm(r - state) for r in self.reps]
            idx = np.argmin(scores)
        
        self.selection_history.append(idx)
        return idx
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """选择动作 (基于Q表)"""
        state_idx = np.argmax(state)
        
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        return np.argmax(self.q_table[state_idx])
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, rep_idx: int):
        """学习"""
        # Q学习
        state_idx = np.argmax(state)
        next_idx = np.argmax(next_state)
        
        alpha = 0.1
        gamma = 0.9
        
        current_q = self.q_table[state_idx, action]
        max_next_q = np.max(self.q_table[next_idx])
        
        self.q_table[state_idx, action] += alpha * (reward + gamma * max_next_q - current_q)
        
        # 更新表征
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        # 更新预测器
        if self.prev_state is not None:
            pred = self.prev_state @ self.predictors[rep_idx]
            error = state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.prev_state, error)
        
        # 记录
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
        
        self.prev_state = state.copy()
        
        # 记录指标
        recon_err = np.linalg.norm(self.reps[rep_idx] - state)
        self.metrics.add_recon_error(recon_err)


class RandomAgent:
    """随机Agent (基线)"""
    
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

def run_experiment(selection_mode: str, n_episodes: int = 50, 
                   n_runs: int = 10) -> Dict:
    """
    运行实验
    
    Args:
        selection_mode: 'prediction' 或 'reconstruction'
        n_episodes: 训练轮数
        n_runs: 重复次数
    
    Returns:
        结果字典
    """
    results = {
        'success_rates': [],
        'mean_rewards': [],
        'mean_lengths': [],
        'mean_recon_errors': []
    }
    
    for run in range(n_runs):
        np.random.seed(run * 100)
        
        # 环境
        env = GridWorldEnv(size=5, n_obstacles=3)
        
        # Agent
        agent = RepAgent(
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
                # 选择表征
                rep_idx = agent.select_representation(state, explore=0.1)
                
                # 选择动作
                action = agent.select_action(state, epsilon=0.1)
                
                # 执行
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
    
    return {
        'success_rate': np.mean(results['success_rates']),
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_length': np.mean(results['mean_lengths']),
        'mean_recon_error': np.mean(results['mean_recon_errors']),
        'std_success': np.std(results['success_rates'])
    }


def main():
    """主实验"""
    print("="*60)
    print("GridWorld Multi-Step Decision Experiment")
    print("="*60)
    print("\n核心目标: 验证预测选择 vs 重构选择")
    print("任务: 网格世界导航 (需要多步前瞻规划)")
    
    # 实验1: 预测导向选择
    print("\n[1] Running Prediction-based Selection...")
    pred_results = run_experiment('prediction', n_episodes=50, n_runs=10)
    
    # 实验2: 重构导向选择
    print("[2] Running Reconstruction-based Selection...")
    recon_results = run_experiment('reconstruction', n_episodes=50, n_runs=10)
    
    # 实验3: 随机基线
    print("[3] Running Random Baseline...")
    random_results = run_experiment('random', n_episodes=50, n_runs=10)
    
    # 结果对比
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print(f"\n{'Method':<25} {'Success Rate':<15} {'Mean Reward':<15} {'Recon Error':<15}")
    print("-"*70)
    print(f"{'Prediction':<25} {pred_results['success_rate']*100:<14.1f}% {pred_results['mean_reward']:<15.2f} {pred_results['mean_recon_error']:<15.4f}")
    print(f"{'Reconstruction':<25} {recon_results['success_rate']*100:<14.1f}% {recon_results['mean_reward']:<15.2f} {recon_results['mean_recon_error']:<15.4f}")
    print(f"{'Random':<25} {random_results['success_rate']*100:<14.1f}% {random_results['mean_reward']:<15.2f} {random_results['mean_recon_error']:<15.4f}")
    
    # 核心对比
    print("\n" + "="*60)
    print("CORE COMPARISON")
    print("="*60)
    
    pred_vs_recon_success = (pred_results['success_rate'] - recon_results['success_rate']) / max(recon_results['success_rate'], 0.01) * 100
    pred_vs_recon_reward = (pred_results['mean_reward'] - recon_results['mean_reward']) / max(abs(recon_results['mean_reward']), 0.01) * 100
    
    print(f"\nPrediction vs Reconstruction:")
    print(f"  Success Rate: {pred_vs_recon_success:+.1f}%")
    print(f"  Mean Reward: {pred_vs_recon_reward:+.1f}%")
    
    # 结论
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    
    if pred_results['success_rate'] > recon_results['success_rate']:
        print("\n[OK] Prediction-based selection SUCCEEDS in multi-step task!")
        print("  -> 预测选择的前瞻规划优势得到验证")
    else:
        print("\n[X] Prediction-based selection did NOT outperform")
        print("  -> 需要进一步优化预测选择策略")


if __name__ == "__main__":
    main()
