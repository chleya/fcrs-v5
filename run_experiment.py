# -*- coding: utf-8 -*-
"""
FCRS-v5.3.0 核心实验一键复现脚本

固定随机种子，100%复现里程碑结果：
预测选择 vs 重构选择 +22% 成功率提升

Usage:
    python run_experiment.py
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# 导入模块
from predictive.src.core.grid_world import GridWorldEnv
from predictive.src.experiments.gridworld_v2 import RepresentationAgent

# 可视化
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_experiment(selection_mode: str, n_episodes: int = 100, n_runs: int = 10, seed: int = 20260309):
    """
    运行实验
    
    Args:
        selection_mode: 'prediction', 'reconstruction', 'random'
        n_episodes: 训练轮数
        n_runs: 重复次数
        seed: 随机种子
    
    Returns:
        结果字典
    """
    results = {
        'success_rates': [],
        'mean_rewards': [],
        'mean_lengths': [],
        'mean_recon_errors': [],
    }
    
    for run in range(n_runs):
        np.random.seed(seed + run)
        
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
                # 表征选择
                rep_idx = agent.select_representation(state, explore=0.2)
                
                # 动作选择 (基于表征)
                action = agent.select_action(agent.reps[rep_idx], explore=0.2)
                
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
        'success_rate': np.mean(results['success_rates']) * 100,
        'std_success': np.std(results['success_rates']) * 100,
        'mean_reward': np.mean(results['mean_rewards']),
        'mean_length': np.mean(results['mean_lengths']),
        'mean_recon_error': np.mean(results['mean_recon_errors']),
    }


def main():
    """主函数"""
    print("="*60)
    print("FCRS-v5.3.0 核心实验一键复现")
    print("固定随机种子: 20260309")
    print("="*60)
    
    # 固定种子
    SEED = 20260309
    
    # 三种模式
    modes = ['prediction', 'reconstruction', 'random']
    results = {}
    
    for mode in modes:
        print(f"\n正在运行 {mode} 模式...")
        results[mode] = run_experiment(mode, n_episodes=100, n_runs=10, seed=SEED)
        print(f"  成功率: {results[mode]['success_rate']:.1f}%")
        print(f"  累计回报: {results[mode]['mean_reward']:.2f}")
        print(f"  重构误差: {results[mode]['mean_recon_error']:.4f}")
    
    # 输出表格
    print("\n" + "="*60)
    print("最终复现结果对比")
    print("="*60)
    print(f"{'模式':<20} {'成功率':<12} {'累计回报':<12} {'重构误差':<12}")
    print("-"*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<20} {r['success_rate']:<11.1f}% {r['mean_reward']:<12.2f} {r['mean_recon_error']:<12.4f}")
    
    # 核心突破验证
    print("\n" + "="*60)
    print("核心突破验证")
    print("="*60)
    
    pred_sr = results['prediction']['success_rate']
    recon_sr = results['reconstruction']['success_rate']
    random_sr = results['random']['success_rate']
    
    print(f"预测选择 vs 重构选择 成功率提升: +{pred_sr - recon_sr:.1f}%")
    print(f"预测选择 vs 随机选择 成功率提升: +{pred_sr - random_sr:.1f}%")
    
    if pred_sr > recon_sr:
        print("\n[成功] 预测选择在多步决策任务中优于重构选择!")
    else:
        print("\n[警告] 结果需要进一步验证")
    
    # ========== 可视化 ==========
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 成功率柱状图
    ax1 = axes[0]
    modes_display = ['Prediction', 'Reconstruction', 'Random']
    colors = ['#2563eb', '#dc2626', '#6b7280']
    success_rates = [results['prediction']['success_rate'], 
                    results['reconstruction']['success_rate'], 
                    results['random']['success_rate']]
    
    bars1 = ax1.bar(modes_display, success_rates, color=colors)
    ax1.set_title('Task Success Rate (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel('Success Rate (%)')
    
    # 添加数值标签
    for bar, rate in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 累计回报柱状图
    ax2 = axes[1]
    rewards = [results['prediction']['mean_reward'], 
              results['reconstruction']['mean_reward'], 
              results['random']['mean_reward']]
    
    bars2 = ax2.bar(modes_display, rewards, color=colors)
    ax2.set_title('Average Cumulative Reward', fontsize=12)
    ax2.set_ylabel('Cumulative Reward')
    
    # 添加数值标签
    for bar, reward in zip(bars2, rewards):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{reward:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('fcrs_v53_results.png', dpi=300, bbox_inches='tight')
    print("结果可视化图已保存为 fcrs_v53_results.png")


if __name__ == "__main__":
    main()
