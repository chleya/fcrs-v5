# -*- coding: utf-8 -*-
"""
FCRS-v5 评估指标模块

双维度评估体系:
- 感知指标: 单步重构误差 (适用于静态特征提取)
- 智能指标: 多步预测误差、序列决策累计回报 (适用于前瞻规划)
"""

import numpy as np
from typing import List, Dict, Optional


class Metrics:
    """
    统一指标管理
    """
    
    def __init__(self):
        # 感知指标
        self.recon_errors = []
        self.pred_errors = []
        
        # 智能指标
        self.multi_step_errors = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        
        # 历史记录
        self.history = {
            'recon': [],
            'pred': [],
            'multi_step': [],
            'reward': [],
            'success': []
        }
    
    # ========== 感知指标 ==========
    
    def add_recon_error(self, error: float):
        """添加单步重构误差"""
        self.recon_errors.append(error)
        self.history['recon'].append(error)
        
        # 限制长度
        if len(self.recon_errors) > 1000:
            self.recon_errors = self.recon_errors[-1000:]
    
    def get_mean_recon_error(self) -> float:
        """平均重构误差"""
        if not self.recon_errors:
            return 0.0
        return np.mean(self.recon_errors[-100:])
    
    def add_pred_error(self, error: float):
        """添加单步预测误差"""
        self.pred_errors.append(error)
        self.history['pred'].append(error)
        
        if len(self.pred_errors) > 1000:
            self.pred_errors = self.pred_errors[-1000:]
    
    def get_mean_pred_error(self) -> float:
        """平均预测误差"""
        if not self.pred_errors:
            return 0.0
        return np.mean(self.pred_errors[-100:])
    
    # ========== 智能指标 ==========
    
    def add_multi_step_error(self, error: float):
        """添加多步预测误差"""
        self.multi_step_errors.append(error)
        self.history['multi_step'].append(error)
    
    def get_mean_multi_step_error(self) -> float:
        """平均多步预测误差"""
        if not self.multi_step_errors:
            return 0.0
        return np.mean(self.multi_step_errors[-100:])
    
    def add_episode_result(self, reward: float, length: int, success: bool):
        """添加一轮结果"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.successes.append(success)
        
        self.history['reward'].append(reward)
        self.history['success'].append(1.0 if success else 0.0)
    
    def get_mean_reward(self) -> float:
        """平均累计回报"""
        if not self.episode_rewards:
            return 0.0
        return np.mean(self.episode_rewards[-50:])
    
    def get_success_rate(self) -> float:
        """成功率"""
        if not self.successes:
            return 0.0
        return np.mean(self.successes[-50:])
    
    def get_mean_length(self) -> float:
        """平均步数"""
        if not self.episode_lengths:
            return 0.0
        return np.mean(self.episode_lengths[-50:])
    
    # ========== 综合报告 ==========
    
    def get_perception_report(self) -> Dict:
        """感知指标报告"""
        return {
            'mean_recon_error': self.get_mean_recon_error(),
            'mean_pred_error': self.get_mean_pred_error(),
            'std_recon_error': float(np.std(self.recon_errors[-100:])) if len(self.recon_errors) >= 100 else 0.0
        }
    
    def get_intelligence_report(self) -> Dict:
        """智能指标报告"""
        return {
            'mean_multi_step_error': self.get_mean_multi_step_error(),
            'mean_reward': self.get_mean_reward(),
            'success_rate': self.get_success_rate(),
            'mean_length': self.get_mean_length()
        }
    
    def get_full_report(self) -> Dict:
        """完整报告"""
        return {
            'perception': self.get_perception_report(),
            'intelligence': self.get_intelligence_report()
        }
    
    def print_report(self):
        """打印报告"""
        print("\n" + "="*60)
        print("Evaluation Report")
        print("="*60)
        
        print("\n[Perception Metrics]")
        p = self.get_perception_report()
        print(f"  Mean Recon Error: {p['mean_recon_error']:.4f}")
        print(f"  Mean Pred Error: {p['mean_pred_error']:.4f}")
        
        print("\n[Intelligence Metrics]")
        i = self.get_intelligence_report()
        print(f"  Mean Multi-Step Error: {i['mean_multi_step_error']:.4f}")
        print(f"  Mean Reward: {i['mean_reward']:.2f}")
        print(f"  Success Rate: {i['success_rate']*100:.1f}%")
        print(f"  Mean Length: {i['mean_length']:.1f}")


def compare_methods(method_results: Dict[str, Metrics], method_names: List[str] = None) -> Dict:
    """
    对比多个方法的指标
    
    Args:
        method_results: {方法名: Metrics对象}
        method_names: 要对比的方法名列表
    
    Returns:
        对比结果
    """
    if method_names is None:
        method_names = list(method_results.keys())
    
    comparison = {
        'perception': {},
        'intelligence': {}
    }
    
    for name in method_names:
        if name not in method_results:
            continue
        
        m = method_results[name]
        
        # 感知指标
        comparison['perception'][name] = {
            'recon_error': m.get_mean_recon_error(),
            'pred_error': m.get_mean_pred_error()
        }
        
        # 智能指标
        comparison['intelligence'][name] = {
            'multi_step_error': m.get_mean_multi_step_error(),
            'reward': m.get_mean_reward(),
            'success_rate': m.get_success_rate(),
            'length': m.get_mean_length()
        }
    
    return comparison


def print_comparison(comparison: Dict):
    """打印对比结果"""
    print("\n" + "="*60)
    print("Method Comparison")
    print("="*60)
    
    print("\n[Perception Metrics] (Lower is better)")
    print("-"*50)
    print(f"{'Method':<20} {'Recon Error':<15} {'Pred Error':<15}")
    print("-"*50)
    
    for name, metrics in comparison['perception'].items():
        print(f"{name:<20} {metrics['recon_error']:<15.4f} {metrics['pred_error']:<15.4f}")
    
    print("\n[Intelligence Metrics] (Multi-step/Success higher is better)")
    print("-"*50)
    print(f"{'Method':<20} {'Multi-Step Err':<15} {'Reward':<15} {'Success Rate':<15}")
    print("-"*50)
    
    for name, metrics in comparison['intelligence'].items():
        print(f"{name:<20} {metrics['multi_step_error']:<15.4f} {metrics['reward']:<15.2f} {metrics['success_rate']*100:<15.1f}%")


# ==================== 测试 ====================

def test_metrics():
    """测试指标模块"""
    print("="*60)
    print("Metrics Test")
    print("="*60)
    
    # 创建两个方法的指标
    m1 = Metrics()
    m2 = Metrics()
    
    # 模拟数据
    for i in range(100):
        # 方法1: 重构导向
        m1.add_recon_error(0.5 + np.random.randn() * 0.1)
        m1.add_pred_error(0.6 + np.random.randn() * 0.1)
        
        # 方法2: 预测导向
        m2.add_recon_error(0.6 + np.random.randn() * 0.1)
        m2.add_pred_error(0.4 + np.random.randn() * 0.1)
    
    # 智能指标
    for _ in range(20):
        m1.add_episode_result(5.0, 10, True)
        m2.add_episode_result(7.0, 8, True)
    
    # 对比
    comparison = compare_methods({'Recon': m1, 'Predictive': m2})
    print_comparison(comparison)


if __name__ == "__main__":
    test_metrics()
