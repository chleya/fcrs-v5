# -*- coding: utf-8 -*-
"""
FCRS-v5 CartPole 任务适配

连续控制任务: CartPole-v1
- 状态: [位置, 速度, 角度, 角速度]
- 动作: 左/右
- 目标: 保持杆直立
"""

import numpy as np
import sys
import os

# 尝试导入gym
try:
    import gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False
    print("Warning: gym not installed")

from typing import Tuple, Optional


class CartPoleEnv:
    """
    CartPole环境封装
    
    适配FCRS-v5的接口
    """
    
    def __init__(self):
        if HAS_GYM:
            self.env = gym.make('CartPole-v1')
        else:
            # 简单模拟环境
            self.env = SimpleCartPole()
        
        # 状态维度
        self.input_dim = 4  # [x, x_dot, theta, theta_dot]
        self.action_dim = 2  # 左/右
        
        # 用于压缩的归一化参数
        self.state_mean = np.zeros(4)
        self.state_std = np.ones(4)
    
    def generate_input(self) -> np.ndarray:
        """生成输入 (归一化状态)"""
        if HAS_GYM:
            state = self.env.reset()
        else:
            state = self.env.reset()
        
        # 归一化
        state = (state - self.state_mean) / (self.state_std + 1e-8)
        return state.astype(np.float32)
    
    def reset(self) -> np.ndarray:
        if HAS_GYM:
            state = self.env.reset()
        else:
            state = self.env.reset()
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if HAS_GYM:
            state, reward, done, _ = self.env.step(action)
        else:
            state, reward, done = self.env.step(action)
        
        return state.astype(np.float32), reward, done
    
    def get_state(self) -> np.ndarray:
        if HAS_GYM:
            return self.env.state
        return self.env.state


class SimpleCartPole:
    """简单CartPole模拟"""
    
    def __init__(self):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_pole + self.mass_cart
        self.length = 0.5  # actually half the pole's length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        self.state = None
        self.steps_beyond_done = None
    
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(-0.05, 0.05, 4)
        self.steps_beyond_done = None
        return np.array(self.state)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        state = self.state
        x, x_dot, theta, theta_dot = state
        
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        # 动力学
        temp = (force + self.mass_pole * self.length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.mass_pole * costheta**2 / self.total_mass))
        xacc = temp - self.mass_pole * thetaacc * costheta / self.total_mass
        
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        # 边界
        done = x < -2.4 or x > 2.4 or theta < -0.2095 or theta > 0.2095
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            reward = 0.0
        
        return np.array(self.state), reward, done


class FCRSCartPole:
    """
    FCRS-v5 适配 CartPole
    """
    
    def __init__(self, input_dim=4, compress_dim=2, pool_capacity=5, lr=0.01):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.pool_capacity = pool_capacity
        self.lr = lr
        
        # 表征池
        self.reps = [np.random.randn(input_dim) * 0.1 for _ in range(pool_capacity)]
        
        # 预测器
        self.predictors = [np.eye(input_dim) * 0.5 for _ in range(pool_capacity)]
        
        # 策略网络
        self.W_policy = np.random.randn(input_dim, 2) * 0.1
        self.b_policy = np.zeros(2)
        
        # 历史
        self.state_history = []
        self.explore = 0.3
        
        # 统计
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_count = 0
    
    def select_representation(self, state: np.ndarray, mode='prediction') -> int:
        """选择表征"""
        if mode == 'prediction':
            errors = []
            for p in self.predictors:
                if len(self.state_history) > 0:
                    pred = self.state_history[-1] @ p
                    error = np.linalg.norm(pred - state)
                else:
                    error = 1.0
                errors.append(error)
            return np.argmin(errors)
        else:  # reconstruction
            return np.argmin([np.linalg.norm(r - state) for r in self.reps])
    
    def select_action(self, representation: np.ndarray) -> int:
        """选择动作"""
        q_values = representation @ self.W_policy + self.b_policy
        
        if np.random.random() < self.explore:
            return np.random.randint(2)
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, rep_idx: int):
        """更新"""
        # 表征更新
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        self.reps[rep_idx] /= (np.linalg.norm(self.reps[rep_idx]) + 1e-8)
        
        # 预测器更新
        if len(self.state_history) > 0:
            pred = self.state_history[-1] @ self.predictors[rep_idx]
            error = next_state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.state_history[-1], error)
        
        # 策略更新
        rep = self.reps[rep_idx]
        q_values = rep @ self.W_policy + self.b_policy
        
        for a in range(2):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward - q_values[a]) * rep
                self.b_policy[a] += self.lr * (reward - q_values[a])
            else:
                self.W_policy[:, a] -= 0.01 * q_values[a] * rep
        
        # 历史
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def run_episode(self, env: CartPoleEnv, mode='prediction', max_steps=500) -> int:
        """运行一轮"""
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # 选择表征
            rep_idx = self.select_representation(state, mode)
            
            # 选择动作
            action = self.select_action(self.reps[rep_idx])
            
            # 执行
            next_state, reward, done = env.step(action)
            
            # 更新
            self.update(state, action, reward, next_state, rep_idx)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        return total_reward


def test_cartpole():
    """测试CartPole"""
    print("="*60)
    print("FCRS-v5 CartPole Test")
    print("="*60)
    
    # 环境
    env = CartPoleEnv()
    
    # 测试三种模式
    modes = ['prediction', 'reconstruction', 'random']
    results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        
        # 创建系统
        fcrs = FCRSCartPole(
            input_dim=4,
            compress_dim=2,
            pool_capacity=5,
            lr=0.01
        )
        
        # 训练
        for ep in range(100):
            reward = fcrs.run_episode(env, mode=mode, max_steps=200)
            if ep % 20 == 0:
                print(f"  Episode {ep}: {reward}")
        
        # 测试
        test_rewards = []
        for _ in range(10):
            reward = fcrs.run_episode(env, mode=mode, max_steps=200)
            test_rewards.append(reward)
        
        results[mode] = {
            'train_last': reward,
            'test_mean': np.mean(test_rewards)
        }
        
        print(f"  Test mean: {np.mean(test_rewards):.1f}")
    
    # 结果对比
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<15}: Train={r['train_last']:.1f}, Test={r['test_mean']:.1f}")


if __name__ == "__main__":
    test_cartpole()
