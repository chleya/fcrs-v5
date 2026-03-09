# -*- coding: utf-8 -*-
"""
FCRS-v5 Acrobot 任务适配

Acrobot特点:
- 双杆摆动系统
- 目标: 摆动到一定高度
- 需要多步规划和能量积累
"""

import numpy as np


class AcrobotEnv:
    """Acrobot环境简化版"""
    
    def __init__(self):
        # 状态: [theta1, theta2, dtheta1, dtheta2]
        self.state = None
        self.link_length_1 = 1.0
        self.link_length_2 = 1.0
        self.link_mass_1 = 1.0
        self.link_mass_2 = 1.0
        self.gravity = 9.8
        self.tau = 0.05
        
        # 目标高度
        self.goal_height = 1.0  # 杆尖高度
    
    def reset(self):
        # 随机初始状态
        self.state = np.array([
            np.random.uniform(-np.pi, 0),  # theta1
            np.random.uniform(-np.pi, np.pi),  # theta2
            np.random.uniform(-0.1, 0.1),  # dtheta1
            np.random.uniform(-0.1, 0.1)   # dtheta2
        ])
        return self.state.copy()
    
    def step(self, action: int) -> tuple:
        theta1, theta2, dtheta1, dtheta2 = self.state
        
        # 动作: -1, 0, +1 ( torque )
        torque = (action - 1) * 0.5  # -0.5, 0, 0.5
        
        # 简化的动力学
        m1, m2 = self.link_mass_1, self.link_mass_2
        l1, l2 = self.link_length_1, self.link_length_2
        
        # 简化的力矩计算
        d2 = m2 * (l1 * l2 * dtheta1**2 * np.sin(theta2) + 
                   0.5 * m2 * l1 * l2 * dtheta2**2 * np.sin(theta2))
        
        dtheta1 += (torque - d2) / (m1 + m2)
        dtheta2 += -dtheta1
        
        # 限制
        dtheta1 = np.clip(dtheta1, -4, 4)
        dtheta2 = np.clip(dtheta2, -9, 9)
        
        theta1 += dtheta1 * self.tau
        theta2 += dtheta2 * self.tau
        
        self.state = np.array([theta1, theta2, dtheta1, dtheta2])
        
        # 计算高度
        x = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
        height = -l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2)
        
        # 奖励
        done = height > self.goal_height
        reward = -1.0 if not done else 100.0
        
        return self.state.copy(), reward, done
    
    @property
    def input_dim(self):
        return 4
    
    @property
    def action_dim(self):
        return 3


class FCRSAcrobot:
    """FCRS-v5 适配 Acrobot"""
    
    def __init__(self, input_dim=4, pool_capacity=5, lr=0.01, explore=0.3):
        self.input_dim = input_dim
        self.pool_capacity = pool_capacity
        self.lr = lr
        self.explore = explore
        
        self.reps = [np.random.randn(input_dim) * 0.1 for _ in range(pool_capacity)]
        self.predictors = [np.eye(input_dim) * 0.5 for _ in range(pool_capacity)]
        self.W_policy = np.random.randn(input_dim, 3) * 0.1
        self.b_policy = np.zeros(3)
        
        self.state_history = []
    
    def select_representation(self, state: np.ndarray, mode='prediction') -> int:
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
        return np.argmin([np.linalg.norm(r - state) for r in self.reps])
    
    def select_action(self, rep: np.ndarray) -> int:
        v = rep @ self.W_policy + self.b_policy
        if np.random.random() < self.explore:
            return np.random.randint(3)
        return np.argmax(v)
    
    def update(self, state, action, reward, next_state, rep_idx):
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        
        if len(self.state_history) > 0:
            pred = self.state_history[-1] @ self.predictors[rep_idx]
            error = next_state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.state_history[-1], error)
        
        v = self.reps[rep_idx] @ self.W_policy + self.b_policy
        for a in range(3):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward - v[a]) * self.reps[rep_idx]
                self.b_policy[a] += self.lr * (reward - v[a])
        
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def run_episode(self, env, mode='prediction', max_steps=500):
        state = env.reset()
        total_reward = 0
        
        for _ in range(max_steps):
            idx = self.select_representation(state, mode)
            action = self.select_action(self.reps[idx])
            next_state, reward, done = env.step(action)
            self.update(state, action, reward, next_state, idx)
            state = next_state
            total_reward += reward
            if done:
                break
        
        return total_reward


def test_acrobot():
    print("="*60)
    print("FCRS-v5 Acrobot Test")
    print("="*60)
    
    env = AcrobotEnv()
    
    modes = ['prediction', 'reconstruction', 'random']
    results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        
        fcrs = FCRSAcrobot(input_dim=4, pool_capacity=5, lr=0.01, explore=0.3)
        
        for ep in range(50):
            reward = fcrs.run_episode(env, mode=mode, max_steps=500)
            if ep % 10 == 0:
                print(f"  Episode {ep}: {reward:.1f}")
        
        test_rewards = [fcrs.run_episode(env, mode=mode, max_steps=500) for _ in range(10)]
        
        results[mode] = {
            'test_mean': np.mean(test_rewards),
            'success': sum(1 for r in test_rewards if r > 0)
        }
        
        print(f"  Test mean: {np.mean(test_rewards):.1f}, Success: {results[mode]['success']}/10")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<15}: {r['test_mean']:.1f}, Success: {r['success']}/10")


if __name__ == "__main__":
    test_acrobot()
