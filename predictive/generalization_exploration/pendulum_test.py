# -*- coding: utf-8 -*-
"""
FCRS-v5 Pendulum 任务适配

Pendulum特点:
- 摆锤摆动
- 目标: 摆动到直立位置
- 连续动作空间
"""

import numpy as np


class PendulumEnv:
    """Pendulum环境简化版"""
    
    def __init__(self):
        self.state = None
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
    
    def reset(self):
        self.state = np.array([
            np.random.uniform(-np.pi, np.pi),  # theta
            np.random.uniform(-1, 1)  # omega
        ])
        return self.state.copy()
    
    def step(self, action: int) -> tuple:
        theta, omega = self.state
        
        # 动作映射: 0=-2, 1=0, 2=+2
        torque = (action - 1) * self.max_torque
        
        # 动力学
        d_theta = omega
        d_omega = (torque - self.m * self.g * self.l * np.sin(theta)) / (self.m * self.l**2)
        
        theta += d_theta * self.dt
        omega += d_omega * self.dt
        
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        omega = np.clip(omega, -self.max_speed, self.max_speed)
        
        self.state = np.array([theta, omega])
        
        # 奖励: 接近直立(0)且低角速度
        angle = np.abs(theta)
        reward = -(angle**2 + 0.1 * omega**2)
        
        # 完成条件
        done = angle < 0.1
        
        return self.state.copy(), reward, done
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def action_dim(self):
        return 3


class FCRSPendulum:
    """FCRS-v5 适配 Pendulum"""
    
    def __init__(self, input_dim=2, pool_capacity=5, lr=0.01, explore=0.3):
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
    
    def run_episode(self, env, mode='prediction', max_steps=200):
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


def test_pendulum():
    print("="*60)
    print("FCRS-v5 Pendulum Test")
    print("="*60)
    
    env = PendulumEnv()
    
    modes = ['prediction', 'reconstruction', 'random']
    results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        
        fcrs = FCRSPendulum(input_dim=2, pool_capacity=5, lr=0.01, explore=0.3)
        
        for ep in range(100):
            reward = fcrs.run_episode(env, mode=mode, max_steps=200)
            if ep % 20 == 0:
                print(f"  Episode {ep}: {reward:.1f}")
        
        test_rewards = [fcrs.run_episode(env, mode=mode, max_steps=200) for _ in range(10)]
        
        results[mode] = {
            'test_mean': np.mean(test_rewards),
            'success': sum(1 for r in test_rewards if r > -10)
        }
        
        print(f"  Test mean: {np.mean(test_rewards):.1f}, Success: {results[mode]['success']}/10")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<15}: {r['test_mean']:.1f}, Success: {r['success']}/10")


if __name__ == "__main__":
    test_pendulum()
