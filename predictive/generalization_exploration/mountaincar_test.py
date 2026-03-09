# -*- coding: utf-8 -*-
"""
FCRS-v5 MountainCar 任务适配

MountainCar特点:
- 状态: 位置(-1.2~0.6), 速度
- 动作: 左/右/不动
- 目标: 摆动到山顶(位置>0.5)
- 需要多步规划才能成功
"""

import numpy as np


class MountainCarEnv:
    """MountainCar环境"""
    
    def __init__(self):
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        
        self.action_force = 0.001
        self.gravity = 0.0025
        
        self.state = None
    
    def reset(self) -> np.ndarray:
        self.state = np.array([np.random.uniform(-0.6, -0.4), 0.0])
        return self.state.copy()
    
    def step(self, action: int) -> tuple:
        position, velocity = self.state
        
        # 动作映射: 0=左, 1=不动, 2=右
        if action == 0:
            force = -self.action_force
        elif action == 2:
            force = self.action_force
        else:
            force = 0
        
        # 动力学
        velocity += force * np.cos(3 * position) - self.gravity * np.sin(3 * position)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        
        self.state = np.array([position, velocity])
        
        # 奖励: 每步-1, 到达目标+100
        done = position >= self.goal_position
        reward = -1.0 if not done else 100.0
        
        return self.state.copy(), reward, done
    
    @property
    def input_dim(self):
        return 2
    
    @property
    def action_dim(self):
        return 3


class FCRSMountainCar:
    """FCRS-v5 适配 MountainCar"""
    
    def __init__(self, input_dim=2, pool_capacity=5, lr=0.01, explore=0.3):
        self.input_dim = input_dim
        self.pool_capacity = pool_capacity
        self.lr = lr
        self.explore = explore
        
        # 表征池
        self.reps = [np.random.randn(input_dim) * 0.1 for _ in range(pool_capacity)]
        
        # 预测器
        self.predictors = [np.eye(input_dim) * 0.5 for _ in range(pool_capacity)]
        
        # 策略网络
        self.W_policy = np.random.randn(input_dim, 3) * 0.1
        self.b_policy = np.zeros(3)
        
        self.state_history = []
        
        self.episode_rewards = []
    
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
        else:  # reconstruction
            return np.argmin([np.linalg.norm(r - state) for r in self.reps])
    
    def select_action(self, representation: np.ndarray) -> int:
        q_values = representation @ self.W_policy + self.b_policy
        if np.random.random() < self.explore:
            return np.random.randint(3)
        return np.argmax(q_values)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, rep_idx: int):
        
        # 表征更新
        self.reps[rep_idx] += self.lr * (state - self.reps[rep_idx])
        
        # 预测器更新
        if len(self.state_history) > 0:
            pred = self.state_history[-1] @ self.predictors[rep_idx]
            error = next_state - pred
            self.predictors[rep_idx] += self.lr * np.outer(self.state_history[-1], error)
        
        # 策略更新
        rep = self.reps[rep_idx]
        q_values = rep @ self.W_policy + self.b_policy
        
        for a in range(3):
            if a == action:
                self.W_policy[:, a] += self.lr * (reward - q_values[a]) * rep
                self.b_policy[a] += self.lr * (reward - q_values[a])
            else:
                self.W_policy[:, a] -= 0.01 * q_values[a] * rep
        
        self.state_history.append(state)
        if len(self.state_history) > 100:
            self.state_history = self.state_history[-100:]
    
    def run_episode(self, env: MountainCarEnv, mode='prediction', max_steps=1000) -> int:
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            rep_idx = self.select_representation(state, mode)
            action = self.select_action(self.reps[rep_idx])
            
            next_state, reward, done = env.step(action)
            
            self.update(state, action, reward, next_state, rep_idx)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        return total_reward


def test_mountaincar():
    print("="*60)
    print("FCRS-v5 MountainCar Test")
    print("="*60)
    
    env = MountainCarEnv()
    
    modes = ['prediction', 'reconstruction', 'random']
    results = {}
    
    for mode in modes:
        print(f"\n[{mode}] Running...")
        
        fcrs = FCRSMountainCar(
            input_dim=2,
            pool_capacity=5,
            lr=0.01,
            explore=0.3
        )
        
        # 训练
        for ep in range(100):
            reward = fcrs.run_episode(env, mode=mode, max_steps=1000)
            if ep % 20 == 0:
                print(f"  Episode {ep}: {reward:.1f}")
        
        # 测试
        test_rewards = []
        for _ in range(10):
            reward = fcrs.run_episode(env, mode=mode, max_steps=1000)
            test_rewards.append(reward)
        
        results[mode] = {
            'train_last': reward,
            'test_mean': np.mean(test_rewards),
            'test_success': sum(1 for r in test_rewards if r > 0)
        }
        
        print(f"  Test mean: {np.mean(test_rewards):.1f}, Success: {results[mode]['test_success']}/10")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for mode in modes:
        r = results[mode]
        print(f"{mode:<15}: Train={r['train_last']:.1f}, Test={r['test_mean']:.1f}, Success={r['test_success']}/10")


if __name__ == "__main__":
    test_mountaincar()
