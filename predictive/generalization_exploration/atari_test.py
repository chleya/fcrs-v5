# -*- coding: utf-8 -*-
"""
FCRS-v5 Atari视觉任务适配

目标: 验证FCRS在高维视觉输入下的表现
"""

import numpy as np


class AtariWrapper:
    """简化Atari环境封装"""
    def __init__(self, game_name='pong', seed=42):
        self.game_name = game_name
        np.random.seed(seed)
        
        # 简化: 使用随机生成的"游戏画面"
        self.frame_size = (84, 84)  # 标准Atari分辨率
        self.state = None
        self.score = 0
        self.step_count = 0
        self.max_steps = 1000
        
        # 模拟游戏状态
        self.ball_pos = np.array([0.5, 0.5])
        self.ball_vel = np.array([0.02, 0.015])
        self.paddle_pos = 0.5
    
    def reset(self):
        self.state = self._get_frame()
        self.score = 0
        self.step_count = 0
        return self.state
    
    def _get_frame(self):
        """生成模拟帧 (84x84灰度)"""
        frame = np.random.randint(0, 256, self.frame_size, dtype=np.uint8)
        
        # 添加球 (白色区域)
        bx, by = int(self.ball_pos[0] * 84), int(self.ball_pos[1] * 84)
        if 0 <= bx < 84 and 0 <= by < 84:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= bx+dx < 84 and 0 <= by+dy < 84:
                        frame[by+dy, bx+dx] = 255
        
        # 添加挡板 (左侧白色区域)
        py = int(self.paddle_pos * 84)
        for dy in range(-10, 10):
            if 0 <= py+dy < 84:
                frame[py+dy, 5] = 255
        
        return frame
    
    def step(self, action):
        # 动作: 0=不动, 1=上, 2=下
        if action == 1:
            self.paddle_pos = max(0.1, self.paddle_pos - 0.05)
        elif action == 2:
            self.paddle_pos = min(0.9, self.paddle_pos + 0.05)
        
        # 球运动
        self.ball_pos += self.ball_vel
        if self.ball_pos[0] <= 0.05:
            self.ball_vel[0] *= -1
            self.score += 1  # 得分
        if self.ball_pos[0] >= 0.95:
            self.ball_vel[0] *= -1
        if self.ball_pos[1] <= 0 or self.ball_pos[1] >= 1:
            self.ball_vel[1] *= -1
        
        # 碰撞检测
        if self.ball_pos[0] < 0.1 and abs(self.ball_pos[1] - self.paddle_pos) < 0.15:
            self.ball_vel[0] *= -1
            self.ball_pos[0] = 0.1
        
        self.state = self._get_frame()
        self.step_count += 1
        
        reward = 1.0 if self.ball_pos[0] <= 0.05 else -0.01
        done = self.step_count >= self.max_steps or self.score >= 10
        
        return self.state, reward, done
    
    @property
    def input_dim(self):
        return self.frame_size[0] * self.frame_size[1]  # 7056
    
    @property
    def action_dim(self):
        return 3  # 不动/上/下


class FCRSAtari:
    """FCRS适配Atari (使用CNN简化版)"""
    
    def __init__(self, input_dim=7056, action_dim=3, n_reps=5, lr=0.01):
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.n_reps = n_reps
        self.lr = lr
        
        # 降维层 (简化为线性投影)
        self.W_down = np.random.randn(input_dim, 64) * 0.01
        self.b_down = np.zeros(64)
        
        # 表征池 (64维)
        self.reps = [np.random.randn(64) * 0.1 for _ in range(n_reps)]
        self.predictors = [np.eye(64) * 0.5 for _ in range(n_reps)]
        
        # 策略
        self.W = np.random.randn(64, action_dim) * 0.1
        self.b = np.zeros(action_dim)
        
        self.history = []
        self.explore = 0.3
    
    def _preprocess(self, frame):
        """预处理: 展平 + 降维"""
        x = frame.flatten().astype(np.float32) / 255.0
        x = x @ self.W_down + self.b_down
        x = np.tanh(x)  # 激活
        return x
    
    def act(self, state):
        x = self._preprocess(state)
        
        # 预测导向选择表征
        if self.history:
            errors = [np.linalg.norm(self.history[-1] @ p - x) for p in self.predictors]
            idx = np.argmin(errors)
        else:
            idx = 0
        
        # 动作选择
        q = self.reps[idx] @ self.W + self.b
        action = np.random.randint(self.action_dim) if np.random.random() < self.explore else np.argmax(q)
        
        return action, idx
    
    def update(self, state, action, reward, next_state, idx):
        x = self._preprocess(state)
        nx = self._preprocess(next_state)
        
        # 表征更新
        self.reps[idx] += self.lr * (x - self.reps[idx])
        
        # 预测器更新
        if self.history:
            pred = self.history[-1] @ self.predictors[idx]
            self.predictors[idx] += self.lr * np.outer(self.history[-1], nx - pred)
        
        # 策略更新
        q = self.reps[idx] @ self.W + self.b
        for a in range(self.action_dim):
            if a == action:
                self.W[:, a] += self.lr * (reward - q[a]) * self.reps[idx]
                self.b[a] += self.lr * (reward - q[a])
        
        self.history.append(x)


class RandomAgent:
    def __init__(self, *args, **kwargs):
        pass
    
    def act(self, state):
        return np.random.randint(3), 0
    
    def update(self, *args):
        pass


def run_atari(AgentClass, n_ep=50):
    """运行Atari实验"""
    np.random.seed(100)
    env = AtariWrapper('pong')
    agent = AgentClass(7056, 3)
    
    # 训练
    for ep in range(n_ep):
        state = env.reset()
        total_reward = 0
        
        for _ in range(200):  # 每轮最多200步
            action, idx = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, idx)
            state = next_state
            total_reward += reward
            if done:
                break
        
        if ep % 10 == 0:
            print(f"  Episode {ep}: reward={total_reward:.1f}")
    
    # 测试
    test_rewards = []
    for _ in range(10):
        state = env.reset()
        total_reward = 0
        
        for _ in range(200):
            action, _ = agent.act(state)
            state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break
        
        test_rewards.append(total_reward)
    
    return np.mean(test_rewards)


def main():
    print("="*60)
    print("FCRS-v5 Atari Visual Input Test")
    print("="*60)
    print("Task: Pong (simplified)")
    print("Input: 84x84 = 7056 dims\n")
    
    print("[FCRS] Training...")
    fcrs_reward = run_atari(FCRSAtari)
    print(f"  Test reward: {fcrs_reward:.1f}")
    
    print("\n[Random] Training...")
    random_reward = run_atari(RandomAgent)
    print(f"  Test reward: {random_reward:.1f}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"FCRS:   {fcrs_reward:.1f}")
    print(f"Random: {random_reward:.1f}")
    
    if fcrs_reward > random_reward:
        print(f"\nFCRS outperforms Random by {fcrs_reward - random_reward:+.1f}")
    else:
        print("\nNote: Atari requires more training or CNN")


if __name__ == "__main__":
    main()
