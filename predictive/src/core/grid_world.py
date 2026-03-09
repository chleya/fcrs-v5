# -*- coding: utf-8 -*-
"""
网格世界环境 - 多步序列决策任务
用于验证预测选择的前瞻规划价值
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    网格世界导航任务
    
    特点:
    - 有状态转移 (每步行动改变状态)
    - 有奖励机制 (到达目标+, 撞墙-)
    - 需要多步前瞻规划
    """
    
    def __init__(self, size: int = 5, n_obstacles: int = 3, seed: int = 42):
        self.size = size
        self.n_obstacles = n_obstacles
        self.seed = seed
        
        np.random.seed(seed)
        
        # 位置
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        
        # 障碍物
        self.obstacles = set()
        while len(self.obstacles) < n_obstacles:
            pos = (np.random.randint(0, size), np.random.randint(0, size))
            if pos != self.start and pos != self.goal:
                self.obstacles.add(pos)
        
        # 当前位置
        self.pos = self.start
        
        # 动作: 上、下、左、右
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.n_actions = len(self.actions)
        
        # 状态维度
        self.state_dim = size * size  # 位置编码
    
    def reset(self) -> np.ndarray:
        """重置到起点"""
        self.pos = self.start
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """获取当前位置的one-hot编码"""
        state = np.zeros(self.size * self.size)
        idx = self.pos[0] * self.size + self.pos[1]
        state[idx] = 1.0
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """执行一步"""
        dx, dy = self.actions[action]
        
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy
        
        # 边界检查
        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            # 撞墙
            reward = -1.0
            done = False
        elif (new_x, new_y) in self.obstacles:
            # 撞障碍物
            reward = -1.0
            done = False
        elif (new_x, new_y) == self.goal:
            # 到达目标
            self.pos = (new_x, new_y)
            reward = 10.0
            done = True
        else:
            # 正常移动
            self.pos = (new_x, new_y)
            reward = -0.1  # 每步小惩罚，鼓励尽快到达
            done = False
        
        return self.get_state(), reward, done
    
    def get_distance_to_goal(self) -> float:
        """到目标的曼哈顿距离"""
        return abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1])
    
    def is_at_goal(self) -> bool:
        return self.pos == self.goal
    
    def render(self) -> str:
        """可视化"""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = '#'
        
        grid[self.start[1]][self.start[0]] = 'S'
        grid[self.goal[1]][self.goal[0]] = 'G'
        grid[self.pos[1]][self.pos[0]] = '@'
        
        return '\n'.join([' '.join(row) for row in grid])


class GridWorldEnv:
    """
    网格世界环境封装
    适配现有系统接口
    """
    
    def __init__(self, size: int = 5, n_obstacles: int = 3):
        self.size = size
        self.grid = GridWorld(size, n_obstacles)
        
        # 状态维度 = 位置编码维度
        self.input_dim = size * size
        self.action_dim = 4
    
    def generate_input(self) -> np.ndarray:
        """生成输入 (当前位置)"""
        return self.grid.get_state()
    
    def reset(self) -> np.ndarray:
        return self.grid.reset()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        return self.grid.step(action)
    
    def get_reward(self) -> float:
        """获取当前奖励"""
        if self.grid.is_at_goal():
            return 10.0
        elif self.grid.get_distance_to_goal() == 1:
            return 1.0
        else:
            return -0.1
    
    def run_episode(self, agent, max_steps: int = 50) -> dict:
        """运行一轮"""
        state = self.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            # Agent选择动作
            action = agent.select_action(state)
            
            # 环境执行
            next_state, reward, done = self.step(action)
            
            # Agent学习
            agent.learn(state, action, reward, next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'success': self.grid.is_at_goal(),
            'distance_final': self.grid.get_distance_to_goal()
        }


class SimpleAgent:
    """简单Agent (用于测试环境)"""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 随机策略
        self.q_table = np.random.randn(state_dim, action_dim) * 0.1
    
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """ε-贪心选择"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        
        # Q值最大的动作
        state_idx = np.argmax(state)
        return np.argmax(self.q_table[state_idx])
    
    def learn(self, state: int, action: int, reward: float, next_state: int):
        """Q学习更新"""
        alpha = 0.1
        gamma = 0.9
        
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        self.q_table[state, action] += alpha * (reward + gamma * max_next_q - current_q)


# ==================== 测试 ====================

def test_gridworld():
    """测试网格世界"""
    print("="*60)
    print("GridWorld Test")
    print("="*60)
    
    env = GridWorld(size=5, n_obstacles=3)
    
    print("Initial:")
    print(env.render())
    
    # 随机走10步
    for i in range(10):
        action = np.random.randint(4)
        state, reward, done = env.step(action)
        print(f"\nStep {i+1}, Action {action}, Reward {reward}")
        print(env.render())
        
        if done:
            print("Goal reached!")
            break
    
    # 测试环境封装
    print("\n" + "="*60)
    print("Environment Wrapper Test")
    print("="*60)
    
    env2 = GridWorldEnv(size=5, n_obstacles=3)
    agent = SimpleAgent(env2.input_dim, env2.action_dim)
    
    results = []
    for ep in range(10):
        result = env2.run_episode(agent, max_steps=50)
        results.append(result)
    
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_reward = np.mean([r['total_reward'] for r in results])
    avg_steps = np.mean([r['steps'] for r in results])
    
    print(f"\nSuccess Rate: {success_rate*100:.1f}%")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg Steps: {avg_steps:.1f}")


if __name__ == "__main__":
    test_gridworld()
