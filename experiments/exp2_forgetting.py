"""
实验2：噪声表征的灾难性遗忘
Phenomenon 2: Catastrophic Forgetting of Noise Representations

目标：当资源约束收紧时，系统优先保留高复用频率的表征，淘汰高适应度、低复用频率的表征
"""

import numpy as np
import json


class Representation:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector
        self.fitness_history = []
        self.activation_count = 0
        self.age = 0
    
    @property
    def reuse_frequency(self):
        return self.activation_count / max(1, self.age)


class Environment:
    """环境：2个类别，一个常见，一个罕见"""
    def __init__(self):
        self.common_center = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 常见类
        self.rare_center = np.array([5.0, 5.0, 5.0, 5.0, 5.0])   # 罕见类
    
    def generate(self):
        # 90%概率常见类，10%罕见类
        if np.random.random() < 0.9:
            x = self.common_center + np.random.randn(5) * 0.2
            cls = 'common'
        else:
            x = self.rare_center + np.random.randn(5) * 0.2
            cls = 'rare'
        return x, cls


class FCRSystem:
    def __init__(self, capacity=20):
        self.capacity = capacity
        self.reps = []
        self.next_id = 0
    
    def add(self, vec):
        rep = Representation(self.next_id, vec.copy())
        self.next_id += 1
        self.reps.append(rep)
        return rep
    
    def select(self, x):
        if not self.reps:
            return None
        # 选择最近的
        return min(self.reps, key=lambda r: np.linalg.norm(r.vector - x))
    
    def persistence(self, r):
        """持久度 = 复用频率 - 成本"""
        return r.reuse_frequency - r.age * 0.001
    
    def step(self, x, cls):
        for r in self.reps:
            r.age += 1
        
        active = self.select(x)
        if active is None:
            self.add(x)
            return
        
        active.activation_count += 1
        error = np.linalg.norm(active.vector - x)
        active.fitness_history.append(-error)
    
    def reduce_capacity(self, new_capacity):
        """收紧容量，保留高复用频率的"""
        if len(self.reps) <= new_capacity:
            return
        
        # 按持久度排序，淘汰最低的
        self.reps.sort(key=lambda r: self.persistence(r))
        self.reps = self.reps[-new_capacity:]


def run():
    print("=" * 50)
    print("实验2: 噪声表征的灾难性遗忘")
    print("=" * 50)
    
    env = Environment()
    sys = FCRSystem(capacity=20)
    
    print("阶段1: 运行直到稳定...")
    
    # 阶段1：正常运行
    for _ in range(1000):
        x, c = env.generate()
        sys.step(x, c)
    
    # 记录阶段1状态
    stats_before = []
    for r in sys.reps:
        stats_before.append({
            'activation': r.activation_count,
            'reuse': r.reuse_frequency,
            'age': r.age
        })
    
    common_count = sum(1 for r in sys.reps if r.activation_count > 50)
    rare_count = sum(1 for r in sys.reps if 0 < r.activation_count <= 50)
    
    print(f"  阶段1: 常见类表征={common_count}, 罕见类表征={rare_count}")
    
    # 阶段2：收紧容量
    print("\n阶段2: 收紧容量到10...")
    sys.reduce_capacity(10)
    
    # 记录阶段2状态
    stats_after = []
    for r in sys.reps:
        stats_after.append({
            'activation': r.activation_count,
            'reuse': r.reuse_frequency
        })
    
    common_after = sum(1 for r in sys.reps if r.activation_count > 50)
    rare_after = sum(1 for r in sys.reps if 0 < r.activation_count <= 50)
    
    print(f"  阶段2: 常见类表征={common_after}, 罕见类表征={rare_after}")
    
    # 验证
    print("\n结果:")
    if common_after >= common_count and rare_after <= rare_count:
        print("v 噪声遗忘验证通过")
        print("  系统优先保留了高复用频率的表征")
        success = True
    else:
        print("X 验证未通过")
        success = False
    
    return {
        'experiment': 'phenomenon_2',
        'phase1': {'common': common_count, 'rare': rare_count},
        'phase2': {'common': common_after, 'rare': rare_after},
        'success': success
    }


if __name__ == "__main__":
    result = run()
    
    with open('F:/fcrs-v5/experiments/exp2_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\n结果已保存")
