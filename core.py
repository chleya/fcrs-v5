"""
FCRS-v5.0: 有限竞争表征系统
Finite Competitive Representation System

核心模块：
1. EnvironmentLoop - 环境环
2. RepresentationPool - 表征池
3. EvolutionEngine - 进化引擎
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Representation:
    """表征结构"""
    id: int
    vector: np.ndarray  # 向量表示
    fitness_history: List[float] = field(default_factory=list)  # 适应度历史
    activation_count: int = 0  # 激活次数
    age: int = 0  # 年龄
    
    @property
    def reuse_frequency(self) -> float:
        """复用频率 = 激活次数 / 年龄"""
        if self.age == 0:
            return 0.0
        return self.activation_count / self.age


class EnvironmentLoop:
    """环境环 - 产生问题和反馈"""
    
    def __init__(self, input_dim: int = 10):
        self.input_dim = input_dim
        
    def generate_input(self) -> np.ndarray:
        """生成输入信号"""
        # TODO: 实现输入生成
        return np.random.randn(self.input_dim)
    
    def calculate_error(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """计算预测误差"""
        return float(np.linalg.norm(prediction - target))


class RepresentationPool:
    """表征池 - 存储和管理候选表征"""
    
    def __init__(self, capacity: int = 50, vector_dim: int = 10):
        self.capacity = capacity  # 最大容量N
        self.vector_dim = vector_dim
        self.representations: List[Representation] = []
        self.next_id = 0
        
    def add(self, vector: np.ndarray) -> Representation:
        """添加新表征"""
        rep = Representation(
            id=self.next_id,
            vector=vector,
            age=0
        )
        self.next_id += 1
        self.representations.append(rep)
        return rep
    
    def select(self, input_vector: np.ndarray) -> Optional[Representation]:
        """竞争性选择：选择适应度最高的表征"""
        if not self.representations:
            return None
        
        # 简单策略：选择向量最接近的
        best = None
        best_score = float('-inf')
        
        for rep in self.representations:
            # 点积作为相似度
            score = np.dot(rep.vector, input_vector)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def __len__(self):
        return len(self.representations)
    
    def __repr__(self):
        return f"RepresentationPool({len(self)}/{self.capacity})"


class EvolutionEngine:
    """进化引擎 - 变异-选择-保留"""
    
    def __init__(self, pool: RepresentationPool):
        self.pool = pool
        self.alpha = 0.5  # 适应度权重
        self.beta = 0.3   # 复用频率权重
        self.gamma = 0.2  # 资源成本权重
        
    def calculate_persistence(self, rep: Representation) -> float:
        """
        持久度计算:
        P = α * 平均适应度 + β * 复用频率 - γ * 成本
        """
        # 平均适应度
        if rep.fitness_history:
            avg_fitness = np.mean(rep.fitness_history[-10:])  # 最近10步
        else:
            avg_fitness = 0.0
        
        # 复用频率
        reuse = rep.reuse_frequency
        
        # 资源成本（简单用向量维度）
        cost = rep.vector.shape[0] / 100.0
        
        persistence = (self.alpha * avg_fitness + 
                      self.beta * reuse - 
                      self.gamma * cost)
        
        return persistence
    
    def mutate(self, rep: Representation, strength: float = 0.1) -> np.ndarray:
        """变异操作"""
        # 变异幅度与年龄成反比
        mutation_strength = strength / (1 + rep.age * 0.1)
        
        # 高斯变异
        noise = np.random.randn(*rep.vector.shape) * mutation_strength
        new_vector = rep.vector + noise
        
        return new_vector
    
    def select_for_deletion(self) -> Optional[Representation]:
        """选择要淘汰的表征"""
        if len(self.pool) < self.pool.capacity:
            return None  # 池未满，不需要淘汰
        
        # 选择持久度最低的
        worst = None
        worst_persistence = float('inf')
        
        for rep in self.pool.representations:
            p = self.calculate_persistence(rep)
            if p < worst_persistence:
                worst_persistence = p
                worst = rep
        
        return worst


class FCRSystem:
    """有限竞争表征系统 - 主系统"""
    
    def __init__(self, 
                 pool_capacity: int = 50,
                 vector_dim: int = 10,
                 input_dim: int = 10):
        
        # 初始化三模块
        self.env = EnvironmentLoop(input_dim)
        self.pool = RepresentationPool(pool_capacity, vector_dim)
        self.engine = EvolutionEngine(self.pool)
        
        # 统计
        self.step_count = 0
        
    def step(self):
        """执行一步"""
        self.step_count += 1
        
        # 1. 环境产生输入
        x_t = self.env.generate_input()
        
        # 2. 表征池选择激活
        active_rep = self.pool.select(x_t)
        
        if active_rep is None:
            # 池为空，添加新表征
            self.pool.add(x_t)
            return
        
        # 3. 更新激活计数
        active_rep.activation_count += 1
        
        # 4. 计算适应度
        prediction = active_rep.vector
        error = self.env.calculate_error(prediction, x_t)
        fitness = -error  # 误差越小，适应度越高
        active_rep.fitness_history.append(fitness)
        
        # 5. 年龄增加
        for rep in self.pool.representations:
            rep.age += 1
        
        # 6. 进化引擎 - 变异
        if np.random.random() < 0.1:  # 10%变异率
            new_vector = self.engine.mutate(active_rep)
            
            # 如果池满了，淘汰一个
            if len(self.pool) >= self.pool.capacity:
                to_delete = self.engine.select_for_deletion()
                if to_delete:
                    self.pool.representations.remove(to_delete)
            
            # 添加新表征
            self.pool.add(new_vector)
    
    def run(self, steps: int = 1000):
        """运行多步"""
        for _ in range(steps):
            self.step()
            
        return self.get_statistics()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'step': self.step_count,
            'pool_size': len(self.pool),
            'representations': []
        }
        
        for rep in self.pool.representations:
            stats['representations'].append({
                'id': rep.id,
                'age': rep.age,
                'activation_count': rep.activation_count,
                'reuse_frequency': rep.reuse_frequency,
                'persistence': self.engine.calculate_persistence(rep)
            })
        
        return stats


# 测试
if __name__ == "__main__":
    print("FCRS-v5.0 有限竞争表征系统")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=20, vector_dim=10)
    
    # 运行100步
    stats = system.run(100)
    
    print(f"运行步数: {stats['step']}")
    print(f"表征池大小: {stats['pool_size']}")
    print(f"\n表征详情:")
    
    for rep in stats['representations']:
        print(f"  ID:{rep['id']} 年龄:{rep['age']} "
              f"复用:{rep['reuse_frequency']:.2f} "
              f"持久度:{rep['persistence']:.2f}")
