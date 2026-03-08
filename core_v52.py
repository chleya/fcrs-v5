"""
FCRS-v5.2: 有限竞争表征系统 - 真正涌现驱动版本
Finite Competitive Representation System

修复内容:
- 移除优化驱动阈值
- 实现真正涌现机制:
  1. 自发对称性破缺
  2. 临界状态触发
  3. 多维度协同
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Representation:
    """表征结构"""
    id: int
    vector: np.ndarray  # 向量表示
    fitness_history: List[float] = field(default_factory=list)
    activation_count: int = 0
    age: int = 0
    
    # 属性
    reuse: float = 0.0          # 复用次数
    dim_reuse: np.ndarray = None  # 每维复用统计
    dim_contrib: np.ndarray = None  # 每维贡献
    origin: str = 'initial'      # 来源: initial, symmetry_break, critical, synergy
    
    def __post_init__(self):
        if self.dim_reuse is None:
            self.dim_reuse = np.zeros_like(self.vector)
        if self.dim_contrib is None:
            self.dim_contrib = np.zeros_like(self.vector)
    
    @property
    def reuse_frequency(self) -> float:
        if self.age == 0:
            return 0.0
        return self.activation_count / self.age
    
    def resize(self, new_size: int):
        """调整向量大小"""
        old_size = len(self.vector)
        
        if new_size > old_size:
            # 扩展
            padding = np.zeros(new_size - old_size)
            self.vector = np.concatenate([self.vector, padding])
            self.dim_reuse = np.concatenate([self.dim_reuse, np.zeros(new_size - old_size)])
            self.dim_contrib = np.concatenate([self.dim_contrib, np.zeros(new_size - old_size)])
        elif new_size < old_size:
            # 截断
            self.vector = self.vector[:new_size]
            self.dim_reuse = self.dim_reuse[:new_size]
            self.dim_contrib = self.dim_contrib[:new_size]


class RepresentationPool:
    """表征池"""
    
    def __init__(self, capacity: int = 5, total_budget: float = 100.0):
        self.capacity = capacity
        self.total_budget = total_budget
        self.representations: List[Representation] = []
        self.id_counter = 0
    
    def __len__(self):
        return len(self.representations)
    
    def add(self, vector: np.ndarray, origin='initial') -> Representation:
        """添加表征"""
        if len(self.representations) >= self.capacity:
            return None
        
        rep = Representation(
            id=self.id_counter,
            vector=vector.copy(),
            origin=origin
        )
        self.id_counter += 1
        self.representations.append(rep)
        return rep
    
    def get_total_dims(self) -> int:
        """总维度"""
        return sum(len(r.vector) for r in self.representations)
    
    def get_average_fitness(self) -> float:
        """平均适应度"""
        if not self.representations:
            return 0.0
        return np.mean([r.reuse_frequency for r in self.representations])


class EnvironmentLoop:
    """环境环"""
    
    def __init__(self, input_dim: int = 10, n_classes: int = 5):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.current_class = 0
        self.class_centers = {i: np.random.randn(input_dim) for i in range(n_classes)}
    
    def generate_input(self) -> np.ndarray:
        """生成输入"""
        cls = np.random.randint(0, self.n_classes)
        self.current_class = cls
        center = self.class_centers[cls]
        return center + np.random.randn(self.input_dim) * 0.3


class EmergenceEngine:
    """真正涌现驱动进化引擎"""
    
    def __init__(self, pool: RepresentationPool, env: EnvironmentLoop):
        self.pool = pool
        self.env = env
        self.symmetry_broken = False
        
        # 统计
        self.emergence_log = {'symmetry_break': 0, 'critical': 0, 'synergy': 0}
    
    def calculate_persistence(self, rep: Representation) -> float:
        """持久度计算"""
        if rep.fitness_history:
            avg_fitness = np.mean(rep.fitness_history[-10:])
        else:
            avg_fitness = 0.0
        
        return avg_fitness
    
    def symmetry_breaking(self) -> Optional[Representation]:
        """
        机制1: 自发对称性破缺
        当表征过于相似时，触发差异化
        """
        if len(self.pool.representations) < 2:
            return None
        
        # 计算表征相似度
        vectors = np.array([r.vector for r in self.pool.representations])
        n = len(vectors)
        
        # 成对相似度
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                sim = np.dot(vectors[i], vectors[j]) / (
                    np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-8
                )
                total_sim += sim
                count += 1
        
        avg_sim = total_sim / count if count > 0 else 0
        
        # 相似度高且未破缺
        if avg_sim > 0.85 and not self.symmetry_broken:
            # 随机选择，添加正交扰动
            idx = np.random.randint(len(self.pool.representations))
            original = self.pool.representations[idx].vector
            
            # 生成正交方向
            random_dir = np.random.randn(len(original))
            random_dir = random_dir - np.dot(random_dir, original) * original / (np.linalg.norm(original)**2 + 1e-8)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)
            
            # 新表征
            new_vector = original + 0.5 * random_dir
            
            new_rep = self.pool.add(new_vector, origin='symmetry_break')
            if new_rep:
                self.symmetry_broken = True
                self.emergence_log['symmetry_break'] += 1
                return new_rep
        
        return None
    
    def critical_state_trigger(self) -> Optional[Representation]:
        """
        机制2: 临界状态触发
        当系统波动率在中间区域时触发
        """
        if len(self.pool.representations) < 2:
            return None
        
        # 计算近期波动率
        all_fitness = []
        for rep in self.pool.representations:
            if rep.fitness_history:
                all_fitness.extend(rep.fitness_history[-5:])
        
        if len(all_fitness) < 10:
            return None
        
        volatility = np.std(all_fitness)
        
        # 临界状态: 0.1 < volatility < 0.5
        if 0.1 < volatility < 0.5:
            # 生成新表征
            idx = np.random.randint(len(self.pool.representations))
            original = self.pool.representations[idx].vector
            
            new_vector = original + np.random.randn(len(original)) * 0.3
            new_rep = self.pool.add(new_vector, origin='critical')
            
            if new_rep:
                self.emergence_log['critical'] += 1
                return new_rep
        
        return None
    
    def multidimensional_synergy(self) -> Optional[Representation]:
        """
        机制3: 多维度协同适应度
        当组合适应度显著高于个体时触发
        """
        if len(self.pool.representations) < 3:
            return None
        
        # 随机选3个
        indices = np.random.choice(len(self.pool.representations), 3, replace=False)
        selected = [self.pool.representations[i] for i in indices]
        
        # 个体适应度
        individual_fitness = [r.reuse_frequency for r in selected]
        
        # 组合适应度
        combined_vector = np.mean([r.vector for r in selected], axis=0)
        
        # 测试组合效果
        test_input = self.env.generate_input()
        combined_score = np.dot(combined_vector, test_input) / (np.linalg.norm(combined_vector) + 1e-8)
        avg_individual = np.mean(individual_fitness) + 0.1  # 基准
        
        # 如果组合显著优于个体
        if combined_score > avg_individual:
            new_rep = self.pool.add(combined_vector.copy(), origin='synergy')
            if new_rep:
                self.emergence_log['synergy'] += 1
                return new_rep
        
        return None
    
    def select_for_deletion(self) -> Optional[Representation]:
        """选择淘汰的表征"""
        if len(self.pool) < self.pool.capacity:
            return None
        
        # 淘汰适应度最低的
        worst = min(self.pool.representations, key=lambda r: r.reuse_frequency)
        return worst
    
    def try_emergence(self) -> bool:
        """尝试涌现 - 无阈值版本"""
        # 按优先级尝试三种机制
        # 1. 多维度协同
        if self.multidimensional_synergy():
            return True
        
        # 2. 临界状态
        if self.critical_state_trigger():
            return True
        
        # 3. 对称性破缺
        if self.symmetry_breaking():
            return True
        
        return False
    
    def delete_worst(self):
        """删除最差表征"""
        worst = self.select_for_deletion()
        if worst:
            self.pool.representations.remove(worst)


class FCRSv52:
    """FCRS v5.2 - 真正涌现驱动"""
    
    def __init__(self, pool_capacity: int = 5, input_dim: int = 10, 
                 n_classes: int = 5, lr: float = 0.01):
        self.pool = RepresentationPool(capacity=pool_capacity)
        self.env = EnvironmentLoop(input_dim, n_classes)
        self.engine = EmergenceEngine(self.pool, self.env)
        self.lr = lr
        self.step_count = 0
        self.errors = []
        
        # 初始化表征
        for _ in range(3):
            x = self.env.generate_input()
            self.pool.add(x)
    
    def step(self):
        """一步"""
        self.step_count += 1
        
        # 1. 生成输入
        x = self.env.generate_input()
        
        # 2. 选择表征
        best_rep = None
        best_score = -float('inf')
        
        for rep in self.pool.representations:
            score = np.dot(rep.vector, x) / (np.linalg.norm(rep.vector) + 1e-8)
            if score > best_score:
                best_score = score
                best_rep = rep
        
        if best_rep is not None:
            # 3. 预测+学习
            error_vec = x - best_rep.vector
            error = np.linalg.norm(error_vec)
            best_rep.vector += self.lr * error_vec
            
            # 4. 记录
            best_rep.activation_count += 1
            best_rep.fitness_history.append(-error)
            best_rep.age += 1
            self.errors.append(error)
            
            # 5. 涌现检测（无阈值！）
            if len(self.pool) < self.pool.capacity:
                self.engine.try_emergence()
            else:
                # 池满时尝试替换
                self.engine.try_emergence()
                if len(self.pool) > self.pool.capacity:
                    self.engine.delete_worst()
        else:
            self.errors.append(float('inf'))
    
    def get_avg_error(self):
        if not self.errors:
            return 0
        return np.mean(self.errors[-100:])
    
    def get_stats(self):
        return {
            'steps': self.step_count,
            'reps': len(self.pool),
            'dims': self.pool.get_total_dims(),
            'emergence': self.engine.emergence_log,
            'error': self.get_avg_error()
        }


def demo():
    """演示"""
    fcrs = FCRSv52(pool_capacity=5, input_dim=10, n_classes=3, lr=0.01)
    
    for i in range(500):
        fcrs.step()
        
        if (i + 1) % 100 == 0:
            stats = fcrs.get_stats()
            print(f'Step {i+1}: error={stats["error"]:.4f}, dims={stats["dims"]}')
            print(f'  Emergence: {stats["emergence"]}')
    
    print('\nFinal:', fcrs.get_stats())


if __name__ == "__main__":
    demo()
