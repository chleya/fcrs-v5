"""
FCRS-v5.0: 有限竞争表征系统 v5.1
Finite Competitive Representation System

核心模块：
1. EnvironmentLoop - 环境环
2. RepresentationPool - 表征池  
3. EvolutionEngine - 进化引擎

v5.1新增：
- 新维度诞生机制
- 残差信号
- 维度级竞争
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
    
    # v5.1新增属性
    reuse: float = 0.0          # 复用次数（累积）
    dim_reuse: np.ndarray = None  # 每维复用统计
    dim_contrib: np.ndarray = None  # 每维贡献
    compression_gain: float = 0.0  # 压缩增益
    
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
            # 扩展：补零
            self.vector = np.append(self.vector, np.zeros(new_size - old_size))
            self.dim_reuse = np.append(self.dim_reuse, np.zeros(new_size - old_size))
            self.dim_contrib = np.append(self.dim_contrib, np.zeros(new_size - old_size))
        elif new_size < old_size:
            # 收缩：截断
            self.vector = self.vector[:new_size]
            self.dim_reuse = self.dim_reuse[:new_size]
            self.dim_contrib = self.dim_contrib[:new_size]


class EnvironmentLoop:
    """环境环 - 产生问题和反馈"""
    
    def __init__(self, input_dim: int = 10):
        self.input_dim = input_dim
        # v5.1: 多类别环境
        self.num_classes = 3
        self.class_centers = {
            i: np.random.randn(input_dim) * 2 
            for i in range(self.num_classes)
        }
        self.current_class = 0
        
    def generate_input(self) -> np.ndarray:
        """生成输入信号"""
        self.current_class = np.random.randint(0, self.num_classes)
        x = self.class_centers[self.current_class] + np.random.randn(self.input_dim) * 0.3
        return x
    
    def get_observation(self) -> np.ndarray:
        """获取观测（generate_input的别名）"""
        return self.generate_input()
    
    def calculate_error(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """计算预测误差"""
        # v5.1: 支持不同长度向量
        min_len = min(len(prediction), len(target))
        if min_len == 0:
            return 1.0
        return float(np.linalg.norm(prediction[:min_len] - target[:min_len]))
    
    # v5.1新增：计算残差
    def compute_residual(self, x: np.ndarray, pool_prediction: np.ndarray) -> np.ndarray:
        """计算残差：当前表征联合解释不了的部分"""
        min_len = min(len(x), len(pool_prediction))
        residual = x[:min_len] - pool_prediction[:min_len]
        return residual
    
    # v5.1新增：获取输入和残差
    def get_input_and_residual(self, pool_prediction: np.ndarray = None):
        """获取输入和残差信号"""
        x = self.generate_input()
        
        if pool_prediction is None:
            pool_prediction = np.zeros_like(x)
        
        residual = self.compute_residual(x, pool_prediction)
        return x, residual


class RepresentationPool:
    """表征池 - 存储和管理候选表征"""
    
    def __init__(self, capacity: int = 50, vector_dim: int = 10):
        self.capacity = capacity
        self.vector_dim = vector_dim
        self.representations: List[Representation] = []
        self.next_id = 0
        self.total_budget = 100.0  # v5.1: 维度预算
        
    def add(self, vector: np.ndarray, dim_cost: float = 0.0) -> Representation:
        """添加新表征"""
        rep = Representation(
            id=self.next_id,
            vector=np.array(vector, dtype=np.float64),
            age=0
        )
        self.next_id += 1
        self.representations.append(rep)
        
        if dim_cost > 0:
            self.total_budget -= dim_cost
        
        return rep
    
    def select(self, input_vector: np.ndarray) -> Optional[Representation]:
        """竞争性选择"""
        if not self.representations:
            return None
        
        best = None
        best_score = float('-inf')
        
        for rep in self.representations:
            # v5.1: 支持不同长度向量
            min_len = min(len(rep.vector), len(input_vector))
            score = np.dot(rep.vector[:min_len], input_vector[:min_len])
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def activate(self, x: np.ndarray) -> tuple:
        """激活并返回（激活值，激活的表征）"""
        if not self.representations:
            return 0.0, None
        
        # 选择激活的表征
        active = self.select(x)
        if active is None:
            return 0.0, None
        
        # 计算激活值
        min_len = min(len(active.vector), len(x))
        act = np.dot(active.vector[:min_len], x[:min_len])
        
        # v5.1: 更新维度统计
        if min_len > 0:
            active.dim_contrib += np.abs(active.vector[:min_len] * x[:min_len]) * act
            active.dim_reuse += (np.abs(active.vector[:min_len]) > 0.01).astype(float) * act
        
        return act, active
    
    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        """v5.1: 池子联合预测（加权求和）"""
        if not self.representations:
            return np.zeros_like(x)
        
        # 简单重构：所有表征的加权和
        weights = []
        for r in self.representations:
            min_len = min(len(r.vector), len(x))
            if min_len > 0:
                w = np.dot(r.vector[:min_len], x[:min_len])
                weights.append(max(w, 0))
            else:
                weights.append(0)
        
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        # 加权求和
        result = np.zeros(len(x))
        for w, r in zip(weights, self.representations):
            min_len = min(len(r.vector), len(x))
            result[:min_len] += w * r.vector[:min_len]
        
        return result
    
    def get_top_reps(self, k: int = 3) -> List[Representation]:
        """v5.1: 获取top-K高复用表征"""
        sorted_reps = sorted(
            self.representations, 
            key=lambda r: r.reuse, 
            reverse=True
        )
        return sorted_reps[:k]
    
    def get_total_dims(self) -> int:
        """v5.1: 获取总维度"""
        return sum(len(r.vector) for r in self.representations)
    
    def __len__(self):
        return len(self.representations)
    
    def __repr__(self):
        return f"RepresentationPool({len(self)}/{self.capacity}, 预算:{self.total_budget:.1f})"


class EvolutionEngine:
    """进化引擎 - 变异-选择-保留 + 新维度机制"""
    
    def __init__(self, pool: RepresentationPool, env: EnvironmentLoop):
        self.pool = pool
        self.env = env
        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 0.2
        
        # v5.1: 新维度诞生参数
        self.spawn_reuse_threshold = 5  # 进一步降低
        self.min_compression_gain = 0.01  # 进一步降低
        self.dim_cost = 1.0  # 每加1维扣1单位预算
        
        # 统计
        self.new_dim_history = []  # 新维度诞生历史
    
    def calculate_persistence(self, rep: Representation) -> float:
        """持久度计算"""
        if rep.fitness_history:
            avg_fitness = np.mean(rep.fitness_history[-10:])
        else:
            avg_fitness = 0.0
        
        reuse = rep.reuse_frequency
        cost = len(rep.vector) / 100.0
        
        return self.alpha * avg_fitness + self.beta * reuse - self.gamma * cost
    
    def mutate(self, rep: Representation, strength: float = 0.1) -> np.ndarray:
        """变异操作"""
        mutation_strength = strength / (1 + rep.age * 0.1)
        
        min_len = len(rep.vector)
        noise = np.random.randn(min_len) * mutation_strength
        new_vector = rep.vector + noise
        
        return new_vector
    
    def select_for_deletion(self) -> Optional[Representation]:
        """选择要淘汰的表征"""
        if len(self.pool) < self.pool.capacity:
            return None
        
        worst = None
        worst_persistence = float('inf')
        
        for rep in self.pool.representations:
            p = self.calculate_persistence(rep)
            if p < worst_persistence:
                worst_persistence = p
                worst = rep
        
        return worst
    
    # v5.1: 新维度诞生！
    def try_spawn_new_dim(self, rep: Representation, recent_residuals: np.ndarray) -> bool:
        """尝试为高复用表征诞生新维度"""
        # 检查阈值
        if rep.reuse < self.spawn_reuse_threshold:
            return False
        
        if len(recent_residuals) == 0:
            return False
        
        old_v = rep.vector.copy()
        old_len = len(old_v)
        
        # 计算当前残差 - 始终使用old_len
        if recent_residuals.ndim > 1:
            # 确保不会超出范围
            actual_len = min(recent_residuals.shape[1], old_len)
            residual_mean = np.mean(recent_residuals[:, :actual_len], axis=0)
            # 如果不够，补零
            if actual_len < old_len:
                residual_mean = np.pad(residual_mean, (0, old_len - actual_len))
        else:
            actual_len = min(len(recent_residuals), old_len)
            residual_mean = recent_residuals[:actual_len]
            if actual_len < old_len:
                residual_mean = np.pad(residual_mean, (0, old_len - actual_len))
        
        # 计算旧误差
        old_error = np.mean(np.abs(residual_mean)) + 1e-8
        
        # 从残差方向衍生新维度
        pca_dir = residual_mean / (np.linalg.norm(residual_mean) + 1e-8)
        new_dim = pca_dir * 0.1 + np.random.normal(0, 0.05, old_len)
        
        # 构建新向量（加1维）
        new_v = np.append(old_v, [np.mean(new_dim)])
        
        # 临时计算新误差
        new_error = np.mean(np.abs(residual_mean - new_v[:old_len]))
        
        # 计算压缩增益
        compression_gain = (old_error - new_error) / (old_error + 1e-8)
        
        if compression_gain > self.min_compression_gain and self.pool.total_budget >= self.dim_cost:
            # 通过测试，采纳新维度
            rep.vector = new_v
            rep.reuse += 5  # 奖励
            rep.compression_gain = compression_gain
            self.pool.total_budget -= self.dim_cost
            
            self.new_dim_history.append({
                'step': self.env.current_class,  # 借用一下
                'rep_id': rep.id,
                'gain': compression_gain,
                'new_dim': len(new_v)
            })
            
            print(f"v 新维度诞生! 压缩增益={compression_gain:.3f}, 总维={len(new_v)}")
            return True
        else:
            return False
    
    # v5.1: 维度级竞争（剪枝低贡献维度）
    def prune_low_contrib_dims(self):
        """定期清理低贡献维度"""
        pruned_count = 0
        
        for r in self.pool.representations:
            if len(r.vector) <= 1:
                continue
            
            # 找出贡献<5%的维度
            max_contrib = np.max(r.dim_contrib) if np.max(r.dim_contrib) > 0 else 1.0
            useless_mask = r.dim_contrib < (max_contrib * 0.05)
            
            if np.any(useless_mask):
                # 置零（不删除，保留索引）
                r.vector[useless_mask] = 0.0
                pruned_count += 1
        
        if pruned_count > 0:
            print(f"  维度清理: {pruned_count}个低贡献维度置零")


class FCRSystem:
    """有限竞争表征系统 - 主系统"""
    
    def __init__(self, 
                 pool_capacity: int = 20,
                 vector_dim: int = 10,
                 input_dim: int = 10):
        
        # 初始化三模块
        self.env = EnvironmentLoop(input_dim)
        self.pool = RepresentationPool(pool_capacity, vector_dim)
        self.engine = EvolutionEngine(self.pool, self.env)
        
        # 统计
        self.step_count = 0
        self.recent_residuals = []  # v5.1: 最近残差
        self.dim_history = []  # v5.1: 维度历史
    
    def step(self):
        """执行一步"""
        self.step_count += 1
        
        # v5.1: 获取输入和残差
        x_t, residual = self.env.get_input_and_residual()
        
        # 存残差
        self.recent_residuals.append(residual)
        if len(self.recent_residuals) > 100:
            self.recent_residuals.pop(0)
        
        # 表征池选择激活
        act, active_rep = self.pool.activate(x_t)
        
        if active_rep is None:
            self.pool.add(x_t)
        else:
            # 更新统计
            active_rep.activation_count += 1
            active_rep.reuse += 1  # v5.1: 复用计数
            
            # 计算适应度
            error = self.env.calculate_error(active_rep.vector, x_t)
            fitness = -error
            active_rep.fitness_history.append(fitness)
        
        # 年龄增加
        for rep in self.pool.representations:
            rep.age += 1
        
        # 进化引擎 - 变异
        if np.random.random() < 0.05:  # v5.1: 降低变异率到5%
            if active_rep:
                new_vector = self.engine.mutate(active_rep)
                
                # 淘汰
                if len(self.pool) >= self.pool.capacity:
                    to_delete = self.engine.select_for_deletion()
                    if to_delete:
                        self.pool.representations.remove(to_delete)
                
                # 添加
                self.pool.add(new_vector)
        
        # v5.1: 尝试诞生新维度（对top-3表征）
        if self.step_count % 10 == 0:  # 每10步尝试一次
            top_reps = self.pool.get_top_reps(k=3)
            residuals_array = np.array(self.recent_residuals)
            for r in top_reps:
                self.engine.try_spawn_new_dim(r, residuals_array)
        
        # v5.1: 维度级清理（每500步）
        if self.step_count % 500 == 0:
            self.engine.prune_low_contrib_dims()
        
        # 记录维度历史
        if self.step_count % 100 == 0:
            self.dim_history.append(self.pool.get_total_dims())
    
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
            'total_dims': self.pool.get_total_dims(),
            'budget': self.pool.total_budget,
            'new_dims_born': len(self.engine.new_dim_history),
            'dim_history': self.dim_history,
            'representations': []
        }
        
        for rep in self.pool.representations:
            stats['representations'].append({
                'id': rep.id,
                'dim': len(rep.vector),
                'age': rep.age,
                'reuse': rep.reuse,
                'compression_gain': rep.compression_gain
            })
        
        return stats


# 测试
if __name__ == "__main__":
    print("FCRS-v5.1 有限竞争表征系统 (新维度版)")
    print("=" * 50)
    
    # 创建系统
    system = FCRSystem(pool_capacity=10, vector_dim=10)
    
    # 运行1000步
    print("\n运行1000步...")
    stats = system.run(1000)
    
    print(f"\n结果:")
    print(f"  步数: {stats['step']}")
    print(f"  表征池: {stats['pool_size']}")
    print(f"  总维度: {stats['total_dims']}")
    print(f"  预算: {stats['budget']:.1f}")
    print(f"  新维度诞生: {stats['new_dims_born']}")
    print(f"  维度历史: {stats['dim_history']}")
    
    print(f"\n表征详情:")
    for rep in stats['representations'][:5]:
        print(f"  ID:{rep['id']} 维:{rep['dim']} 复用:{rep['reuse']:.0f}")
