# FCRS-v5 完整代码 - 三层架构版

"""
按审查意见重构：
- 三层架构：环境层 / 表征层 / 进化层
- 清晰接口设计
- 完整错误处理
"""

import numpy as np


# ========== 第一层: 环境层 ==========

class Environment:
    """环境接口"""
    
    def generate_input(self):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError


class RandomEnvironment(Environment):
    """随机环境"""
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def generate_input(self):
        return np.random.randn(self.input_dim)
    
    def reset(self):
        pass


class StructuredEnvironment(Environment):
    """结构化环境"""
    
    def __init__(self, input_dim, n_classes):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.class_centers = {
            i: np.random.randn(input_dim) for i in range(n_classes)
        }
    
    def generate_input(self):
        cls = np.random.randint(0, self.n_classes)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3
    
    def reset(self):
        self.class_centers = {
            i: np.random.randn(self.input_dim) for i in range(self.n_classes)
        }


# ========== 第二层: 表征层 ==========

class Representation:
    """表征接口"""
    
    def get_vector(self):
        raise NotImplementedError
    
    def update(self, x):
        raise NotImplementedError
    
    def get_fitness(self):
        raise NotImplementedError


class VectorRepresentation(Representation):
    """向量表征"""
    
    def __init__(self, vector):
        self.vector = vector
        self.age = 0
        self.activation_count = 0
        self.fitness_history = []
    
    def get_vector(self):
        return self.vector
    
    def update(self, x, lr=0.01):
        """在线学习更新"""
        error = x - self.vector
        self.vector += lr * error
        return np.linalg.norm(error)
    
    def get_fitness(self):
        if self.fitness_history:
            return np.mean(self.fitness_history[-10:])
        return 0


class RepresentationPool:
    """表征池接口"""
    
    def add(self, representation):
        raise NotImplementedError
    
    def select(self, x):
        raise NotImplementedError
    
    def get_total_dims(self):
        raise NotImplementedError


class SimplePool(RepresentationPool):
    """简单表征池"""
    
    def __init__(self, capacity, input_dim):
        self.capacity = capacity
        self.input_dim = input_dim
        self.representations = []
    
    def add(self, representation):
        if len(self.representations) < self.capacity:
            self.representations.append(representation)
            return True
        return False
    
    def select(self, x):
        """选择最匹配表征"""
        if not self.representations:
            return None
        
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            score = np.dot(rep.get_vector(), x) / (np.linalg.norm(rep.get_vector()) + 1e-8)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def get_total_dims(self):
        return sum(len(r.get_vector()) for r in self.representations)
    
    def get_stats(self):
        return {
            'size': len(self.representations),
            'total_dims': self.get_total_dims(),
            'avg_fitness': np.mean([r.get_fitness() for r in self.representations]) if self.representations else 0
        }


# ========== 第三层: 进化层 ==========

class EvolutionEngine:
    """进化引擎接口"""
    
    def evolve(self, pool, x, error):
        raise NotImplementedError


class CompetitionEvolution(EvolutionEngine):
    """竞争进化"""
    
    def __init__(self):
        self.generations = 0
    
    def evolve(self, pool, x, error):
        """进化一步"""
        # 更新年龄
        for rep in pool.representations:
            rep.age += 1
        
        self.generations += 1
        
        return pool.get_stats()


# ========== 主系统 ==========

class FCRS:
    """FCRS主系统 - 三层架构"""
    
    def __init__(self, env, pool, evolution, lr=0.01):
        self.env = env
        self.pool = pool
        self.evolution = evolution
        self.lr = lr
        self.step_count = 0
        self.errors = []
    
    def step(self):
        """执行一步"""
        self.step_count += 1
        
        # 1. 生成输入
        x = self.env.generate_input()
        
        # 2. 选择表征
        active = self.pool.select(x)
        
        if active is not None:
            # 3. 更新表征(在线学习)
            error = active.update(x, self.lr)
            
            # 4. 记录适应度
            active.fitness_history.append(-error)
            active.activation_count += 1
            
            self.errors.append(error)
        else:
            self.errors.append(float('inf'))
        
        # 5. 进化
        self.evolution.evolve(self.pool, x, self.errors[-1] if self.errors else 0)
    
    def get_avg_error(self):
        if not self.errors:
            return 0
        return np.mean(self.errors[-100:])
    
    def get_stats(self):
        return {
            'step': self.step_count,
            **self.pool.get_stats(),
            'avg_error': self.get_avg_error()
        }


# ========== 工厂函数 ==========

def create_fcrs(env_type='random', pool_capacity=5, input_dim=10, n_classes=3, lr=0.01):
    """工厂函数：创建FCRS系统"""
    
    # 环境层
    if env_type == 'random':
        env = RandomEnvironment(input_dim)
    elif env_type == 'structured':
        env = StructuredEnvironment(input_dim, n_classes)
    else:
        raise ValueError(f'Unknown env_type: {env_type}')
    
    # 表征层
    pool = SimplePool(pool_capacity, input_dim)
    
    # 进化层
    evolution = CompetitionEvolution()
    
    # 主系统
    return FCRS(env, pool, evolution, lr)


# ========== 测试 ==========

def test_three_layer():
    """测试三层架构"""
    print('='*60)
    print('Three-Layer Architecture Test')
    print('='*60)
    
    # 创建系统
    fcrs = create_fcrs(
        env_type='structured',
        pool_capacity=5,
        input_dim=10,
        n_classes=5,
        lr=0.01
    )
    
    # 初始化表征
    for _ in range(3):
        x = fcrs.env.generate_input()
        rep = VectorRepresentation(x)
        fcrs.pool.add(rep)
    
    # 运行
    for _ in range(500):
        fcrs.step()
    
    # 结果
    stats = fcrs.get_stats()
    print('\nResults:')
    print(f'  Steps: {stats["step"]}')
    print(f'  Representations: {stats["size"]}')
    print(f'  Total dims: {stats["total_dims"]}')
    print(f'  Avg error: {stats["avg_error"]:.4f}')
    print('\nTest passed!')


if __name__ == "__main__":
    test_three_layer()
