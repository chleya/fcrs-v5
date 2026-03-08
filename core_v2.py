"""
FCRS-v5 三层架构
环境层 / 表征层 / 进化层

按审查意见重构：清晰模块化设计，明确接口
"""

import numpy as np


# ========== 第一层: 环境层 ==========

class Environment:
    """环境接口"""
    
    def generate_input(self):
        """生成单个输入"""
        raise NotImplementedError
    
    def reset(self):
        """重置环境"""
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
    
    def __init__(self, input_dim, complexity):
        self.input_dim = input_dim
        self.complexity = complexity
        self.class_centers = {
            i: np.random.randn(input_dim) for i in range(complexity)
        }
        
    def generate_input(self):
        cls = np.random.randint(0, self.complexity)
        center = self.class_centers[cls]
        return center + np.random.randn(self.input_dim) * 0.3
    
    def reset(self):
        self.class_centers = {
            i: np.random.randn(self.input_dim) for i in range(self.complexity)
        }


class NoisyEnvironment(Environment):
    """噪声环境"""
    
    def __init__(self, input_dim, noise_level=0.5):
        self.input_dim = input_dim
        self.noise_level = noise_level
        
    def generate_input(self):
        x = np.random.randn(self.input_dim)
        return x + np.random.randn(self.input_dim) * self.noise_level
    
    def reset(self):
        pass


# ========== 第二层: 表征层 ==========

class Representation:
    """表征接口"""
    
    def get_vector(self):
        return self.vector
    
    def update(self, *args, **kwargs):
        """更新表征"""
        raise NotImplementedError
        
    def get_fitness(self):
        if self.fitness_history:
            return np.mean(self.fitness_history[-10:])
        return 0


class VectorRepresentation(Representation):
    """向量表征"""
    
    def __init__(self, vector):
        self.vector = vector
        self.age = 0
        self.activation_count = 0
        self.fitness_history = []
        
    def update(self, new_vector=None):
        if new_vector is not None:
            self.vector = new_vector


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
        self.total_budget = capacity * input_dim
        
    def add(self, representation):
        if self.get_total_dims() + len(representation.vector) <= self.total_budget:
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
            score = np.dot(rep.vector, x) / (np.linalg.norm(rep.vector) + 1e-8)
            if score > best_score:
                best_score = score
                best = rep
                
        return best
    
    def get_total_dims(self):
        return sum(len(r.vector) for r in self.representations)
    
    def get_stats(self):
        return {
            'size': len(self.representations),
            'total_dims': self.get_total_dims(),
            'avg_fitness': np.mean([r.get_fitness() for r in self.representations]) if self.representations else 0
        }


# ========== 第三层: 进化层 ==========

class EvolutionEngine:
    """进化引擎接口"""
    
    def evolve(self, pool, x, prediction_error):
        """执行一轮进化"""
        raise NotImplementedError
        
    def get_statistics(self):
        raise NotImplementedError


class CompetitionEvolution(EvolutionEngine):
    """竞争进化"""
    
    def __init__(self):
        self.mutations = 0
        self.survivals = 0
        
    def evolve(self, pool, x, prediction_error):
        """竞争进化逻辑"""
        # 1. 记录适应度
        active = pool.select(x)
        if active:
            active.fitness_history.append(-prediction_error)
            active.activation_count += 1
            
        # 2. 更新年龄
        for rep in pool.representations:
            rep.age += 1
            
        return active
    
    def get_statistics(self):
        return {
            'mutations': self.mutations,
            'survivals': self.survivals
        }


# ========== 主系统 ==========

class FCRSystem:
    """FCRS主系统 - 三层架构"""
    
    def __init__(self, env, pool, evolution):
        self.env = env
        self.pool = pool
        self.evolution = evolution
        self.step_count = 0
        
    def step(self):
        """执行一步"""
        self.step_count += 1
        
        # 1. 生成输入
        x = self.env.generate_input()
        
        # 2. 选择表征并预测
        active = self.pool.select(x)
        
        if active is not None:
            prediction = active.vector
            error = np.linalg.norm(x - prediction)
        else:
            error = float('inf')
            
        # 3. 进化
        self.evolution.evolve(self.pool, x, error)
        
    def get_statistics(self):
        pool_stats = self.pool.get_stats()
        evo_stats = self.evolution.get_statistics()
        
        return {
            'step': self.step_count,
            **pool_stats,
            **evo_stats
        }


# ========== 工厂函数 ==========

def create_system(env_type='random', pool_capacity=5, input_dim=10, complexity=5, noise=0.5):
    """工厂函数：创建系统"""
    
    # 环境层
    if env_type == 'random':
        env = RandomEnvironment(input_dim)
    elif env_type == 'structured':
        env = StructuredEnvironment(input_dim, complexity)
    elif env_type == 'noisy':
        env = NoisyEnvironment(input_dim, noise)
    else:
        raise ValueError('Unknown env_type')
    
    # 表征层
    pool = SimplePool(pool_capacity, input_dim)
    
    # 进化层
    evolution = CompetitionEvolution()
    
    # 主系统
    return FCRSystem(env, pool, evolution)


# ========== 测试 ==========

def test_three_layer():
    """测试三层架构"""
    print('='*60)
    print('三层架构测试')
    print('='*60)
    
    # 使用工厂函数创建
    system = create_system(
        env_type='structured',
        pool_capacity=5,
        input_dim=10,
        complexity=5
    )
    
    # 初始化表征
    for _ in range(3):
        x = system.env.generate_input()
        rep = VectorRepresentation(x)
        system.pool.add(rep)
    
    # 运行
    for i in range(100):
        system.step()
        
        if (i + 1) % 20 == 0:
            stats = system.get_statistics()
            print('Step ' + str(i+1) + ': size=' + str(stats['size']) + 
                  ', dims=' + str(stats['total_dims']))
    
    print('')
    print('测试完成!')


if __name__ == "__main__":
    test_three_layer()
