"""
FCRS-v5 核心代码 - 带错误处理
按审查意见：增加输入验证、参数验证、边界检查
"""

import numpy as np


class ValidationError(Exception):
    """验证错误"""
    pass


class FCRSEnvironment:
    """环境 - 带验证"""
    
    def __init__(self, input_dim):
        # 输入验证
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValidationError(f'input_dim must be positive integer, got {input_dim}')
        
        self.input_dim = input_dim
        self.class_centers = {i: np.random.randn(input_dim) for i in range(3)}
    
    def generate_input(self):
        cls = np.random.randint(0, len(self.class_centers))
        center = self.class_centers[cls]
        return center + np.random.randn(self.input_dim) * 0.3
    
    def reset(self):
        """重置环境"""
        self.class_centers = {i: np.random.randn(self.input_dim) for i in range(3)}


class Representation:
    """表征 - 带验证"""
    
    def __init__(self, vector):
        # 向量验证
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        if len(vector.shape) != 1:
            raise ValidationError(f'vector must be 1D, got shape {vector.shape}')
        
        if not np.isfinite(vector).all():
            raise ValidationError('vector contains NaN or Inf')
        
        self.vector = vector
        self.age = 0
        self.activation_count = 0
        self.reuse = 0
        self.fitness_history = []
    
    def get_fitness(self):
        if self.fitness_history:
            return np.mean(self.fitness_history[-10:])
        return 0


class RepresentationPool:
    """表征池 - 带验证"""
    
    def __init__(self, capacity, input_dim):
        # 参数验证
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValidationError(f'capacity must be positive int, got {capacity}')
        
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValidationError(f'input_dim must be positive int, got {input_dim}')
        
        self.capacity = capacity
        self.input_dim = input_dim
        self.representations = []
        
        # 预算: 每个表征默认维度=input_dim
        self.total_budget = capacity * input_dim
    
    def add(self, representation):
        # 表征验证
        if not isinstance(representation, Representation):
            raise ValidationError('representation must be Representation instance')
        
        if len(representation.vector) != self.input_dim:
            raise ValidationError(
                f'vector dim {len(representation.vector)} != expected {self.input_dim}'
            )
        
        current_dims = self.get_total_dims()
        new_dims = len(representation.vector)
        
        # 边界检查
        if current_dims + new_dims > self.total_budget:
            return False
        
        self.representations.append(representation)
        return True
    
    def select(self, x):
        """选择最匹配的表征"""
        if not self.representations:
            return None
        
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        
        if len(x) != self.input_dim:
            return None
        
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            # 数值安全
            norm = np.linalg.norm(rep.vector)
            if norm < 1e-8:
                continue
            
            score = np.dot(rep.vector, x) / norm
            
            if not np.isfinite(score):
                continue
            
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
            'budget': self.total_budget,
            'avg_fitness': np.mean([r.get_fitness() for r in self.representations]) if self.representations else 0
        }


class EvolutionEngine:
    """进化引擎 - 带参数验证"""
    
    def __init__(self, spawn_threshold=2, min_compression_gain=0.001):
        # 参数验证
        if not isinstance(spawn_threshold, (int, float)) or spawn_threshold < 0:
            raise ValidationError(f'spawn_threshold must be non-negative, got {spawn_threshold}')
        
        if not isinstance(min_compression_gain, (int, float)) or min_compression_gain < 0:
            raise ValidationError(f'min_compression_gain must be non-negative, got {min_compression_gain}')
        
        self.spawn_reuse_threshold = spawn_threshold
        self.min_compression_gain = min_compression_gain
        self.new_dim_history = []
    
    def try_spawn_new_dim(self, representation, residuals):
        """尝试生成新维度"""
        # 边界检查
        if representation.reuse < self.spawn_reuse_threshold:
            return False
        
        # 计算压缩增益
        if len(residuals) < 10:
            return False
        
        # 数值安全
        residuals = np.array(residuals)
        if not np.isfinite(residuals).all():
            return False
        
        recent = residuals[-10:]
        compression_gain = np.std(recent) / (np.mean(np.abs(recent)) + 1e-8)
        
        if compression_gain < self.min_compression_gain:
            return False
        
        # 生成新表征
        new_dim = len(representation.vector) + 1
        
        print(f'新维度诞生! 压缩增益={compression_gain:.3f}, 新维={new_dim}')
        
        self.new_dim_history.append({
            'dim': new_dim,
            'gain': compression_gain
        })
        
        return True


class FCRSystem:
    """FCRS主系统 - 带完整错误处理"""
    
    def __init__(self, pool_capacity=5, vector_dim=10, 
                 spawn_threshold=2, min_compression_gain=0.001):
        # 参数验证
        if not isinstance(pool_capacity, int) or pool_capacity <= 0:
            raise ValidationError(f'pool_capacity must be positive int, got {pool_capacity}')
        
        if not isinstance(vector_dim, int) or vector_dim <= 0:
            raise ValidationError(f'vector_dim must be positive int, got {vector_dim}')
        
        # 初始化各层
        self.env = FCRSEnvironment(vector_dim)
        self.pool = RepresentationPool(pool_capacity, vector_dim)
        self.engine = EvolutionEngine(spawn_threshold, min_compression_gain)
        
        # 初始化表征
        for _ in range(3):
            x = self.env.generate_input()
            rep = Representation(x)
            self.pool.add(rep)
        
        self.step_count = 0
        self.dim_history = []
        self.recent_residuals = []
    
    def step(self):
        """执行一步 - 带错误处理"""
        try:
            self.step_count += 1
            
            # 生成输入
            x = self.env.generate_input()
            
            # 选择表征
            active = self.pool.select(x)
            
            if active is not None:
                active.activation_count += 1
                active.reuse += 1
                
                # 预测
                pred = active.vector
                error = np.linalg.norm(x - pred)
                
                # 记录
                active.fitness_history.append(-error)
                
                # 收集残差
                self.recent_residuals.append(error)
                if len(self.recent_residuals) > 100:
                    self.recent_residuals.pop(0)
                
                # 尝试新维度
                if self.step_count % 10 == 0:
                    self.engine.try_spawn_new_dim(active, self.recent_residuals)
            
            # 更新年龄
            for rep in self.pool.representations:
                rep.age += 1
            
            # 记录历史
            if self.step_count % 50 == 0:
                self.dim_history.append(self.pool.get_total_dims())
        
        except Exception as e:
            print(f'Error in step {self.step_count}: {e}')
            raise
    
    def get_statistics(self):
        pool_stats = self.pool.get_stats()
        
        return {
            'step': self.step_count,
            **pool_stats,
            'new_dims_born': len(self.engine.new_dim_history),
            'dim_history': self.dim_history
        }


# ========== 测试错误处理 ==========

def test_error_handling():
    """测试错误处理"""
    print('='*60)
    print('Error Handling Tests')
    print('='*60)
    
    # 测试1: 无效参数
    print('\n1. Invalid parameters:')
    try:
        FCRSystem(pool_capacity=-1, vector_dim=10)
    except ValidationError as e:
        print(f'  Caught: {e}')
    
    # 测试2: 无效向量
    print('\n2. Invalid vector:')
    try:
        r = Representation(np.array([[1, 2], [3, 4]]))  # 2D array
    except ValidationError as e:
        print(f'  Caught: {e}')
    
    # 测试3: NaN向量
    print('\n3. NaN vector:')
    try:
        r = Representation(np.array([1, np.nan, 3]))
    except ValidationError as e:
        print(f'  Caught: {e}')
    
    # 测试4: 正常系统
    print('\n4. Normal system:')
    system = FCRSystem(pool_capacity=5, vector_dim=10)
    for _ in range(10):
        system.step()
    stats = system.get_statistics()
    print(f'  OK: {stats["size"]} representations, {stats["total_dims"]} dims')
    
    print('\nAll tests passed!')


if __name__ == "__main__":
    test_error_handling()
