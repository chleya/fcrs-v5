"""
FCRS涌现驱动版本
核心：从"优化驱动"改为"涌现驱动"

优化驱动（旧）：
  计算压缩增益 → 判断阈值 → 决定是否增加维度

涌现驱动（新）：
  预测失败 → 自发生成 → 竞争筛选 → 保留/淘汰
"""

import numpy as np


class EmergentRepresentation:
    """涌现表征"""
    
    def __init__(self, vector, origin='spontaneous'):
        self.vector = vector
        self.origin = origin  # 'spontaneous' or 'learned'
        self.age = 0
        self.activation_count = 0
        self.fitness_history = []
        self.survival_trials = 0
        self.survival_success = 0
        
    def get_fitness(self):
        if self.fitness_history:
            return np.mean(self.fitness_history[-10:])
        return 0
    
    def update(self, x, lr):
        """在线学习更新"""
        error = x - self.vector
        self.vector += lr * error
        return np.linalg.norm(error)


class EmergentPool:
    """涌现表征池"""
    
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
        """选择最匹配的表征"""
        if not self.representations:
            return None
        
        best = None
        best_score = -float('inf')
        
        for rep in self.representations:
            score = np.dot(rep.get_fitness(), rep.vector) if rep.get_fitness() > 0 else np.dot(rep.vector, x)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def emergent_generate(self, x, failure_signal):
        """
        涌现生成：当预测失败时自发生成新表征
        关键：不是基于阈值判断，而是基于系统内在动力
        """
        # 失败信号 = 预测误差超过历史平均
        if failure_signal < 0.5:  # 失败信号弱
            return None
        
        # 涌现生成策略
        strategies = []
        
        # 1. 残差表征：预测误差作为新表征
        residual = x - self.get_best_prediction(x)
        if np.linalg.norm(residual) > 0.1:
            strategies.append(residual)
        
        # 2. 扰动表征：对输入加噪声
        noise = np.random.randn(self.input_dim) * 0.5
        strategies.append(x + noise)
        
        # 3. 正交表征：生成与现有表征正交的向量
        if self.representations:
            existing = np.mean([r.vector for r in self.representations], axis=0)
            orthogonal = x - np.dot(x, existing) * existing / (np.linalg.norm(existing)**2 + 1e-8)
            strategies.append(orthogonal)
        
        # 随机选择一种策略
        if strategies:
            new_vector = strategies[np.random.randint(len(strategies))]
            new_rep = EmergentRepresentation(new_vector, origin='spontaneous')
            new_rep.survival_trials = 1
            
            return new_rep
        
        return None
    
    def get_best_prediction(self, x):
        """获取最佳预测"""
        if not self.representations:
            return np.zeros(self.input_dim)
        
        best_rep = max(self.representations, key=lambda r: r.get_fitness())
        return best_rep.vector
    
    def compete(self, x, error):
        """竞争筛选"""
        # 失败信号
        failure_signal = 0
        if self.representations:
            avg_fitness = np.mean([r.get_fitness() for r in self.representations])
            failure_signal = 1 if error > -avg_fitness else 0
        
        # 尝试涌现生成
        new_rep = self.emergent_generate(x, failure_signal)
        
        if new_rep:
            # 生存测试
            test_error = np.linalg.norm(x - new_rep.vector)
            
            if test_error < error * 1.5:  # 宽松条件
                # 尝试加入池
                if len(self.representations) >= self.capacity:
                    # 淘汰最弱的
                    self.representations.sort(key=lambda r: r.get_fitness())
                    self.representations.pop(0)
                
                self.add(new_rep)
                return True, new_rep
        
        return False, None
    
    def get_total_dims(self):
        return sum(len(r.vector) for r in self.representations)


class EmergentEnvironment:
    """涌现环境"""
    
    def __init__(self, input_dim, complexity):
        self.input_dim = input_dim
        self.complexity = complexity
        self.class_centers = {i: np.random.randn(input_dim) for i in range(complexity)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.complexity)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


class EmergentFCRS:
    """涌现驱动FCRS"""
    
    def __init__(self, pool_capacity=5, input_dim=10, complexity=5, lr=0.01):
        self.pool = EmergentPool(pool_capacity, input_dim)
        self.env = EmergentEnvironment(input_dim, complexity)
        self.lr = lr
        self.step_count = 0
        self.errors = []
        self.emergent_births = 0
        
        # 初始化表征
        for _ in range(3):
            x = self.env.generate_input()
            rep = EmergentRepresentation(x, origin='initial')
            self.pool.add(rep)
    
    def step(self):
        """一步"""
        self.step_count += 1
        
        # 1. 生成输入
        x = self.env.generate_input()
        
        # 2. 选择表征
        active = self.pool.select(x)
        
        if active is not None:
            # 3. 预测+学习
            error = active.update(x, self.lr)
            
            # 4. 记录适应度
            active.fitness_history.append(-error)
            active.activation_count += 1
            
            self.errors.append(error)
            
            # 5. 涌现检测（核心区别）
            emerged, new_rep = self.pool.compete(x, error)
            
            if emerged:
                self.emergent_births += 1
                print(f'Emerge! Total: {self.emergent_births}, Dims: {self.pool.get_total_dims()}')
        else:
            self.errors.append(float('inf'))
        
        # 6. 更新年龄
        for rep in self.pool.representations:
            rep.age += 1
    
    def get_avg_error(self):
        if not self.errors:
            return 0
        return np.mean(self.errors[-100:])
    
    def get_stats(self):
        return {
            'steps': self.step_count,
            'reps': len(self.pool.representations),
            'dims': self.pool.get_total_dims(),
            'emergent_births': self.emergent_births,
            'avg_error': self.get_avg_error()
        }


def test_emergent():
    """测试涌现驱动"""
    print('='*60)
    print('Emergence-Driven FCRS Test')
    print('='*60)
    
    # 创建系统
    fcrs = EmergentFCRS(pool_capacity=5, input_dim=10, complexity=3, lr=0.01)
    
    # 运行
    for i in range(500):
        fcrs.step()
        
        if (i + 1) % 100 == 0:
            stats = fcrs.get_stats()
            print(f'Step {i+1}: error={stats["avg_error"]:.4f}, dims={stats["dims"]}, births={stats["emergent_births"]}')
    
    # 结果
    stats = fcrs.get_stats()
    print('\n' + '='*60)
    print('Results')
    print('='*60)
    print(f'Steps: {stats["steps"]}')
    print(f'Representations: {stats["reps"]}')
    print(f'Total Dims: {stats["dims"]}')
    print(f'Emergent Births: {stats["emergent_births"]}')
    print(f'Avg Error: {stats["avg_error"]:.4f}')


if __name__ == "__main__":
    test_emergent()
