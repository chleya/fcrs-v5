"""
FCRS-v5 涌现驱动版本
核心改进：从"优化驱动"改为"涌现驱动"

优化驱动（旧）：
  计算压缩增益 → 判断阈值 → 决定是否增加维度

涌现驱动（新）：
  预测失败 → 自发生成 → 竞争筛选 → 保留/淘汰
"""

import numpy as np


class EmergentRepresentation:
    """涌现表征"""
    
    def __init__(self, vector, parent=None):
        self.vector = vector
        self.parent = parent  # 父表征（如果有）
        self.age = 0
        self.activation_count = 0
        self.fitness_history = []
        self.survival_trials = 0  # 生存考验次数
        self.survival_success = 0  # 生存成功次数
        
    def get_fitness(self):
        if self.fitness_history:
            return np.mean(self.fitness_history[-10:])
        return 0


class EmergentPool:
    """涌现表征池"""
    
    def __init__(self, capacity, input_dim):
        self.capacity = capacity
        self.input_dim = input_dim
        self.representations = []
        self.total_budget = capacity * input_dim
        
    def add(self, representation):
        """添加表征"""
        if self.get_total_dims() + len(representation.vector) <= self.total_budget:
            self.representations.append(representation)
            return True
        return False
    
    def get_total_dims(self):
        return sum(len(r.vector) for r in self.representations)
    
    def select_for_activation(self, x):
        """选择最匹配的表征"""
        if not self.representations:
            return None
            
        best_rep = None
        best_match = -float('inf')
        
        for rep in self.representations:
            match = np.dot(rep.vector, x) / (np.linalg.norm(rep.vector) + 1e-8)
            if match > best_match:
                best_match = match
                best_rep = rep
        
        return best_rep
    
    def compete(self, x, prediction_error):
        """竞争筛选：预测失败触发新表征生成"""
        # 如果预测误差超过阈值，触发涌现
        error_threshold = 0.5
        
        if prediction_error > error_threshold:
            # 涌现新表征
            new_vector = self._emergent_generate(x, prediction_error)
            
            if new_vector is not None:
                new_rep = EmergentRepresentation(new_vector)
                new_rep.survival_trials = 1
                
                # 竞争筛选：只有通过考验才能保留
                if self._survival_test(new_rep, x):
                    new_rep.survival_success = 1
                    self.add(new_rep)
                    
                    # 如果超过容量，淘汰最弱的
                    if len(self.representations) > self.capacity:
                        self._淘汰最弱()
                    
                    return True, new_rep
        
        return False, None
    
    def _emergent_generate(self, x, error):
        """基于预测失败自发生成新表征"""
        # 思路：预测失败说明当前表征无法处理这个输入
        # 所以新表征应该"填补"这个空白
        
        # 方法：对输入进行扰动，生成新的表征
        noise = np.random.randn(self.input_dim) * 0.1
        new_vector = x + noise
        
        # 裁剪到合理范围
        new_vector = np.clip(new_vector, -2, 2)
        
        return new_vector
    
    def _survival_test(self, new_rep, x):
        """生存测试：新表征必须证明自己有价值"""
        # 简单测试：新表征能否处理当前输入
        match = np.dot(new_rep.vector, x) / (np.linalg.norm(new_rep.vector) + 1e-8)
        
        # 如果匹配度高于随机，说明有潜在价值
        return match > 0.1
    
    def _淘汰最弱(self):
        """淘汰最弱的表征"""
        if not self.representations:
            return
            
        # 按适应度排序，淘汰最弱的
        self.representations.sort(key=lambda r: r.get_fitness())
        self.representations.pop(0)


class EmergentEnvironment:
    """涌现环境"""
    
    def __init__(self, input_dim, complexity=5):
        self.input_dim = input_dim
        self.complexity = complexity
        self.class_centers = {
            i: np.random.randn(input_dim) for i in range(complexity)
        }
    
    def generate_input(self):
        """生成输入"""
        cls = np.random.randint(0, self.complexity)
        center = self.class_centers[cls]
        noise = np.random.randn(self.input_dim) * 0.3
        return center + noise


class EmergentSystem:
    """涌现驱动系统"""
    
    def __init__(self, pool_capacity=5, input_dim=10, complexity=5):
        self.pool = EmergentPool(pool_capacity, input_dim)
        self.env = EmergentEnvironment(input_dim, complexity)
        
        # 初始化：创建初始表征
        for _ in range(3):
            x = self.env.generate_input()
            rep = EmergentRepresentation(x)
            self.pool.add(rep)
        
        self.step_count = 0
        self.emergent_births = 0
        self.dim_history = []
        
    def step(self):
        """单步"""
        self.step_count += 1
        
        # 1. 生成输入
        x = self.env.generate_input()
        
        # 2. 选择表征
        active_rep = self.pool.select_for_activation(x)
        
        # 3. 计算预测
        if active_rep is not None:
            active_rep.activation_count += 1
            
            # 预测：使用表征向量重建输入
            prediction = active_rep.vector
            error = np.linalg.norm(x - prediction)
            
            # 记录适应度
            active_rep.fitness_history.append(-error)
        else:
            error = float('inf')
        
        # 4. 涌现检测（核心区别于优化驱动）
        emerged, new_rep = self.pool.compete(x, error)
        
        if emerged:
            self.emergent_births += 1
            print('涌现! 新维度=' + str(len(new_rep.vector)) + ', 误差=' + str(round(error, 3)))
        
        # 5. 更新年龄
        for rep in self.pool.representations:
            rep.age += 1
        
        # 6. 记录历史
        if self.step_count % 50 == 0:
            self.dim_history.append(self.pool.get_total_dims())
    
    def get_statistics(self):
        return {
            'step': self.step_count,
            'pool_size': len(self.pool.representations),
            'total_dims': self.pool.get_total_dims(),
            'emergent_births': self.emergent_births,
            'budget': self.pool.total_budget - self.pool.get_total_dims()
        }


def test_emergent():
    """测试涌现驱动系统"""
    print('='*60)
    print('涌现驱动系统测试')
    print('='*60)
    
    # 创建系统
    system = EmergentSystem(pool_capacity=5, input_dim=10, complexity=5)
    
    # 运行
    for i in range(500):
        system.step()
        
        if (i + 1) % 100 == 0:
            stats = system.get_statistics()
            print('Step ' + str(i+1) + ': dims=' + str(stats['total_dims']) + 
                  ', births=' + str(stats['emergent_births']))
    
    # 最终统计
    stats = system.get_statistics()
    print('')
    print('='*60)
    print('最终结果')
    print('='*60)
    print('总维度: ' + str(stats['total_dims']))
    print('涌现次数: ' + str(stats['emergent_births']))
    print('表征数: ' + str(stats['pool_size']))


if __name__ == "__main__":
    test_emergent()
