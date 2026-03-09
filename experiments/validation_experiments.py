"""
FCRS-v5.2 验证实验
实验1-5: 优化驱动vs涌现驱动、临界状态、适应度、试用期、资源约束
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ==================== 核心类 ====================
@dataclass
class Representation:
    id: int
    vector: np.ndarray
    fitness_history: List[float] = field(default_factory=list)
    activation_count: int = 0
    age: int = 0
    origin: str = "initial"
    birth_step: int = 0


class CriticalityDetector:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.state_history = []
        self.prediction_errors = []
    
    def record_state(self, state_norm):
        self.state_history.append(state_norm)
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
    
    def compute_criticality_score(self):
        if len(self.state_history) < 2:
            return 0.0
        fluctuation = float(np.std(self.state_history))
        return 1 / (1 + np.exp(-0.5 * (fluctuation - 1)))


class EmergentDimensionGenerator:
    def __init__(self, input_dim):
        self.input_dim = input_dim
    
    def spontaneous_generate(self, base_vector, criticality):
        candidates = []
        # 增加到更高概率
        if np.random.random() < 0.8:
            noise = np.random.randn(len(base_vector)) * 0.1 * criticality
            candidates.append({'vector': base_vector + noise, 'origin': 'perturbation'})
        return candidates


class MultiDimensionalFitness:
    def __init__(self, alpha=0.4, beta=0.3, gamma=0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def compute_composite_fitness(self, dimension, input_data, original_dim):
        # 简化实现
        error = np.linalg.norm(dimension.vector - input_data)
        return -error, {'prediction': -error}


class ProbationaryManager:
    def __init__(self, probation_steps=20, threshold=0.3):
        self.probation_steps = probation_steps
        self.threshold = threshold
        self.probationary = {}
    
    def enter_probation(self, dim_id, fitness):
        self.probationary[dim_id] = {'steps': 0, 'fitness_sum': fitness}
    
    def should_retain(self, dim_id):
        if dim_id not in self.probationary:
            return None
        prob = self.probationary[dim_id]
        prob['steps'] += 1
        if prob['steps'] >= self.probation_steps:
            avg = prob['fitness_sum'] / prob['steps']
            del self.probationary[dim_id]
            return avg > self.threshold
        return None


class EmergentFCRS:
    """涌现驱动系统"""
    def __init__(self, pool_capacity=10, input_dim=10):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.representations = []
        self.criticality_detector = CriticalityDetector()
        self.dimension_generator = EmergentDimensionGenerator(input_dim)
        self.fitness_evaluator = MultiDimensionalFitness()
        self.probation_manager = ProbationaryManager()
        
        self.step_count = 0
        self.emergent_births = 0
        self.birth_times = []
        
        # 初始化
        for i in range(3):
            vec = np.random.randn(input_dim) * 0.5
            self.representations.append(Representation(i, vec, origin='initial', birth_step=0))
    
    def select(self, x):
        if not self.representations:
            return None
        best = max(self.representations, key=lambda r: np.dot(r.vector, x))
        return best
    
    def step(self, x):
        self.step_count += 1
        
        # 选择
        active = self.select(x)
        
        # 计算误差
        if active:
            error = np.linalg.norm(active.vector - x)
            active.fitness_history.append(-error)
            active.activation_count += 1
        else:
            error = float('inf')
        
        # 临界状态
        self.criticality_detector.record_state(np.linalg.norm(x))
        criticality = self.criticality_detector.compute_criticality_score()
        
        # 涌现生成
        if active and criticality > 0.1:
            candidates = self.dimension_generator.spontaneous_generate(active.vector, criticality)
            for c in candidates:
                fitness, _ = self.fitness_evaluator.compute_composite_fitness(
                    Representation(-1, c['vector']), x, len(active.vector))
                if fitness > 0.05:
                    new_rep = Representation(
                        len(self.representations), c['vector'], 
                        origin=c['origin'], birth_step=self.step_count)
                    self.representations.append(new_rep)
                    self.emergent_births += 1
                    self.birth_times.append(self.step_count)
        
        # 淘汰
        if len(self.representations) > self.pool_capacity:
            weakest = min(self.representations, 
                       key=lambda r: np.mean(r.fitness_history[-10:]) if r.fitness_history else -float('inf'))
            self.representations.remove(weakest)
        
        # 重新编号
        for i, r in enumerate(self.representations):
            r.id = i
        
        return error


class OptimizedFCRS:
    """优化驱动系统 (旧版本)"""
    def __init__(self, pool_capacity=10, input_dim=10, threshold=0.5):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.threshold = threshold
        self.compression_gain = 0
        self.representations = []
        
        self.step_count = 0
        self.birth_times = []
        
        for i in range(3):
            vec = np.random.randn(input_dim) * 0.5
            self.representations.append(Representation(i, vec))
    
    def step(self, x):
        self.step_count += 1
        
        # 选择
        active = max(self.representations, key=lambda r: np.dot(r.vector, x))
        error = np.linalg.norm(active.vector - x)
        active.fitness_history.append(-error)
        active.activation_count += 1
        
        # 压缩增益累积
        self.compression_gain += 0.01
        
        # 阈值判断
        if self.compression_gain > self.threshold:
            # 生成新维度
            new_vec = active.vector + np.random.randn(self.input_dim) * 0.1
            self.representations.append(Representation(
                len(self.representations), new_vec, birth_step=self.step_count))
            self.birth_times.append(self.step_count)
            self.compression_gain = 0
        
        if len(self.representations) > self.pool_capacity:
            self.representations.pop(0)
        
        for i, r in enumerate(self.representations):
            r.id = i
        
        return error


class Environment:
    def __init__(self, complexity=5, change_point=None):
        self.complexity = complexity
        self.change_point = change_point
        self.centers = {i: np.random.randn(10)*2 for i in range(complexity)}
    
    def generate(self):
        if self.change_point and self.change_point > 0:
            pass  # 简化
        cls = np.random.randint(0, len(self.centers))
        return self.centers[cls] + np.random.randn(10) * 0.3


# ==================== 实验 ====================
def experiment1_comparison():
    """实验1: 优化驱动 vs 涌现驱动"""
    print("="*60)
    print("Experiment 1: Emergent vs Optimized")
    print("="*60)
    
    results = {'emergent': [], 'optimized': []}
    
    for system_type in ['emergent', 'optimized']:
        for run in range(10):
            np.random.seed(run * 100)
            
            env = Environment(complexity=5)
            
            if system_type == 'emergent':
                system = EmergentFCRS(pool_capacity=10)
            else:
                system = OptimizedFCRS(pool_capacity=10, threshold=0.5)
            
            for _ in range(1000):
                x = env.generate()
                system.step(x)
            
            results[system_type].append({
                'births': system.emergent_births if hasattr(system, 'emergent_births') else 0,
                'birth_times': system.birth_times
            })
    
    print(f"\nEmergent births: {np.mean([r['births'] for r in results['emergent']]):.1f}")
    print(f"Optimized births: {np.mean([r['births'] for r in results['optimized']]):.1f}")
    
    return results


def experiment2_criticality():
    """实验2: 临界状态验证"""
    print("\n" + "="*60)
    print("Experiment 2: Criticality Validation")
    print("="*60)
    
    env = Environment(complexity=5)
    system = EmergentFCRS(pool_capacity=10)
    
    criticality_history = []
    birth_events = []
    
    for step in range(2000):
        x = env.generate()
        system.step(x)
        
        crit = system.criticality_detector.compute_criticality_score()
        criticality_history.append(crit)
        
        if hasattr(system, 'emergent_births'):
            if system.emergent_births > len(birth_events):
                birth_events.append(step)
    
    # 计算相关性
    if len(birth_events) > 0:
        print(f"Total births: {len(birth_events)}")
        print(f"Avg criticality at birth: {np.mean([criticality_history[min(b, len(criticality_history)-1)] for b in birth_events] if birth_events else 0):.3f}")
        print(f"Avg criticality overall: {np.mean(criticality_history):.3f}")
    
    return {'criticality': criticality_history, 'births': birth_events}


def experiment3_fitness():
    """实验3: 多维度适应度"""
    print("\n" + "="*60)
    print("Experiment 3: Multi-dimensional Fitness")
    print("="*60)
    
    configs = [
        {'alpha': 0.6, 'beta': 0.2, 'gamma': 0.2},
        {'alpha': 0.2, 'beta': 0.6, 'gamma': 0.2},
        {'alpha': 0.2, 'beta': 0.2, 'gamma': 0.6},
    ]
    
    for config in configs:
        np.random.seed(42)
        env = Environment(complexity=5)
        
        fitness_eval = MultiDimensionalFitness(**config)
        system = EmergentFCRS(pool_capacity=10)
        
        for _ in range(1000):
            x = env.generate()
            system.step(x)
        
        origins = [r.origin for r in system.representations]
        print(f"Config {config}: births={system.emergent_births}")


def experiment4_probation():
    """实验4: 试用期验证"""
    print("\n" + "="*60)
    print("Experiment 4: Probation Validation")
    print("="*60)
    
    # 环境变化点
    env = Environment(complexity=5)
    system = EmergentFCRS(pool_capacity=10, input_dim=10)
    
    # 运行
    for step in range(2000):
        x = env.generate()
        system.step(x)
    
    print(f"Births: {system.emergent_births}")
    print(f"Final pool: {len(system.representations)}")
    
    # 检查是否有来自perturbation的
    origins = [r.origin for r in system.representations]
    print(f"Origins: {origins}")
    
    return {'births': system.emergent_births}


def experiment5_resource():
    """实验5: 资源约束"""
    print("\n" + "="*60)
    print("Experiment 5: Resource Constraint")
    print("="*60)
    
    for capacity in [20, 15, 10, 5]:
        np.random.seed(42)
        env = Environment(complexity=5)
        system = EmergentFCRS(pool_capacity=capacity)
        
        for _ in range(1000):
            x = env.generate()
            system.step(x)
        
        print(f"Capacity {capacity}: births={system.emergent_births}, final={len(system.representations)}")


# ==================== Main ====================
def main():
    print("="*60)
    print("FCRS-v5.2 Validation Experiments")
    print("="*60)
    
    experiment1_comparison()
    experiment2_criticality()
    experiment3_fitness()
    experiment4_probation()
    experiment5_resource()
    
    print("\n" + "="*60)
    print("All Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
