"""
FCRS-v5.2 严格验证实验
按照审查文档的5个实验设计实施
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json


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
    """临界状态检测器"""
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.state_history: List[float] = []
        self.prediction_errors: List[float] = []
        self.dimension_counts: List[int] = []
    
    def record_state(self, state_norm: float):
        self.state_history.append(state_norm)
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)
    
    def record_prediction_error(self, error: float):
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.window_size:
            self.prediction_errors.pop(0)
    
    def record_dimension_count(self, count: int):
        self.dimension_counts.append(count)
        if len(self.dimension_counts) > self.window_size:
            self.dimension_counts.pop(0)
    
    def compute_fluctuation(self) -> float:
        if len(self.state_history) < 2:
            return 0.0
        return float(np.std(np.array(self.state_history)))
    
    def compute_prediction_uncertainty(self) -> float:
        if len(self.prediction_errors) < 2:
            return 1.0
        return float(np.var(np.array(self.prediction_errors)))
    
    def compute_criticality_score(self) -> float:
        fluctuation = self.compute_fluctuation()
        uncertainty = self.compute_prediction_uncertainty()
        combined = fluctuation * (1 + uncertainty)
        score = 1 / (1 + np.exp(-0.5 * (combined - 1)))
        return score
    
    def is_critical(self, threshold: float = 0.5) -> bool:
        return self.compute_criticality_score() > threshold


class EmergentDimensionGenerator:
    """涌现维度生成器"""
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
    
    def spontaneous_generate(self, base_vector: np.ndarray, criticality: float) -> List[Dict]:
        candidates = []
        
        # 扰动概率与临界度成正比
        if np.random.random() < criticality * 0.3:
            # 扰动方式1: 噪声
            noise = np.random.randn(len(base_vector)) * 0.1 * criticality
            candidates.append({
                'vector': base_vector + noise,
                'origin': 'perturbation',
                'strength': criticality
            })
        
        return candidates


class MultiDimensionalFitness:
    """多维度适应度评估器"""
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def evaluate_prediction_fitness(self, dimension: Representation, input_data: np.ndarray) -> float:
        if len(dimension.vector) == 0:
            return 0.0
        min_len = min(len(dimension.vector), len(input_data))
        error = np.linalg.norm(dimension.vector[:min_len] - input_data[:min_len])
        return -error
    
    def evaluate_compression_fitness(self, original_dim: int, new_dim: int) -> float:
        if original_dim == 0:
            return 1.0 if new_dim > 0 else 0.0
        return max(0, 1 - new_dim / original_dim)
    
    def evaluate_behavior_fitness(self, dimension: Representation) -> float:
        if dimension.age > 0:
            return dimension.activation_count / dimension.age
        return 0.0
    
    def compute_composite_fitness(self, dimension: Representation, input_data: np.ndarray, 
                                 original_dim: int) -> tuple:
        F_pred = self.evaluate_prediction_fitness(dimension, input_data)
        F_comp = self.evaluate_compression_fitness(original_dim, len(dimension.vector))
        F_beh = self.evaluate_behavior_fitness(dimension)
        
        F_pred_norm = 1 / (1 + np.exp(-F_pred)) if F_pred < 10 else 1.0
        
        weights = np.array([self.alpha, self.beta, self.gamma])
        fitness_vector = np.array([F_pred_norm, F_comp, F_beh])
        composite = np.dot(fitness_vector, weights)
        
        details = {
            'prediction': F_pred,
            'compression': F_comp,
            'behavior': F_beh
        }
        
        return composite, details


class ProbationaryManager:
    """试用期管理器"""
    def __init__(self, probation_steps: int = 20, retention_threshold: float = 0.3):
        self.probation_steps = probation_steps
        self.retention_threshold = retention_threshold
        self.probationary_dimensions: Dict[int, Dict] = {}
    
    def enter_probation(self, dimension_id: int, initial_performance: float):
        self.probationary_dimensions[dimension_id] = {
            'steps_in_probation': 0,
            'performance_sum': initial_performance,
            'survival_trials': 0,
            'survival_successes': 0
        }
    
    def update_probation(self, dimension_id: int, performance: float) -> bool:
        if dimension_id not in self.probationary_dimensions:
            return False
        
        prob = self.probationary_dimensions[dimension_id]
        prob['steps_in_probation'] += 1
        prob['performance_sum'] += performance
        
        if performance > self.retention_threshold:
            prob['survival_successes'] += 1
        prob['survival_trials'] += 1
        
        return True
    
    def should_retain(self, dimension_id: int) -> Optional[bool]:
        if dimension_id not in self.probationary_dimensions:
            return None
        
        prob = self.probationary_dimensions[dimension_id]
        
        if prob['steps_in_probation'] >= self.probation_steps:
            avg_performance = prob['performance_sum'] / prob['steps_in_probation']
            avg_normalized = 1 / (1 + np.exp(-avg_performance)) if avg_performance < 10 else 1.0
            survival_rate = prob['survival_successes'] / max(1, prob['survival_trials'])
            retention_score = 0.6 * avg_normalized + 0.4 * survival_rate
            
            del self.probationary_dimensions[dimension_id]
            return retention_score > self.retention_threshold
        
        return None


class EmergentFCRS:
    """涌现驱动FCRS"""
    def __init__(self, pool_capacity: int = 10, input_dim: int = 10,
                 alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.representations: List[Representation] = []
        
        self.criticality_detector = CriticalityDetector()
        self.dimension_generator = EmergentDimensionGenerator(input_dim)
        self.fitness_evaluator = MultiDimensionalFitness(alpha, beta, gamma)
        self.probation_manager = ProbationaryManager()
        
        self.step_count = 0
        self.emergent_births = 0
        self.survival_rejections = 0
        self.birth_times: List[int] = []
        self.criticality_history: List[float] = []
        
        # 初始化
        for i in range(3):
            vector = np.random.randn(input_dim) * 0.5
            rep = Representation(i, vector, origin='initial', birth_step=0)
            self.representations.append(rep)
    
    def select_representation(self, x: np.ndarray) -> Optional[Representation]:
        if not self.representations:
            return None
        
        best = None
        best_score = float('-inf')
        
        for rep in self.representations:
            min_len = min(len(rep.vector), len(x))
            if min_len == 0:
                continue
            score = np.dot(rep.vector[:min_len], x[:min_len])
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def compute_prediction(self, rep: Representation, x: np.ndarray) -> float:
        min_len = min(len(rep.vector), len(x))
        if min_len == 0:
            return float('inf')
        prediction = rep.vector[:min_len]
        target = x[:min_len]
        return float(np.linalg.norm(prediction - target))
    
    def step(self, x: np.ndarray) -> float:
        self.step_count += 1
        
        # 选择
        active_rep = self.select_representation(x)
        
        # 计算误差
        if active_rep is not None:
            error = self.compute_prediction(active_rep, x)
            active_rep.fitness_history.append(-error)
            active_rep.activation_count += 1
        else:
            error = float('inf')
        
        # 更新临界状态
        self.criticality_detector.record_state(np.linalg.norm(x))
        self.criticality_detector.record_prediction_error(error)
        self.criticality_detector.record_dimension_count(len(self.representations))
        
        criticality = self.criticality_detector.compute_criticality_score()
        
        # 涌现生成
        if criticality > 0.1 and active_rep is not None:
            candidates = self.dimension_generator.spontaneous_generate(
                active_rep.vector, criticality)
            
            original_dim = len(active_rep.vector)
            
            for candidate in candidates:
                temp_rep = Representation(-1, candidate['vector'])
                fitness, _ = self.fitness_evaluator.compute_composite_fitness(
                    temp_rep, x, original_dim)
                
                if fitness > 0.2:
                    new_rep = Representation(
                        len(self.representations),
                        candidate['vector'],
                        origin=candidate['origin'],
                        birth_step=self.step_count
                    )
                    self.representations.append(new_rep)
                    self.probation_manager.enter_probation(new_rep.id, fitness)
                    self.emergent_births += 1
                    self.birth_times.append(self.step_count)
        
        # 更新试用期
        for rep in self.representations[:]:
            should_retain = self.probation_manager.should_retain(rep.id)
            if should_retain is False:
                self.representations.remove(rep)
                self.survival_rejections += 1
            elif should_retain is True:
                pass
        
        # 容量限制
        if len(self.representations) > self.pool_capacity:
            # 淘汰最弱的
            self.representations.sort(
                key=lambda r: np.mean(r.fitness_history[-10:]) if r.fitness_history else -float('inf')
            )
            self.representations.pop(0)
        
        # 重新编号
        for i, rep in enumerate(self.representations):
            rep.id = i
        
        # 记录
        if self.step_count % 50 == 0:
            self.criticality_history.append(criticality)
        
        return error


class OptimizedFCRS:
    """优化驱动FCRS (基线)"""
    def __init__(self, pool_capacity: int = 10, input_dim: int = 10, threshold: float = 0.5):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim
        self.threshold = threshold
        self.compression_gain = 0.0
        self.representations: List[Representation] = []
        
        self.step_count = 0
        self.birth_times: List[int] = []
        
        for i in range(3):
            vector = np.random.randn(input_dim) * 0.5
            self.representations.append(Representation(i, vector, birth_step=0))
    
    def step(self, x: np.ndarray) -> float:
        self.step_count += 1
        
        # 选择
        active_rep = max(self.representations, 
                       key=lambda r: np.dot(r.vector, x))
        
        # 误差
        error = self.compute_prediction(active_rep, x)
        active_rep.fitness_history.append(-error)
        active_rep.activation_count += 1
        
        # 压缩增益累积
        self.compression_gain += 0.01
        
        # 阈值判断
        if self.compression_gain > self.threshold:
            new_vector = active_rep.vector + np.random.randn(self.input_dim) * 0.1
            new_rep = Representation(
                len(self.representations),
                new_vector,
                origin='threshold',
                birth_step=self.step_count
            )
            self.representations.append(new_rep)
            self.birth_times.append(self.step_count)
            self.compression_gain = 0
        
        # 容量限制
        if len(self.representations) > self.pool_capacity:
            self.representations.pop(0)
        
        for i, rep in enumerate(self.representations):
            rep.id = i
        
        return error
    
    def compute_prediction(self, rep, x):
        min_len = min(len(rep.vector), len(x))
        return float(np.linalg.norm(rep.vector[:min_len] - x[:min_len]))


class Environment:
    """环境"""
    def __init__(self, complexity: int = 5, input_dim: int = 10):
        self.complexity = complexity
        self.input_dim = input_dim
        self.class_centers = {
            i: np.random.randn(input_dim) * 2 
            for i in range(complexity)
        }
    
    def generate(self):
        cls = np.random.randint(0, self.complexity)
        return self.class_centers[cls] + np.random.randn(self.input_dim) * 0.3


# ==================== 实验1: 优化驱动vs涌现驱动 ====================
def experiment1_comparison(n_runs: int = 30, steps: int = 5000):
    """
    实验1: 优化驱动 vs 涌现驱动对比
    
    目标: 验证两种机制的维度生成呈现不同特征
    预期: 涌现驱动呈现脉冲式,优化驱动呈现阶梯式
    """
    print("="*60)
    print("Experiment 1: Emergent vs Optimized Comparison")
    print("="*60)
    
    results = {
        'emergent': {'births': [], 'birth_intervals': [], 'cv': []},
        'optimized': {'births': [], 'birth_intervals': [], 'cv': []}
    }
    
    for run in range(n_runs):
        np.random.seed(1000 + run)
        
        # 涌现驱动
        env = Environment(complexity=5, input_dim=10)
        emergent = EmergentFCRS(pool_capacity=10, input_dim=10)
        
        for _ in range(steps):
            x = env.generate()
            emergent.step(x)
        
        results['emergent']['births'].append(emergent.emergent_births)
        if len(emergent.birth_times) > 1:
            intervals = np.diff(emergent.birth_times)
            results['emergent']['birth_intervals'].append(np.std(intervals) / (np.mean(intervals) + 1e-8))
        else:
            results['emergent']['birth_intervals'].append(0)
        
        # 优化驱动
        np.random.seed(1000 + run)
        env = Environment(complexity=5, input_dim=10)
        optimized = OptimizedFCRS(pool_capacity=10, input_dim=10, threshold=0.5)
        
        for _ in range(steps):
            x = env.generate()
            optimized.step(x)
        
        results['optimized']['births'].append(len(optimized.birth_times))
        if len(optimized.birth_times) > 1:
            intervals = np.diff(optimized.birth_times)
            results['optimized']['birth_intervals'].append(np.std(intervals) / (np.mean(intervals) + 1e-8))
        else:
            results['optimized']['birth_intervals'].append(0)
    
    # 输出结果
    print(f"\nResults (n={n_runs}):")
    print(f"  Emergent births: {np.mean(results['emergent']['births']):.1f} ± {np.std(results['emergent']['births']):.1f}")
    print(f"  Optimized births: {np.mean(results['optimized']['births']):.1f} ± {np.std(results['optimized']['births']):.1f}")
    print(f"  Emergent CV: {np.mean(results['emergent']['birth_intervals']):.3f}")
    print(f"  Optimized CV: {np.mean(results['optimized']['birth_intervals']):.3f}")
    
    return results


# ==================== 实验2: 临界状态验证 ====================
def experiment2_criticality(steps: int = 10000):
    """
    实验2: 临界状态验证
    
    目标: 验证临界度与新维度生成的关联
    预期: 临界度峰值后5-20步内生成概率显著提高
    """
    print("\n" + "="*60)
    print("Experiment 2: Criticality Validation")
    print("="*60)
    
    np.random.seed(42)
    env = Environment(complexity=5, input_dim=10)
    system = EmergentFCRS(pool_capacity=10, input_dim=10)
    
    criticality_sequence = []
    birth_events = []
    
    for step in range(steps):
        x = env.generate()
        system.step(x)
        
        crit = system.criticality_detector.compute_criticality_score()
        criticality_sequence.append(crit)
        
        if system.step_count in system.birth_times:
            birth_events.append(step)
    
    # 计算滞后相关性
    correlations = []
    for lag in range(1, 30):
        births_after_crit_peak = []
        
        for i in range(len(criticality_sequence) - lag):
            if criticality_sequence[i] > 0.7:  # 临界峰值
                # 检查lag步后是否有生成
                for birth in birth_events:
                    if i < birth <= i + lag:
                        births_after_crit_peak.append(1)
                        break
                else:
                    births_after_crit_peak.append(0)
        
        if births_after_crit_peak:
            correlations.append(np.mean(births_after_crit_peak))
        else:
            correlations.append(0)
    
    max_corr = max(correlations) if correlations else 0
    max_lag = correlations.index(max_corr) if correlations else 0
    
    print(f"\nResults:")
    print(f"  Total births: {len(birth_events)}")
    print(f"  Mean criticality: {np.mean(criticality_sequence):.3f}")
    print(f"  Max correlation: {max_corr:.3f} at lag {max_lag}")
    print(f"  Criticality > 0.7 events: {sum(1 for c in criticality_sequence if c > 0.7)}")
    
    return {'correlations': correlations, 'birth_events': birth_events}


# ==================== 实验3: 多维度适应度 ====================
def experiment3_fitness():
    """
    实验3: 多维度适应度验证
    
    目标: 验证不同权重配置产生不同特征
    """
    print("\n" + "="*60)
    print("Experiment 3: Multi-dimensional Fitness")
    print("="*60)
    
    configs = [
        {'alpha': 0.6, 'beta': 0.2, 'gamma': 0.2, 'name': 'High-Prediction'},
        {'alpha': 0.2, 'beta': 0.6, 'gamma': 0.2, 'name': 'High-Compression'},
        {'alpha': 0.2, 'beta': 0.2, 'gamma': 0.6, 'name': 'High-Behavior'},
    ]
    
    results = {}
    
    for config in configs:
        np.random.seed(42)
        env = Environment(complexity=5, input_dim=10)
        system = EmergentFCRS(
            pool_capacity=10, input_dim=10,
            alpha=config['alpha'], beta=config['beta'], gamma=config['gamma']
        )
        
        for _ in range(5000):
            x = env.generate()
            system.step(x)
        
        results[config['name']] = {
            'births': system.emergent_births,
            'rejections': system.survival_rejections,
            'final_pool': len(system.representations)
        }
        
        print(f"\n{config['name']}:")
        print(f"  Births: {system.emergent_births}")
        print(f"  Rejections: {system.survival_rejections}")
        print(f"  Final pool: {len(system.representations)}")
    
    return results


# ==================== 实验4: 试用期筛选 ====================
def experiment4_probation():
    """
    实验4: 试用期筛选验证
    
    目标: 验证试用期机制能有效筛选维度
    预期: 30%-60%的试用期维度被淘汰
    """
    print("\n" + "="*60)
    print("Experiment 4: Probation Screening")
    print("="*60)
    
    np.random.seed(42)
    env = Environment(complexity=5, input_dim=10)
    system = EmergentFCRS(pool_capacity=10, input_dim=10)
    
    # 运行
    for _ in range(10000):
        x = env.generate()
        system.step(x)
    
    # 分析
    probation_total = system.emergent_births
    probation_rejected = system.survival_rejections
    rejection_rate = probation_rejected / max(1, probation_total)
    
    print(f"\nResults:")
    print(f"  Total probationary: {probation_total}")
    print(f"  Rejected: {probation_rejected}")
    print(f"  Rejection rate: {rejection_rate:.1%}")
    
    if 0.3 <= rejection_rate <= 0.6:
        print("  [OK] Within expected range (30%-60%)")
    
    return {'total': probation_total, 'rejected': probation_rejected, 'rate': rejection_rate}


# ==================== 实验5: 资源约束 ====================
def experiment5_resource():
    """
    实验5: 资源约束效应
    
    目标: 验证资源受限时的自组织行为
    预期: 资源减少时维度自适应缩减
    """
    print("\n" + "="*60)
    print("Experiment 5: Resource Constraint")
    print("="*60)
    
    resource_levels = [200, 150, 100, 50, 25]
    results = {}
    
    for resource in resource_levels:
        np.random.seed(42)
        
        # 模拟资源变化
        env = Environment(complexity=5, input_dim=10)
        
        # 创建不同容量的系统
        class AdaptiveCapacityFCRS(EmergentFCRS):
            def __init__(self, capacity):
                super().__init__(pool_capacity=capacity, input_dim=10)
                self.capacity = capacity
        
        system = AdaptiveCapacityFCRS(capacity=resource // 20)
        
        # 运行
        for _ in range(2000):
            x = env.generate()
            system.step(x)
        
        results[resource] = {
            'births': system.emergent_births,
            'final_pool': len(system.representations)
        }
        
        print(f"Resource {resource}: births={system.emergent_births}, final={len(system.representations)}")
    
    return results


# ==================== Main ====================
def main():
    print("="*60)
    print("FCRS-v5.2 Validation Experiments")
    print("="*60)
    
    # 运行所有实验
    exp1_results = experiment1_comparison(n_runs=30, steps=5000)
    exp2_results = experiment2_criticality(steps=10000)
    exp3_results = experiment3_fitness()
    exp4_results = experiment4_probation()
    exp5_results = experiment5_resource()
    
    # 总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("Exp1: Emergent vs Optimized - COMPLETE")
    print("Exp2: Criticality Validation - COMPLETE")
    print("Exp3: Multi-dim Fitness - COMPLETE")
    print("Exp4: Probation Screening - COMPLETE")
    print("Exp5: Resource Constraint - COMPLETE")
    
    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results,
        'exp5': exp5_results
    }


if __name__ == "__main__":
    main()
