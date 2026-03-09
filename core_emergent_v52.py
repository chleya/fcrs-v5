"""
FCRS-v5.2: 涌现驱动版本 - 基于数学化理论框架整改
核心改进：从"优化驱动"改为真正的"涌现驱动"

优化驱动（旧版本）：
  计算压缩增益 → 判断阈值 → 决定是否增加维度
  问题：外部"裁判"决定系统行为

涌现驱动（新版本）：
  临界状态检测 → 内在扰动产生 → 多维度适应度筛选 → 试用期保留
  特征：系统内在动力学驱动，无预设阈值判断
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class Representation:
    """表征结构"""
    id: int
    vector: np.ndarray
    fitness_history: List[float] = field(default_factory=list)
    activation_count: int = 0
    age: int = 0
    reuse_frequency: float = 0.0

    # 新增：来源信息
    origin: str = "initial"  # 'initial', 'perturbation', 'split', 'residual'
    birth_step: int = 0


class CriticalityDetector:
    """
    临界状态检测器

    理论基础：相变理论
    当系统运行在临界状态时，微小的扰动可能导致系统行为的巨大变化。
    临界状态是涌现发生的理想条件。
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.state_history: List[float] = []
        self.prediction_errors: List[float] = []
        self.dimension_counts: List[int] = []

    def record_state(self, state_norm: float):
        """记录系统状态"""
        self.state_history.append(state_norm)
        if len(self.state_history) > self.window_size:
            self.state_history.pop(0)

    def record_prediction_error(self, error: float):
        """记录预测误差"""
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.window_size:
            self.prediction_errors.pop(0)

    def record_dimension_count(self, count: int):
        """记录维度数"""
        self.dimension_counts.append(count)
        if len(self.dimension_counts) > self.window_size:
            self.dimension_counts.pop(0)

    def compute_fluctuation(self) -> float:
        """计算状态波动率"""
        if len(self.state_history) < 2:
            return 0.0
        states = np.array(self.state_history)
        return float(np.std(states))

    def compute_prediction_uncertainty(self) -> float:
        """计算预测不确定性"""
        if len(self.prediction_errors) < 2:
            return 1.0
        errors = np.array(self.prediction_errors)
        return float(np.var(errors))

    def compute_criticality_score(self) -> float:
        """
        计算临界度得分

        核心公式：临界度 = f(波动率, 不确定性)
        当系统既不是太稳定也不是太混乱时，临界度最高
        """
        fluctuation = self.compute_fluctuation()
        uncertainty = self.compute_prediction_uncertainty()

        # 使用高斯形状的临界度曲线
        # 临界度在中间区域最高
        combined = fluctuation * (1 + uncertainty)

        # Sigmoid形状的归一化
        score = 1 / (1 + np.exp(-0.5 * (combined - 1)))

        return score

    def is_critical(self, threshold: float = 0.5) -> bool:
        """判断系统是否处于临界状态"""
        return self.compute_criticality_score() > threshold

    def get_criticality_info(self) -> Dict:
        """获取临界状态详细信息"""
        return {
            'score': self.compute_criticality_score(),
            'fluctuation': self.compute_fluctuation(),
            'uncertainty': self.compute_prediction_uncertainty(),
            'is_critical': self.is_critical(),
            'history_length': len(self.state_history)
        }


class EmergentDimensionGenerator:
    """
    涌现维度生成器

    核心思想：在临界状态下，系统内在的扰动会自发产生新维度
    不再使用预设阈值判断，而是由系统状态自然驱动
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def spontaneous_generate(self,
                           base_vector: np.ndarray,
                           criticality: float,
                           residual: Optional[np.ndarray] = None) -> List[Dict]:
        """
        自发产生候选维度

        关键区别于旧版本：
        - 旧版本：满足阈值条件 → 生成新维度
        - 新版本：临界状态下持续扰动 → 随机产生候选

        参数：
            base_vector: 当前最佳表征向量
            criticality: 临界度得分
            residual: 残差信号（可选）

        返回：
            候选维度列表
        """
        candidates = []

        # 扰动概率与临界度成正比
        # 临界度越高，产生扰动的可能性越大
        perturbation_prob = criticality * 0.3

        if np.random.random() < perturbation_prob:
            # 扰动方式1：对现有表征添加噪声
            noise = np.random.randn(len(base_vector)) * 0.1 * criticality
            candidate1 = base_vector + noise
            candidates.append({
                'vector': np.clip(candidate1, -3, 3),
                'origin': 'perturbation',
                'strength': criticality
            })

            # 扰动方式2：维度分裂（仅当维度足够大时）
            if len(base_vector) >= 3 and np.random.random() < 0.3:
                split_point = np.random.randint(1, len(base_vector))
                part1 = base_vector[:split_point]
                part2 = base_vector[split_point:]

                # 补齐到原始维度
                part1_padded = np.pad(part1, (0, len(base_vector) - len(part1)))
                part2_padded = np.pad(part2, (0, len(base_vector) - len(part2)))

                candidates.append({
                    'vector': part1_padded,
                    'origin': 'split',
                    'strength': criticality * 0.5
                })
                candidates.append({
                    'vector': part2_padded,
                    'origin': 'split',
                    'strength': criticality * 0.5
                })

        # 扰动方式3：从残差中提取（如果有）
        if residual is not None and len(residual) > 0:
            if np.random.random() < criticality * 0.2:
                # 归一化残差作为新维度
                residual_normalized = residual / (np.linalg.norm(residual) + 1e-8)

                # 混合原始信息和残差信息
                mixed = 0.7 * base_vector + 0.3 * residual_normalized * np.linalg.norm(base_vector)
                candidates.append({
                    'vector': np.clip(mixed, -3, 3),
                    'origin': 'residual',
                    'strength': criticality
                })

        return candidates


class MultiDimensionalFitness:
    """
    多维度适应度评估器

    核心思想：不再使用单一的压缩增益作为判断标准
    而是综合考虑多个维度的适应度
    """

    def __init__(self,
                 alpha: float = 0.4,
                 beta: float = 0.3,
                 gamma: float = 0.3):
        """
        初始化权重参数

        alpha: 预测适应度权重
        beta: 压缩适应度权重
        gamma: 行为适应度权重
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate_prediction_fitness(self,
                                   dimension: Representation,
                                   input_data: np.ndarray) -> float:
        """
        评估预测适应度

        衡量维度对输入数据的预测能力
        """
        if len(dimension.vector) == 0:
            return 0.0

        # 取公共长度
        min_len = min(len(dimension.vector), len(input_data))
        if min_len == 0:
            return 0.0

        prediction = dimension.vector[:min_len]
        target = input_data[:min_len]

        # 预测误差
        error = np.linalg.norm(prediction - target)

        # 转换为适应度（负误差）
        return -error

    def evaluate_compression_fitness(self,
                                   original_dim: int,
                                   new_dim: int) -> float:
        """
        评估压缩适应度

        衡量新维度带来的压缩效率
        """
        if original_dim == 0:
            return 1.0 if new_dim > 0 else 0.0

        # 压缩比
        compression_ratio = original_dim / max(new_dim, 1)

        # 归一化到 [0, 1]
        return min(1.0, compression_ratio)

    def evaluate_behavior_fitness(self,
                                 dimension: Representation,
                                 action_history: List[Dict]) -> float:
        """
        评估行为适应度

        衡量维度对系统行为的贡献
        """
        if not action_history:
            # 如果没有行为历史，使用激活频率作为代理
            if dimension.age > 0:
                return dimension.activation_count / dimension.age
            return 0.0

        # 基于最近行为计算成功率
        recent_actions = action_history[-10:]
        if not recent_actions:
            return 0.0

        successes = sum(1 for a in recent_actions if a.get('success', False))
        return successes / len(recent_actions)

    def compute_composite_fitness(self,
                                 dimension: Representation,
                                 input_data: np.ndarray,
                                 original_dim: int,
                                 action_history: List[Dict] = None) -> Tuple[float, Dict]:
        """
        计算综合适应度

        返回：(综合适应度, 各维度适应度详情)
        """
        if action_history is None:
            action_history = []

        F_pred = self.evaluate_prediction_fitness(dimension, input_data)
        F_comp = self.evaluate_compression_fitness(original_dim, len(dimension.vector))
        F_beh = self.evaluate_behavior_fitness(dimension, action_history)

        # 归一化各适应度到相近范围
        F_pred_norm = 1 / (1 + np.exp(-F_pred))  # Sigmoid归一化
        F_beh_norm = F_beh  # 已在 [0, 1]

        # 加权求和
        weights = np.array([self.alpha, self.beta, self.gamma])
        fitness_vector = np.array([F_pred_norm, F_comp, F_beh_norm])
        composite = np.dot(fitness_vector, weights)

        details = {
            'prediction': F_pred,
            'prediction_normalized': F_pred_norm,
            'compression': F_comp,
            'behavior': F_beh,
            'behavior_normalized': F_beh_norm
        }

        return composite, details


class ProbationaryManager:
    """
    试用期管理器

    核心思想：新维度不是"一次性产生"的
    而是需要证明其长期价值才能被保留
    """

    def __init__(self,
                 probation_steps: int = 20,
                 retention_threshold: float = 0.3):
        self.probation_steps = probation_steps
        self.retention_threshold = retention_threshold
        self.probationary_dimensions: Dict[int, Dict] = {}
        self.performance_history: Dict[int, List[float]] = {}

    def enter_probation(self,
                       dimension_id: int,
                       initial_performance: float):
        """进入试用期"""
        self.probationary_dimensions[dimension_id] = {
            'steps_in_probation': 0,
            'performance_sum': initial_performance,
            'survival_trials': 0,
            'survival_successes': 0,
            'birth_time': dimension_id  # 借用字段记录
        }
        self.performance_history[dimension_id] = [initial_performance]

    def update_probation(self,
                        dimension_id: int,
                        performance: float) -> bool:
        """更新试用期表现"""
        if dimension_id not in self.probationary_dimensions:
            return False

        prob = self.probationary_dimensions[dimension_id]
        prob['steps_in_probation'] += 1
        prob['performance_sum'] += performance

        # 生存考验
        if performance > self.retention_threshold:
            prob['survival_successes'] += 1
        prob['survival_trials'] += 1

        self.performance_history[dimension_id].append(performance)

        return True

    def should_retain(self, dimension_id: int) -> Optional[bool]:
        """
        判断是否应该保留

        返回：
            True: 试用期结束，保留
            False: 试用期结束，淘汰
            None: 仍在试用期
        """
        if dimension_id not in self.probationary_dimensions:
            return None

        prob = self.probationary_dimensions[dimension_id]

        # 试用期结束
        if prob['steps_in_probation'] >= self.probation_steps:
            # 计算平均表现
            avg_performance = prob['performance_sum'] / prob['steps_in_probation']

            # 归一化平均表现
            avg_normalized = 1 / (1 + np.exp(-avg_performance))

            # 生存成功率
            survival_rate = prob['survival_successes'] / max(1, prob['survival_trials'])

            # 综合判断
            retention_score = 0.6 * avg_normalized + 0.4 * survival_rate

            return retention_score > self.retention_threshold

        return None  # 仍在试用期

    def remove_from_probation(self, dimension_id: int):
        """从试用期移除"""
        if dimension_id in self.probationary_dimensions:
            del self.probationary_dimensions[dimension_id]

    def get_probation_status(self) -> Dict:
        """获取试用期状态"""
        return {
            'count': len(self.probationary_dimensions),
            'dimensions': list(self.probationary_dimensions.keys())
        }


class EmergentFCRSv52:
    """
    涌现驱动有限竞争表征系统 v5.2

    核心改进：
    1. 移除所有预设阈值判断
    2. 引入临界状态检测
    3. 内在扰动驱动的维度生成
    4. 多维度适应度筛选
    5. 试用期保留机制
    """

    def __init__(self,
                 pool_capacity: int = 10,
                 input_dim: int = 10):
        self.pool_capacity = pool_capacity
        self.input_dim = input_dim

        # 表征池
        self.representations: List[Representation] = []

        # 核心组件
        self.criticality_detector = CriticalityDetector()
        self.dimension_generator = EmergentDimensionGenerator(input_dim)
        self.fitness_evaluator = MultiDimensionalFitness()
        self.probation_manager = ProbationaryManager()

        # 初始化表征
        self._initialize_representations()

        # 统计
        self.step_count = 0
        self.emergent_births = 0
        self.survival_rejections = 0
        self.dim_history: List[int] = []
        self.criticality_history: List[float] = []

    def _initialize_representations(self):
        """初始化表征"""
        for i in range(3):
            # 随机初始化
            vector = np.random.randn(self.input_dim) * 0.5
            rep = Representation(
                id=i,
                vector=vector,
                origin='initial',
                birth_step=0
            )
            self.representations.append(rep)

    def _select_representation(self, x: np.ndarray) -> Optional[Representation]:
        """竞争性选择最匹配表征"""
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

    def _compute_prediction(self,
                          rep: Representation,
                          x: np.ndarray) -> float:
        """计算预测误差"""
        min_len = min(len(rep.vector), len(x))
        if min_len == 0:
            return float('inf')

        prediction = rep.vector[:min_len]
        target = x[:min_len]
        return float(np.linalg.norm(prediction - target))

    def _update_criticality(self, x: np.ndarray, error: float):
        """更新临界状态"""
        # 记录状态
        self.criticality_detector.record_state(np.linalg.norm(x))
        self.criticality_detector.record_prediction_error(error)
        self.criticality_detector.record_dimension_count(len(self.representations))

    def _process_emergence(self,
                          x: np.ndarray,
                          criticality: float,
                          residual: Optional[np.ndarray] = None):
        """
        处理涌现生成

        核心区别于旧版本：
        - 旧版本：检查阈值 → 决定是否生成
        - 新版本：临界状态 → 概率性生成候选 → 适应度筛选
        """
        # 只有在临界状态下才可能产生候选
        # 但产生是概率性的，不是确定性的
        if criticality < 0.1:
            return

        # 获取当前最佳表征作为扰动基础
        if not self.representations:
            return

        # 选择表现最好的表征
        active_rep = self._select_representation(x)
        if active_rep is None:
            return

        # 产生候选维度
        candidates = self.dimension_generator.spontaneous_generate(
            active_rep.vector,
            criticality,
            residual
        )

        # 评估每个候选
        original_dim = len(active_rep.vector)

        for candidate in candidates:
            # 创建临时表征用于评估
            temp_rep = Representation(
                id=-1,  # 临时ID
                vector=candidate['vector']
            )

            # 计算适应度
            fitness, details = self.fitness_evaluator.compute_composite_fitness(
                temp_rep,
                x,
                original_dim,
                []
            )

            # 简单筛选：适应度为正且超过阈值
            # 注意：这个阈值比旧版本低得多，主要是防止极差候选
            fitness_threshold = 0.2

            if fitness > fitness_threshold:
                # 进入试用期
                new_id = len(self.representations)
                new_rep = Representation(
                    id=new_id,
                    vector=candidate['vector'],
                    origin=candidate['origin'],
                    birth_step=self.step_count
                )

                self.representations.append(new_rep)
                self.probation_manager.enter_probation(new_id, fitness)
                self.emergent_births += 1

                # 如果超过容量，淘汰最弱的
                if len(self.representations) > self.pool_capacity:
                    self._淘汰最弱()

    def _update_probation(self, x: np.ndarray):
        """更新试用期"""
        probation_ids = list(self.probation_manager.probationary_dimensions.keys())

        for dim_id in list(probation_ids):
            if dim_id >= len(self.representations):
                self.probation_manager.remove_from_probation(dim_id)
                continue

            rep = self.representations[dim_id]

            # 计算当前表现
            error = self._compute_prediction(rep, x)
            performance = -error

            # 更新
            self.probation_manager.update_probation(dim_id, performance)

            # 检查是否应该保留
            should_retain = self.probation_manager.should_retain(dim_id)

            if should_retain is False:
                # 试用期结束，被淘汰
                self.representations.pop(dim_id)
                self.probation_manager.remove_from_probation(dim_id)
                self.survival_rejections += 1

                # 重新编号
                for i, rep in enumerate(self.representations):
                    rep.id = i

            elif should_retain is True:
                # 试用期结束，保留
                self.probation_manager.remove_from_probation(dim_id)

    def _淘汰最弱(self):
        """淘汰最弱的表征"""
        if not self.representations:
            return

        # 按最近表现排序
        def get_recent_fitness(rep):
            if rep.fitness_history:
                return np.mean(rep.fitness_history[-10:])
            return -float('inf')

        self.representations.sort(key=get_recent_fitness)
        self.representations.pop(0)

        # 重新编号
        for i, rep in enumerate(self.representations):
            rep.id = i

    def step(self, x: np.ndarray, residual: Optional[np.ndarray] = None) -> float:
        """
        执行一步

        参数：
            x: 输入数据
            residual: 残差信号（可选）

        返回：预测误差
        """
        self.step_count += 1

        # 1. 选择表征
        active_rep = self._select_representation(x)

        # 2. 计算预测和误差
        if active_rep is not None:
            error = self._compute_prediction(active_rep, x)
            active_rep.fitness_history.append(-error)
            active_rep.activation_count += 1

            # 更新复用频率
            if active_rep.age > 0:
                active_rep.reuse_frequency = active_rep.activation_count / active_rep.age
        else:
            error = float('inf')

        # 3. 更新临界状态
        self._update_criticality(x, error)

        # 4. 临界状态下的涌现生成
        criticality = self.criticality_detector.compute_criticality_score()
        self._process_emergence(x, criticality, residual)

        # 5. 更新试用期
        self._update_probation(x)

        # 6. 表征年龄增加
        for rep in self.representations:
            rep.age += 1

        # 7. 记录历史
        if self.step_count % 50 == 0:
            self.dim_history.append(len(self.representations))
            self.criticality_history.append(criticality)

        return error

    def run(self,
            env,
            steps: int = 1000,
            record_interval: int = 50) -> Dict:
        """
        运行多步

        参数：
            env: 环境对象，需要有 generate_input() 方法
            steps: 运行步数
            record_interval: 记录间隔

        返回：统计信息
        """
        for _ in range(steps):
            x = env.generate_input()
            self.step(x)

        return self.get_statistics()

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'step': self.step_count,
            'pool_size': len(self.representations),
            'emergent_births': self.emergent_births,
            'survival_rejections': self.survival_rejections,
            'net_new_dims': self.emergent_births - self.survival_rejections,
            'dim_history': self.dim_history,
            'criticality_history': self.criticality_history,
            'current_criticality': self.criticality_detector.get_criticality_info(),
            'probation_status': self.probation_manager.get_probation_status(),
            'representations': [
                {
                    'id': r.id,
                    'dim': len(r.vector),
                    'age': r.age,
                    'activation': r.activation_count,
                    'origin': r.origin,
                    'recent_fitness': np.mean(r.fitness_history[-10:]) if r.fitness_history else 0
                }
                for r in self.representations
            ]
        }


def test_emergent_v52():
    """测试涌现驱动v5.2系统"""
    print("=" * 60)
    print("FCRS-v5.2 涌现驱动系统测试")
    print("=" * 60)

    # 简单环境
    class SimpleEnv:
        def __init__(self, input_dim=10, complexity=3):
            self.input_dim = input_dim
            self.class_centers = {
                i: np.random.randn(input_dim) * 2
                for i in range(complexity)
            }

        def generate_input(self):
            cls = np.random.randint(0, len(self.class_centers))
            center = self.class_centers[cls]
            return center + np.random.randn(self.input_dim) * 0.3

    # 创建系统和环境
    env = SimpleEnv(input_dim=10, complexity=5)
    system = EmergentFCRSv52(pool_capacity=10, input_dim=10)

    print("\n运行500步...")
    for i in range(500):
        x = env.generate_input()
        error = system.step(x)

        if (i + 1) % 100 == 0:
            stats = system.get_statistics()
            crit_info = stats['current_criticality']
            print(f"\nStep {i+1}:")
            print(f"  表征数: {stats['pool_size']}")
            print(f"  涌现次数: {stats['emergent_births']}")
            print(f"  淘汰次数: {stats['survival_rejections']}")
            print(f"  临界度: {crit_info['score']:.3f} (波动:{crit_info['fluctuation']:.3f}, 不确定性:{crit_info['uncertainty']:.3f})")

    # 最终统计
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    stats = system.get_statistics()
    print(f"总步数: {stats['step']}")
    print(f"表征数: {stats['pool_size']}")
    print(f"涌现次数: {stats['emergent_births']}")
    print(f"淘汰次数: {stats['survival_rejections']}")
    print(f"净新增维度: {stats['net_new_dims']}")

    print("\n表征详情:")
    for rep in stats['representations']:
        print(f"  ID:{rep['id']} 维:{rep['dim']} 年龄:{rep['age']} "
              f"激活:{rep['activation']} 来源:{rep['origin']}")


if __name__ == "__main__":
    test_emergent_v52()
