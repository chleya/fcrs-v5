"""
FCRS-v5 Predictive: 核心模块 (Bug修复版)

修复内容:
1. 预测选择机制 - 使用长期预测能力评估
2. 权重初始化 - Xavier初始化
3. 状态转移模型 - 修正ReLU顺序
4. 配置管理 - 统一参数
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ==================== 配置 ====================

class Config:
    """统一配置"""
    # 基础参数
    INPUT_DIM = 10
    COMPRESS_DIM = 3
    POOL_CAPACITY = 10
    
    # 训练参数
    LEARNING_RATE = 0.01
    EXPLORATION_RATE = 0.1
    
    # 优化参数
    Xavier_FACTOR = 1.0  # Xavier初始化系数
    
    @classmethod
    def set(cls, **kwargs):
        for k, v in kwargs.items():
            if hasattr(cls, k):
                setattr(cls, k, v)


# ==================== 工具函数 ====================

def xavier_init(fan_in: int, fan_out: int, factor: float = 1.0) -> np.ndarray:
    """Xavier初始化"""
    limit = factor * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def stable_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """安全归一化"""
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    return x / norm


# ==================== 预测器 ====================

class Predictor:
    """
    预测器：学习状态转移 P(s'|s)
    
    修复:
    - Xavier初始化
    - 更好的归一化处理
    """
    
    def __init__(self, dim: int, lr: float = 0.01):
        self.dim = dim
        self.lr = lr
        
        # Xavier初始化
        self.W = xavier_init(dim, dim)
        self.b = np.zeros(dim)
        
        # 历史预测误差
        self.errors = []
        self.error_history = []
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """预测下一状态"""
        # 线性变换
        out = state @ self.W + self.b
        
        # 归一化 (先归一化，再激活，防止除零)
        out = stable_normalize(out)
        
        # ReLU激活
        out = np.maximum(0, out)
        
        # 再次归一化
        out = stable_normalize(out)
        
        return out
    
    def update(self, state_curr: np.ndarray, state_next: np.ndarray):
        """更新预测器"""
        # 预测
        pred = self.predict(state_curr)
        
        # 计算误差
        error = state_next - pred
        self.errors.append(np.linalg.norm(error))
        self.error_history.append(np.linalg.norm(error))
        
        # 限制历史长度
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
        
        # 梯度更新
        self.W -= self.lr * np.outer(state_curr, error)
        self.b -= self.lr * error
    
    def get_mean_error(self) -> float:
        """获取平均预测误差"""
        if len(self.error_history) == 0:
            return 1.0
        return np.mean(self.error_history[-100:])


# ==================== 表征 ====================

class Representation:
    """
    表征：包含向量和其专用预测器
    
    修复:
    - 每个表征有自己的预测器
    - 更好的初始化
    """
    
    def __init__(self, dim: int, lr: float = 0.01):
        self.dim = dim
        
        # 表征向量 (Xavier初始化)
        self.vector = xavier_init(dim, 1).flatten()
        self.vector = stable_normalize(self.vector)
        
        # 表征自己的预测器
        self.predictor = Predictor(dim, lr)
    
    def get_prediction_error(self) -> float:
        """获取该表征的预测误差"""
        return self.predictor.get_mean_error()


# ==================== 表征池 ====================

class RepresentationPool:
    """
    表征池：管理多个表征
    
    修复:
    - 基于预测误差选择（而非重构误差）
    - 真正的预测导向选择
    """
    
    def __init__(self, dim: int, capacity: int, lr: float = 0.01):
        self.dim = dim
        self.capacity = capacity
        self.lr = lr
        
        # 表征列表
        self.reps: List[Representation] = []
        
        # 选择统计
        self.selection_counts = []
    
    def add(self, rep: Representation):
        """添加表征"""
        if len(self.reps) < self.capacity:
            self.reps.append(rep)
    
    def initialize(self, n: int = 3):
        """初始化n个随机表征"""
        for _ in range(n):
            rep = Representation(self.dim, self.lr)
            self.add(rep)
    
    def select_by_prediction(self, x: np.ndarray, exploration: float = 0.1) -> int:
        """
        基于预测选择表征
        
        修复: 使用每个表征自己的预测器评估预测能力
        """
        # ε-贪心探索
        if np.random.random() < exploration:
            idx = np.random.randint(len(self.reps))
            self.selection_counts.append(idx)
            return idx
        
        # 计算每个表征的"预测价值"
        # 策略: 选择"历史预测误差最小"的表征
        scores = []
        for i, rep in enumerate(self.reps):
            # 方法1: 表征自己的预测误差
            pred_error = rep.get_prediction_error()
            
            # 方法2: 当前输入与表征的匹配度
            recon_error = np.linalg.norm(rep.vector - x)
            
            # 综合: 预测误差为主，重构误差为辅
            # 预测误差越小越好，重构误差越小越好
            score = -pred_error * 0.7 - recon_error * 0.3
            scores.append(score)
        
        # 选择得分最高的
        best_idx = np.argmax(scores)
        self.selection_counts.append(best_idx)
        
        return best_idx
    
    def select_by_reconstruction(self, x: np.ndarray) -> int:
        """基于重构选择（对照组）"""
        scores = [np.linalg.norm(r.vector - x) for r in self.reps]
        return np.argmin(scores)
    
    def update(self, idx: int, x: np.ndarray, state_curr: np.ndarray, state_next: np.ndarray):
        """更新选中的表征"""
        rep = self.reps[idx]
        
        # 更新表征向量
        rep.vector += self.lr * (x - rep.vector)
        rep.vector = stable_normalize(rep.vector)
        
        # 更新表征的预测器
        rep.predictor.update(state_curr, state_next)


# ==================== 状态转移模型 ====================

class StateTransitionModel:
    """
    状态转移模型: P(s'|s, a)
    
    修复:
    - 修正ReLU和归一化顺序
    - 添加异常处理
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 权重 (SA -> next_state)
        self.W = xavier_init(state_dim + action_dim, state_dim)
        self.b = np.zeros(state_dim)
    
    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """预测下一状态"""
        # 拼接s和a
        sa = np.concatenate([state, action])
        
        # 线性变换
        next_state = sa @ self.W + self.b
        
        # 修复: 先归一化，再激活，再归一化
        next_state = stable_normalize(next_state)
        next_state = np.maximum(0, next_state)  # ReLU
        next_state = stable_normalize(next_state)
        
        return next_state
    
    def update(self, state: np.ndarray, action: np.ndarray, next_state_target: np.ndarray):
        """更新模型"""
        # 预测
        next_state_pred = self.predict(state, action)
        
        # 误差
        error = next_state_target - next_state_pred
        
        # 梯度更新
        sa = np.concatenate([state, action])
        self.W -= self.lr * np.outer(sa, error)
        self.b -= self.lr * error
    
    def imagine(self, state: np.ndarray, actions: List[np.ndarray]) -> List[np.ndarray]:
        """心理模拟: 想象执行一系列动作后的状态"""
        current = state
        trajectory = [current]
        
        for a in actions:
            current = self.predict(current, a)
            trajectory.append(current)
        
        return trajectory


# ==================== 整合系统 ====================

class PredictiveSystem:
    """
    预测导向有限竞争表征系统 (Bug修复版)
    
    核心: 压缩 → 预测 → 选择 → 更新
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # 表征池
        self.pool = RepresentationPool(
            dim=self.config.INPUT_DIM,
            capacity=self.config.POOL_CAPACITY,
            lr=self.config.LEARNING_RATE
        )
        self.pool.initialize(3)
        
        # 状态转移模型
        self.transition = StateTransitionModel(
            state_dim=self.config.INPUT_DIM,
            action_dim=self.config.INPUT_DIM,
            lr=self.config.LEARNING_RATE
        )
        
        # 历史
        self.state_history = []
        self.selection_history = []
        
        # 统计
        self.pred_errors = []
        self.recon_errors = []
    
    def step(self, x: np.ndarray, explore: float = None) -> dict:
        """执行一步"""
        explore = explore or self.config.EXPLORATION_RATE
        
        # 1. 获取当前状态
        if len(self.state_history) > 0:
            state_curr = self.state_history[-1]
        else:
            state_curr = np.zeros(self.config.INPUT_DIM)
        
        # 2. 基于预测选择表征
        idx = self.pool.select_by_prediction(x, exploration=explore)
        rep = self.pool.reps[idx]
        
        # 3. 重构误差
        recon_err = np.linalg.norm(rep.vector - x)
        self.recon_errors.append(recon_err)
        
        # 4. 预测更新
        state_next = stable_normalize(x)
        
        # 更新表征的预测器
        rep.predictor.update(state_curr, state_next)
        
        # 更新表征向量
        self.pool.update(idx, x, state_curr, state_next)
        
        # 5. 记录
        self.state_history.append(state_next)
        self.selection_history.append(idx)
        
        # 限制历史
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        return {
            "recon_error": recon_err,
            "pred_error": rep.predictor.get_mean_error(),
            "selected_idx": idx
        }
    
    def run(self, env, steps: int, explore: float = None) -> dict:
        """运行多步"""
        for _ in range(steps):
            x = env.generate_input()
            self.step(x, explore)
        
        return self.get_statistics()
    
    def get_statistics(self) -> dict:
        """获取统计"""
        recon = np.array(self.recon_errors[-100:]) if self.recon_errors else np.array([0])
        
        return {
            "mean_recon_error": float(np.mean(recon)),
            "std_recon_error": float(np.std(recon)),
            "pool_size": len(self.pool.reps),
            "selections": self.pool.selection_counts[-100:]
        }


# ==================== 基线系统 ====================

class ReconstructionBaseline:
    """重构基线: 使用重构误差选择"""
    
    def __init__(self, dim: int, capacity: int, lr: float = 0.01):
        self.dim = dim
        self.reps = [xavier_init(dim, 1).flatten() for _ in range(capacity)]
        for r in self.reps:
            r /= (np.linalg.norm(r) + 1e-8)
        self.lr = lr
        self.errors = []
    
    def step(self, x: np.ndarray) -> float:
        # 基于重构选择
        idx = np.argmin([np.linalg.norm(r - x) for r in self.reps])
        
        # 重构误差
        err = np.linalg.norm(self.reps[idx] - x)
        self.errors.append(err)
        
        # 更新
        self.reps[idx] += self.lr * (x - self.reps[idx])
        self.reps[idx] /= (np.linalg.norm(self.reps[idx]) + 1e-8)
        
        return err


class RandomBaseline2:
    """随机基线"""
    
    def __init__(self, dim: int, capacity: int):
        self.dim = dim
        self.reps = [xavier_init(dim, 1).flatten() for _ in range(capacity)]
        for r in self.reps:
            r /= (np.linalg.norm(r) + 1e-8)
        self.errors = []
    
    def step(self, x: np.ndarray) -> float:
        # 随机选择
        idx = np.random.randint(len(self.reps))
        
        err = np.linalg.norm(self.reps[idx] - x)
        self.errors.append(err)
        
        return err


# ==================== 简单环境 ====================

class SimpleEnv:
    """简单环境: 多类高斯分布"""
    
    def __init__(self, dim: int = 10, n_classes: int = 5, noise: float = 0.3):
        self.dim = dim
        self.n_classes = n_classes
        self.noise = noise
        
        # 类中心
        np.random.seed(42)
        self.centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
    
    def generate_input(self) -> np.ndarray:
        c = self.centers[np.random.randint(0, self.n_classes)]
        return c + np.random.randn(self.dim) * self.noise


# ==================== 测试 ====================

def test_core_fix():
    """测试核心修复"""
    print("="*60)
    print("Core Fix Test")
    print("="*60)
    
    results = {'pred': [], 'recon': [], 'random': []}
    
    for run in range(20):
        np.random.seed(run * 100)
        
        # 预测系统
        Config.set(LEARNING_RATE=0.01, EXPLORATION_RATE=0.1)
        env = SimpleEnv(10, 5, 0.3)
        sys = PredictiveSystem()
        
        for _ in range(500):
            x = env.generate_input()
            sys.step(x)
        
        results['pred'].append(np.mean(sys.recon_errors[-100:]))
        
        # 重构基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        baseline = ReconstructionBaseline(10, 3, 0.01)
        
        for _ in range(500):
            x = env.generate_input()
            baseline.step(x)
        
        results['recon'].append(np.mean(baseline.errors[-100:]))
        
        # 随机基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        rand = RandomBaseline2(10, 3)
        
        for _ in range(500):
            x = env.generate_input()
            rand.step(x)
        
        results['random'].append(np.mean(rand.errors[-100:]))
    
    print(f"\n预测选择:   {np.mean(results['pred']):.4f} +/- {np.std(results['pred']):.4f}")
    print(f"重构选择:   {np.mean(results['recon']):.4f} +/- {np.std(results['recon']):.4f}")
    print(f"随机选择:   {np.mean(results['random']):.4f} +/- {np.std(results['random']):.4f}")
    
    # 计算改进
    pred_vs_recon = (np.mean(results['recon']) - np.mean(results['pred'])) / np.mean(results['recon']) * 100
    pred_vs_random = (np.mean(results['random']) - np.mean(results['pred'])) / np.mean(results['random']) * 100
    
    print(f"\n预测 vs 重构: {pred_vs_recon:+.1f}%")
    print(f"预测 vs 随机: {pred_vs_random:+.1f}%")
    
    if pred_vs_recon > 5:
        print("\n✅ 预测选择显著优于重构选择!")
    elif pred_vs_recon > 0:
        print("\n⚠️ 预测选择略优于重构选择")
    else:
        print("\n❌ 预测选择未优于重构选择")


def test_selection_distribution():
    """测试选择分布"""
    print("\n" + "="*60)
    print("Selection Distribution Test")
    print("="*60)
    
    env = SimpleEnv(10, 5, 0.3)
    sys = PredictiveSystem()
    
    for _ in range(500):
        x = env.generate_input()
        sys.step(x)
    
    # 统计选择
    from collections import Counter
    counts = Counter(sys.pool.selection_counts)
    
    print(f"选择分布: {dict(counts)}")
    
    if len(counts) > 1:
        print("✅ 选择有多样性!")
    else:
        print("❌ 选择无多样性!")


if __name__ == "__main__":
    test_core_fix()
    test_selection_distribution()
    print("\nDone!")
