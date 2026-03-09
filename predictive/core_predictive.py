"""
预测导向有限竞争表征系统
FCRS-v5: 预测导向压缩 → 预测 → 选择 → 反馈 → 更新

核心理念：压缩的目标不是重构，而是预测未来
"""

import numpy as np
from typing import List, Tuple, Optional


class PredictiveCompressor:
    """
    预测导向压缩器
    
    核心思想：压缩的目标不是重构输入，而是预测未来
    压缩的质量不是用"重构误差"衡量，而是用"预测误差"衡量
    """
    
    def __init__(self, input_dim: int, compress_dim: int, learning_rate: float = 0.01):
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.lr = learning_rate
        
        # 编码器：输入 -> 压缩表示
        self.encoder_weights = np.random.randn(input_dim, compress_dim) * 0.1
        self.encoder_bias = np.zeros(compress_dim)
        
        # 预测器：当前压缩表示 -> 下一时刻压缩表示
        self.predictor_weights = np.random.randn(compress_dim, compress_dim) * 0.1
        self.predictor_bias = np.zeros(compress_dim)
        
        # 历史记录
        self.compressed_history = []
        self.prediction_errors = []
    
    def compress(self, x: np.ndarray) -> np.ndarray:
        """压缩：将高维输入压缩为低维表示"""
        # 线性投影
        compressed = x @ self.encoder_weights + self.encoder_bias
        # ReLU激活
        compressed = np.maximum(0, compressed)
        # L2归一化
        norm = np.linalg.norm(compressed)
        if norm > 0:
            compressed = compressed / norm
        return compressed
    
    def predict(self, compressed: np.ndarray) -> np.ndarray:
        """预测：基于当前压缩表示，预测下一时刻的压缩表示"""
        predicted = compressed @ self.predictor_weights + self.predictor_bias
        predicted = np.maximum(0, predicted)  # ReLU
        return predicted
    
    def update(self, compressed_curr: np.ndarray, compressed_next: np.ndarray):
        """更新预测器参数，基于预测误差"""
        # 预测
        predicted = self.predict(compressed_curr)
        
        # 计算预测误差
        error = compressed_next - predicted
        self.prediction_errors.append(np.linalg.norm(error))
        
        # 梯度更新
        self.predictor_weights -= self.lr * np.outer(compressed_curr, error)
        self.predictor_bias -= self.lr * error
    
    def evaluate_quality(self) -> dict:
        """评估压缩质量"""
        if len(self.prediction_errors) == 0:
            return {"mean_error": 0, "variance": 0, "trend": 0}
        
        errors = np.array(self.prediction_errors[-100:])
        return {
            "mean_error": float(np.mean(errors)),
            "variance": float(np.var(errors)),
            "trend": float(np.mean(errors[-10:]) - np.mean(errors[:10])) if len(errors) >= 20 else 0
        }


class PredictiveSelector:
    """
    预测驱动表征选择器
    
    核心思想：选择"预测误差最小"的表征，而非"与当前输入最匹配"的表征
    """
    
    def __init__(self, representations: List[np.ndarray], compressor: PredictiveCompressor):
        self.representations = representations
        self.compressor = compressor
        self.prediction_scores = {}
    
    def compute_scores(self, current_input: np.ndarray) -> dict:
        """计算每个表征的预测得分"""
        scores = {}
        
        for i, rep in enumerate(self.representations):
            # 用表征压缩当前输入
            compressed = self.compressor.compress(current_input)
            
            # 预测下一时刻
            predicted = self.compressor.predict(compressed)
            
            # 得分 = 负预测误差（误差越小，得分越高）
            score = -np.linalg.norm(predicted - compressed)
            scores[i] = score
        
        self.prediction_scores = scores
        return scores
    
    def select(self, current_input: np.ndarray, exploration_rate: float = 0.1) -> int:
        """ε-贪心选择"""
        # 探索
        if np.random.random() < exploration_rate:
            return np.random.randint(len(self.representations))
        
        # 计算预测得分
        scores = self.compute_scores(current_input)
        
        # 选择得分最高的
        best_idx = max(scores, key=scores.get)
        return best_idx


class PredictiveFCRS:
    """
    预测导向有限竞争表征系统
    
    整合：压缩 → 预测 → 选择 → 反馈 → 更新
    """
    
    def __init__(self,
                 input_dim: int = 10,
                 compress_dim: int = 5,
                 pool_capacity: int = 10,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1):
        
        self.input_dim = input_dim
        self.compress_dim = compress_dim
        self.pool_capacity = pool_capacity
        self.exploration_rate = exploration_rate
        
        # 核心组件
        self.compressor = PredictiveCompressor(input_dim, compress_dim, learning_rate)
        self.representations: List[np.ndarray] = []
        self.selector: Optional[PredictiveSelector] = None
        
        # 统计
        self.step_count = 0
        self.prediction_error_history = []
        self.reconstruction_error_history = []
        
        # 初始化表征池
        self._init_representations()
    
    def _init_representations(self):
        """初始化表征池"""
        for _ in range(3):
            rep = np.random.randn(self.input_dim)
            rep = rep / (np.linalg.norm(rep) + 1e-8)
            self.representations.append(rep)
        
        self.selector = PredictiveSelector(self.representations, self.compressor)
    
    def step(self, x: np.ndarray) -> dict:
        """执行一步"""
        self.step_count += 1
        
        # 1. 压缩
        compressed = self.compressor.compress(x)
        self.compressor.compressed_history.append(compressed)
        
        # 2. 预测 & 更新
        if len(self.compressor.compressed_history) >= 2:
            compressed_prev = self.compressor.compressed_history[-2]
            predicted = self.compressor.predict(compressed_prev)
            
            # 预测误差
            pred_error = np.linalg.norm(predicted - compressed)
            self.prediction_error_history.append(pred_error)
            
            # 更新预测器
            self.compressor.update(compressed_prev, compressed)
        
        # 3. 选择表征
        selected_idx = self.selector.select(x, self.exploration_rate)
        selected_rep = self.representations[selected_idx]
        
        # 4. 重构误差
        recon_error = np.linalg.norm(x - selected_rep)
        self.reconstruction_error_history.append(recon_error)
        
        # 5. 更新表征
        self.representations[selected_idx] += 0.01 * (x - selected_rep)
        self.representations[selected_idx] /= (np.linalg.norm(self.representations[selected_idx]) + 1e-8)
        
        return {
            "step": self.step_count,
            "prediction_error": self.prediction_error_history[-1] if self.prediction_error_history else None,
            "reconstruction_error": recon_error,
            "pool_size": len(self.representations)
        }
    
    def run(self, env, steps: int) -> dict:
        """运行多步"""
        for _ in range(steps):
            x = env.generate_input()
            self.step(x)
        
        return self.get_statistics()
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        pred_errors = np.array(self.prediction_error_history[-100:]) if self.prediction_error_history else np.array([])
        recon_errors = np.array(self.reconstruction_error_history[-100:]) if self.reconstruction_error_history else np.array([])
        
        return {
            "step": self.step_count,
            "pool_size": len(self.representations),
            "mean_prediction_error": float(np.mean(pred_errors)) if len(pred_errors) > 0 else None,
            "mean_reconstruction_error": float(np.mean(recon_errors)) if len(recon_errors) > 0 else None,
            "compression_ratio": self.input_dim / self.compress_dim
        }


class RandomBaseline:
    """随机基线"""
    
    def __init__(self, compress_dim: int = 5):
        self.compress_dim = compress_dim
    
    def compress(self, x: np.ndarray) -> np.ndarray:
        compressed = np.random.randn(self.compress_dim)
        return compressed / (np.linalg.norm(compressed) + 1e-8)
    
    def predict(self, compressed: np.ndarray) -> np.ndarray:
        return np.random.randn(self.compress_dim)
