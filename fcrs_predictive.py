"""
FCRS-v5.3: Compression-Prediction Driven
核心: 总结压缩 → 预测 → 选择方向

智能 = 现状的总结压缩 → 预测 → 有选择的方向
"""

import numpy as np
from collections import deque


class CompressedState:
    """压缩状态"""
    def __init__(self, dim=5):
        self.dim = dim
        self.vector = np.zeros(dim)  # 压缩向量
        self.history = deque(maxlen=100)  # 原始历史
    
    def add(self, state):
        """添加到历史"""
        self.history.append(state.copy())
        if len(self.history) >= 10:
            self.compress()
    
    def compress(self):
        """压缩: PCA-like 简化"""
        if len(self.history) < 2:
            return
        
        # 简单压缩: 取最近N个的滑动平均 + 方差
        recent = np.array(list(self.history)[-10:])
        
        # 压缩为: 均值 + 主成分方向
        mean = np.mean(recent, axis=0)
        
        # 只保留前dim个主成分
        if len(mean) > self.dim:
            # 简单: 均匀采样
            indices = np.linspace(0, len(mean)-1, self.dim, dtype=int)
            self.vector = mean[indices]
        else:
            self.vector = mean[:self.dim]
        
        return self.vector
    
    def predict(self):
        """预测: 基于压缩状态的简单预测"""
        if len(self.history) < 5:
            return self.vector
        
        # 线性预测: 用最近的变化趋势
        recent = np.array(list(self.history)[-5:])
        diffs = np.diff(recent, axis=0)
        trend = np.mean(diffs, axis=0)
        
        # 压缩趋势
        if len(trend) > self.dim:
            indices = np.linspace(0, len(trend)-1, self.dim, dtype=int)
            trend_compact = trend[indices]
        else:
            trend_compact = trend[:self.dim]
        
        # 预测 = 当前 + 趋势
        predicted = self.vector + trend_compact * 0.5
        return predicted
    
    def error(self, actual):
        """预测误差"""
        pred = self.predict()
        return np.linalg.norm(pred - actual[:self.dim])


class Representation:
    """表征"""
    def __init__(self, id, vector, compression_state=None):
        self.id = id
        self.vector = vector  # 压缩向量
        self.compression_state = compression_state
        self.fitness = 0.0
        self.prediction_errors = []
        self.survival_score = 0.0


class PredictiveFCRS:
    """
    压缩-预测驱动的FCRS
    
    核心循环:
    观察 → 压缩 → 预测 → 选择行动 → 反馈 → 更新压缩
    """
    
    def __init__(self, input_dim=10, capacity=10, compress_dim=5):
        self.input_dim = input_dim
        self.capacity = capacity
        self.compress_dim = compress_dim
        
        # 表征池
        self.representations = []
        
        # 全局压缩状态
        self.global_compression = CompressedState(dim=compress_dim)
        
        # 经验历史
        self.experience_history = deque(maxlen=1000)
        
        # 初始化几个表征
        for i in range(3):
            vec = np.random.randn(compress_dim) * 0.5
            rep = Representation(i, vec, CompressedState(compress_dim))
            self.representations.append(rep)
        
        self.step_count = 0
    
    def compress_input(self, x):
        """压缩输入"""
        # 输入 -> 压缩
        if len(x) > self.compress_dim:
            indices = np.linspace(0, len(x)-1, self.compress_dim, dtype=int)
            compressed = x[indices]
        else:
            compressed = x[:self.compress_dim]
        return compressed
    
    def predict_next(self, rep, current_input):
        """用表征预测下一状态"""
        # 表征的压缩状态 + 当前输入 -> 预测
        if rep.compression_state is None:
            rep.compression_state = CompressedState(self.compress_dim)
        
        # 添加到历史
        rep.compression_state.add(current_input)
        
        # 预测
        predicted = rep.compression_state.predict()
        return predicted
    
    def evaluate_prediction(self, rep, actual_next):
        """评估预测质量"""
        predicted = rep.compression_state.predict()
        error = np.linalg.norm(predicted - actual_next[:self.compress_dim])
        return -error  # 负误差 = 适应度
    
    def select_best(self, x):
        """选择最佳表征: 预测最准的"""
        if not self.representations:
            return None
        
        # 评估每个表征的预测能力
        scores = []
        for rep in self.representations:
            # 用过去预测当前
            if rep.compression_state and len(rep.compression_state.history) > 0:
                # 历史最后一个 -> 预测当前
                last_state = rep.compression_state.history[-1]
                pred_error = np.linalg.norm(last_state[:self.compress_dim] - x[:self.compress_dim])
                score = -pred_error
            else:
                score = -1.0  # 无历史，给低分
            
            scores.append(score)
            rep.survival_score = score
        
        best_idx = np.argmax(scores)
        return self.representations[best_idx]
    
    def generate_candidate(self, best_rep, x):
        """基于预测误差生成新候选"""
        candidates = []
        
        # 预测误差大 -> 需要新表征
        if best_rep.compression_state:
            pred_error = best_rep.compression_state.error(x[:self.compress_dim])
        else:
            pred_error = 1.0
        
        # 误差大时，产生新表征
        if pred_error > 0.3 and np.random.random() < pred_error:
            # 新表征 = 当前表征 + 误差方向的学习
            new_vec = best_rep.vector + np.random.randn(self.compress_dim) * 0.2
            
            # 压缩当前输入
            new_compress = CompressedState(self.compress_dim)
            new_compress.add(x)
            
            candidates.append(Representation(
                len(self.representations), 
                new_vec, 
                new_compress
            ))
        
        return candidates
    
    def update(self, x):
        """主更新循环"""
        self.step_count += 1
        
        # 1. 压缩当前输入
        compressed = self.compress_input(x)
        
        # 2. 全局压缩状态更新
        self.global_compression.add(x)
        
        # 3. 记录经验
        self.experience_history.append({
            'input': x.copy(),
            'compressed': compressed.copy(),
            'step': self.step_count
        })
        
        # 4. 选择最佳表征 (基于预测能力)
        best_rep = self.select_best(compressed)
        
        if best_rep is None:
            return 0.0
        
        # 5. 计算当前预测误差
        if best_rep.compression_state and len(best_rep.compression_state.history) > 1:
            current_error = best_rep.compression_state.error(compressed)
        else:
            current_error = 1.0
        
        # 6. 预测下一状态
        predicted_next = self.predict_next(best_rep, compressed)
        
        # 7. 基于预测生成候选
        candidates = self.generate_candidate(best_rep, compressed)
        
        for candidate in candidates:
            # 评估候选的预测能力
            cand_score = self.evaluate_prediction(candidate, compressed)
            
            if cand_score > best_rep.survival_score - 0.1:
                # 比当前好或者差不多 -> 添加
                candidate.id = len(self.representations)
                self.representations.append(candidate)
        
        # 8. 容量限制 - 淘汰预测差的
        if len(self.representations) > self.capacity:
            # 按预测能力排序
            self.representations.sort(key=lambda r: r.survival_score)
            # 淘汰最差的
            self.representations.pop(0)
        
        # 9. 更新表征的压缩状态
        best_rep.compression_state.add(compressed)
        
        # 重新编号
        for i, rep in enumerate(self.representations):
            rep.id = i
        
        return current_error


class RandomFCRS:
    """基线: 随机选择，无预测"""
    
    def __init__(self, input_dim=10, capacity=10):
        self.capacity = capacity
        self.representations = []
        
        for i in range(3):
            vec = np.random.randn(5) * 0.5
            self.representations.append(Representation(i, vec))
        
        self.step_count = 0
    
    def update(self, x):
        self.step_count += 1
        
        # 随机选择
        best = np.random.choice(self.representations)
        
        # 随机生成新表征
        if np.random.random() < 0.3:
            new_vec = np.random.randn(5)
            self.representations.append(Representation(len(self.representations), new_vec))
        
        # 容量
        if len(self.representations) > self.capacity:
            self.representations.pop(0)
        
        return 1.0


# ==================== 测试 ====================
def test_prediction_driven():
    """测试预测驱动系统"""
    print("="*60)
    print("Test: Compression-Prediction Driven")
    print("="*60)
    
    np.random.seed(42)
    
    # 环境: 5类
    env_centers = {i: np.random.randn(10)*2 for i in range(5)}
    
    def generate():
        c = env_centers[np.random.randint(0, 5)]
        return c + np.random.randn(10)*0.3
    
    # 预测驱动系统
    system = PredictiveFCRS(input_dim=10, capacity=10, compress_dim=5)
    
    errors = []
    for step in range(2000):
        x = generate()
        error = system.update(x)
        errors.append(error)
        
        if step % 500 == 0:
            print(f"Step {step}: error={np.mean(errors[-100:]):.3f}, reps={len(system.representations)}")
    
    print(f"\nFinal: mean_error={np.mean(errors):.3f}, reps={len(system.representations)}")
    print(f"Compression state: {system.global_compression.vector[:3]}")
    
    return errors


def compare_with_random():
    """对比随机系统"""
    print("\n" + "="*60)
    print("Compare: Predictive vs Random")
    print("="*60)
    
    results = {'predictive': [], 'random': []}
    
    for run in range(10):
        np.random.seed(run * 100)
        
        env_centers = {i: np.random.randn(10)*2 for i in range(5)}
        def generate():
            c = env_centers[np.random.randint(0, 5)]
            return c + np.random.randn(10)*0.3
        
        # Predictive
        sys1 = PredictiveFCRS(10, 10, 5)
        errs1 = []
        for _ in range(1000):
            errs1.append(sys1.update(generate()))
        results['predictive'].append(np.mean(errs1))
        
        # Random
        np.random.seed(run * 100)
        sys2 = RandomFCRS(10, 10)
        errs2 = []
        for _ in range(1000):
            errs2.append(sys2.update(generate()))
        results['random'].append(np.mean(errs2))
    
    print(f"Predictive: {np.mean(results['predictive']):.3f}")
    print(f"Random: {np.mean(results['random']):.3f}")
    
    if np.mean(results['predictive']) < np.mean(results['random']):
        print("[OK] Prediction helps!")
    else:
        print("[?] No clear advantage")


def test_adaptation():
    """测试适应性"""
    print("\n" + "="*60)
    print("Test: Adaptation to Environment Change")
    print("="*60)
    
    np.random.seed(42)
    
    system = PredictiveFCRS(10, 10, 5)
    errors = []
    
    # Phase 1: 5类
    env = {i: np.random.randn(10)*2 for i in range(5)}
    for _ in range(500):
        c = env[np.random.randint(0, 5)]
        x = c + np.random.randn(10)*0.3
        errors.append(system.update(x))
    
    p1_err = np.mean(errors)
    print(f"Phase 1 (5 classes): error={p1_err:.3f}")
    
    # Phase 2: 10类 - 更复杂
    env = {i: np.random.randn(10)*2 for i in range(10)}
    for _ in range(500):
        c = env[np.random.randint(0, 10)]
        x = c + np.random.randn(10)*0.3
        errors.append(system.update(x))
    
    p2_err = np.mean(errors[-500:])
    print(f"Phase 2 (10 classes): error={p2_err:.3f}")
    
    # Phase 3: 回到5类
    env = {i: np.random.randn(10)*2 for i in range(5)}
    for _ in range(500):
        c = env[np.random.randint(0, 5)]
        x = c + np.random.randn(10)*0.3
        errors.append(system.update(x))
    
    p3_err = np.mean(errors[-500:])
    print(f"Phase 3 (5 classes): error={p3_err:.3f}")
    
    print(f"\nAdaptation: P1={p1_err:.3f}, P2={p2_err:.3f}, P3={p3_err:.3f}")


# ==================== Main ====================
print("FCRS-v5.3: Compression-Prediction Driven\n")

test_prediction_driven()
compare_with_random()
test_adaptation()

print("\n" + "="*60)
print("Core Loop:")
print("  Observe → Compress → Predict → Select → Feedback")
print("="*60)
