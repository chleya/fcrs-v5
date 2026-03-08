"""
自适应截断策略
根据任务复杂度动态调整截断强度
"""

import numpy as np


class AdaptiveTruncationFCRS:
    """自适应截断FCRS"""
    
    def __init__(self):
        # 表征
        self.representations = []
        for _ in range(5):
            self.representations.append({
                'vector': np.random.randn(50) * 0.1,
                'count': 0,
                'error_history': []
            })
        
        # 截断参数
        self.truncation_rate = 0.5  # 初始截断率
        self.target_complexity = None
        self.current_complexity = 0
        
        # 历史
        self.history = {
            'complexity': [],
            'truncation': [],
            'error': []
        }
    
    def estimate_complexity(self, x):
        """估计环境复杂度"""
        # 简单方法: 输入方差
        return np.var(x)
    
    def adapt_truncation(self, error):
        """根据误差自适应调整截断"""
        # 误差大 → 增加截断（更激进）
        # 误差小 → 减少截断（更保守）
        
        if error > 1.0:
            self.truncation_rate *= 1.1  # 增加截断
        elif error < 0.3:
            self.truncation_rate *= 0.95  # 减少截断
        
        # 限制范围
        self.truncation_rate = max(0.1, min(0.9, self.truncation_rate))
    
    def truncate(self, vector):
        """执行截断"""
        dim = int(len(vector) * (1 - self.truncation_rate))
        return vector[:max(1, dim)]
    
    def select(self, x):
        """选择表征"""
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep['vector'][:20]  # 固定维度选择
            x_sub = x[:20]
            
            norm_v = np.linalg.norm(v)
            norm_x = np.linalg.norm(x_sub)
            
            if norm_v > 0.01 and norm_x > 0.01:
                score = np.dot(v, x_sub) / (norm_v * norm_x)
            else:
                score = -1
            
            if score > best_score:
                best_score = score
                best = i
        
        return best
    
    def step(self, x):
        """一步"""
        # 估计复杂度
        self.current_complexity = self.estimate_complexity(x)
        
        # 选择
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:20]
            x_sub = x[:20]
            
            error = np.linalg.norm(x_sub - v)
            
            # 学习
            self.representations[best]['vector'][:20] += 0.5 * (x_sub - v)
            
            # 记录历史
            self.representations[best]['error_history'].append(error)
            if len(self.representations[best]['error_history']) > 20:
                self.representations[best]['error_history'].pop(0)
            
            # 自适应截断
            avg_error = np.mean(self.representations[best]['error_history'])
            self.adapt_truncation(avg_error)
            
            # 记录
            self.history['complexity'].append(self.current_complexity)
            self.history['truncation'].append(self.truncation_rate)
            self.history['error'].append(error)
            
            return error
        
        return None


class Environment:
    """不同复杂度的环境"""
    def __init__(self, complexity=1.0):
        self.complexity = complexity
        self.centers = [np.random.randn(20) * complexity for _ in range(3)]
    
    def generate(self):
        cls = np.random.randint(0, 3)
        return self.centers[cls] + np.random.randn(20) * 0.1


def test_adaptive():
    """测试自适应截断"""
    print('='*60)
    print('Adaptive Truncation Test')
    print('='*60)
    
    # 测试不同复杂度
    for complexity in [0.5, 1.0, 2.0]:
        print(f'\n=== Complexity = {complexity} ===')
        
        env = Environment(complexity=complexity)
        fcrs = AdaptiveTruncationFCRS()
        
        for step in range(500):
            x = env.generate()
            fcrs.step(x)
        
        # 结果
        avg_truncation = np.mean(fcrs.history['truncation'][-100:])
        avg_error = np.mean(fcrs.history['error'][-100:])
        
        print(f'Avg truncation: {avg_truncation:.3f}')
        print(f'Avg error: {avg_error:.4f}')


def test_comparison():
    """对比自适应 vs 固定"""
    print('\n' + '='*60)
    print('Comparison: Adaptive vs Fixed')
    print('='*60)
    
    # 固定截断
    class FixedFCRS:
        def __init__(self, trunc_rate=0.5):
            self.trunc_rate = trunc_rate
            self.reps = [{'v': np.random.randn(20)*0.1} for _ in range(3)]
        
        def step(self, x):
            best = min(range(3), key=lambda i: np.linalg.norm(x[:20] - self.reps[i]['v']))
            self.reps[best]['v'] += 0.5 * (x[:20] - self.reps[best]['v'])
            return np.linalg.norm(x[:20] - self.reps[best]['v'])
    
    env = Environment(complexity=1.0)
    
    # 自适应
    adaptive = AdaptiveTruncationFCRS()
    for _ in range(500):
        adaptive.step(env.generate())
    
    # 固定
    fixed = FixedFCRS(trunc_rate=0.5)
    for _ in range(500):
        fixed.step(env.generate())
    
    print(f'Adaptive: {np.mean(adaptive.history["error"][-100:]):.4f}')
    print(f'Fixed:    {np.mean([fixed.step(env.generate()) for _ in range(100)]):.4f}')


if __name__ == "__main__":
    test_adaptive()
    test_comparison()
