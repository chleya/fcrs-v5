"""
意识研究 Phase 1: 自适应瓶颈实验 (修复版)
让系统自己决定：要高λ(省力)还是低λ(精确)？
"""

import numpy as np
import random


class Env:
    def __init__(self, complexity=1.0):
        self.complexity = complexity
        # complexity高 = 中心更近 = 更难区分
        self.centers = [
            np.random.randn(10) * (3.0 / complexity) 
            for _ in range(3)
        ]
    
    def generate(self):
        cls = np.random.randint(0, 3)
        noise = np.random.randn(10) * 0.2
        return self.centers[cls] + noise
    
    def surprise(self, x):
        """惊奇度 = 距离最近中心的距离"""
        dists = [np.linalg.norm(x - c) for c in self.centers]
        return min(dists)


class ConsciousFCRS:
    """有意识的FCRS - 自我调节"""
    
    def __init__(self):
        # 初始状态
        self.lambda_val = 0.3  # 初始λ
        self.dimension = 10
        self.representations = [{'vector': np.random.randn(10) * 0.1} for _ in range(3)]
        
        # 历史
        self.errors = []
        self.lambdas = []
        self.surprises = []
        self.broadcasts = []
    
    def select(self, x):
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            
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
    
    def step(self, x, env):
        # 计算惊奇
        surprise = env.surprise(x)
        
        # 自适应λ: 惊奇高→降低λ(更精细), 惊奇低→提高λ(更模糊)
        if surprise > 1.5:
            self.lambda_val *= 0.9  # 降低
        elif surprise < 0.5:
            self.lambda_val *= 1.1  # 提高
        
        # 限制
        self.lambda_val = max(0.01, min(1.0, self.lambda_val))
        
        # 选择
        best = self.select(x)
        
        if best is not None:
            # 学习
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            # 广播决策 (GWT核心!)
            # 惊奇超过阈值时，表征进入"全局工作空间"
            broadcast = surprise > self.lambda_val
            
            # 记录
            self.errors.append(error)
            self.lambdas.append(self.lambda_val)
            self.surprises.append(surprise)
            self.broadcasts.append(1 if broadcast else 0)
            
            # 自适应维度
            if broadcast and self.dimension < 50:
                self.dimension += 1
            elif not broadcast and self.dimension > 10:
                self.dimension -= 1
            
            return error, broadcast
        
        return None, False


def main():
    print('='*60)
    print('Consciousness Phase 1: Adaptive Bottleneck')
    print('='*60)
    
    # 测试不同复杂度
    for complexity in [0.5, 1.0, 2.0, 5.0]:
        print(f'\n=== Complexity = {complexity} ===')
        
        env = Env(complexity=complexity)
        fcrs = ConsciousFCRS()
        
        # 运行
        for _ in range(500):
            x = env.generate()
            fcrs.step(x, env)
        
        # 结果
        avg_lambda = np.mean(fcrs.lambdas[-100:])
        avg_surprise = np.mean(fcrs.surprises[-100:])
        broadcast_rate = np.mean(fcrs.broadcasts[-100:])
        avg_error = np.mean(fcrs.errors[-100:])
        
        print(f'λ: {avg_lambda:.3f}')
        print(f'Surprise: {avg_surprise:.3f}')
        print(f'Broadcast: {broadcast_rate:.1%}')
        print(f'Error: {avg_error:.4f}')
        print(f'Dim: {fcrs.dimension}')
    
    # 验证GWT
    print('\n' + '='*60)
    print('GWT Validation')
    print('='*60)
    print('Theory: Higher surprise -> More broadcasting')


if __name__ == "__main__":
    main()
