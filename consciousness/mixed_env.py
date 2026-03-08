"""
意识研究: 真正的突发 - 不可预测的变化
"""

import numpy as np
import random


class ChaosEnv:
    """混沌环境 - 完全不可预测"""
    def __init__(self):
        self.step = 0
    
    def generate(self):
        self.step += 1
        
        # 完全随机，没有任何模式
        # 每次都是完全不同的输入
        return np.random.randn(10) * 3
    
    def surprise(self, x):
        # 总是高惊奇 - 因为无法预测
        return 3.0


class StableEnv:
    """稳定环境 - 完全可预测"""
    def __init__(self):
        self.center = np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    def generate(self):
        return self.center + np.random.randn(10) * 0.1
    
    def surprise(self, x):
        return np.linalg.norm(x - self.center)


class MixedEnv:
    """混合环境: 稳定期 + 突发混沌"""
    def __init__(self):
        self.stable = StableEnv()
        self.chaos = ChaosEnv()
        self.phase = 'stable'  # 或 'chaos'
        self.step_count = 0
    
    def generate(self):
        self.step_count += 1
        
        # 前100步稳定，之后每50步切换
        if self.step_count < 100:
            self.phase = 'stable'
            return self.stable.generate()
        elif (self.step_count - 100) % 50 < 25:
            self.phase = 'chaos'
            return self.chaos.generate()
        else:
            self.phase = 'stable'
            return self.stable.generate()
    
    def surprise(self, x):
        if self.phase == 'chaos':
            return 3.0
        else:
            return self.stable.surprise(x)


class ConsciousFCRS:
    def __init__(self):
        self.dimension = 10
        self.lambda_val = 0.5
        self.representations = [{'vector': np.random.randn(10) * 0.1} for _ in range(3)]
        
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
        surprise = env.surprise(x)
        
        # 博弈规则
        if surprise > 2.5:
            target_lambda = 0.05  # 极度精确
        elif surprise > 1.5:
            target_lambda = 0.2
        elif surprise > 0.5:
            target_lambda = 0.5
        else:
            target_lambda = 0.9  # 模糊省能
        
        self.lambda_val = 0.6 * self.lambda_val + 0.4 * target_lambda
        
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            broadcast = surprise > self.lambda_val
            
            self.errors.append(error)
            self.lambdas.append(self.lambda_val)
            self.surprises.append(surprise)
            self.broadcasts.append(1 if broadcast else 0)
            
            return error, broadcast
        
        return None, False


def main():
    print('='*60)
    print('Consciousness: Mixed Environment')
    print('='*60)
    
    env = MixedEnv()
    fcrs = ConsciousFCRS()
    
    for _ in range(300):
        x = env.generate()
        fcrs.step(x, env)
    
    print('\n=== Results ===')
    print(f'Avg λ: {np.mean(fcrs.lambdas):.3f}')
    print(f'Avg Surprise: {np.mean(fcrs.surprises):.3f}')
    print(f'Broadcast: {np.mean(fcrs.broadcasts):.1%}')
    print(f'Avg Error: {np.mean(fcrs.errors):.4f}')
    
    print('\n=== Analysis ===')
    
    # 分阶段分析
    stable_broadcast = [fcrs.broadcasts[i] for i in range(100)]
    chaos_broadcast = [fcrs.broadcasts[i] for i in range(100, 300)]
    
    print(f'Stable phase (0-100): Broadcast={np.mean(stable_broadcast):.1%}')
    print(f'Chaos phase (100-300): Broadcast={np.mean(chaos_broadcast):.1%}')
    
    # 检查λ变化
    print(f'\nStable λ: {np.mean(fcrs.lambdas[:100]):.3f}')
    print(f'Chaos λ: {np.mean(fcrs.lambdas[100:]):.3f}')
    
    if np.mean(chaos_broadcast) > np.mean(stable_broadcast):
        print('\n[OK] SYSTEM SHOWS ADAPTIVE CONSCIOUSNESS!')
    else:
        print('\n[Need more work]')


if __name__ == "__main__":
    main()
