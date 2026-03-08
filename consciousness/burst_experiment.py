"""
意识研究: 突发变化环境
让系统经历: 稳定期 → 突发变化 → 适应
"""

import numpy as np
import random


class BurstEnv:
    """突发变化的环境"""
    def __init__(self):
        # 初始中心
        self.centers = {
            0: np.array([3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1: np.array([0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
            2: np.array([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0]),
        }
        self.step_count = 0
    
    def generate(self):
        self.step_count += 1
        
        # 每100步突发改变
        if self.step_count % 100 == 0:
            # 突然移到相反方向!
            self.centers = {
                0: np.array([-3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                1: np.array([0, -3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
                2: np.array([0, 0, -3.0, 0, 0, 0, 0, 0, 0, 0]),
            }
        
        cls = np.random.randint(0, 3)
        return self.centers[cls] + np.random.randn(10) * 0.1
    
    def surprise(self, x):
        # 计算与中心的距离
        dists = [np.linalg.norm(x - c) for c in self.centers.values()]
        return min(dists)


class ConsciousFCRS:
    """有意识的系统"""
    
    def __init__(self):
        self.dimension = 10
        self.lambda_val = 0.5
        self.representations = [{'vector': np.random.randn(10) * 0.1} for _ in range(3)]
        
        self.errors = []
        self.lambdas = []
        self.surprises = []
        self.broadcasts = []
        self.energy = 100
    
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
        
        # 博弈
        if surprise > 2.0:
            target_lambda = 0.1  # 必须精确
        elif surprise > 1.0:
            target_lambda = 0.3
        else:
            target_lambda = 0.8  # 可以模糊
        
        self.lambda_val = 0.7 * self.lambda_val + 0.3 * target_lambda
        
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            # 广播
            broadcast = surprise > self.lambda_val
            
            if broadcast:
                self.energy -= 2
            else:
                self.energy = min(100, self.energy + 0.5)
            
            self.errors.append(error)
            self.lambdas.append(self.lambda_val)
            self.surprises.append(surprise)
            self.broadcasts.append(1 if broadcast else 0)
            
            return error, broadcast
        
        return None, False


def main():
    print('='*60)
    print('Consciousness: Burst Change Experiment')
    print('='*60)
    
    env = BurstEnv()
    fcrs = ConsciousFCRS()
    
    # 运行
    for _ in range(500):
        x = env.generate()
        fcrs.step(x, env)
    
    # 结果
    print('\n=== Results ===')
    print(f'Avg λ: {np.mean(fcrs.lambdas):.3f}')
    print(f'Avg Surprise: {np.mean(fcrs.surprises):.3f}')
    print(f'Broadcast: {np.mean(fcrs.broadcasts):.1%}')
    print(f'Avg Error: {np.mean(fcrs.errors):.4f}')
    
    # 分析
    print('\n=== Analysis ===')
    
    # 检查突发变化时刻
    for i in range(100, 500, 100):
        print(f'\nStep {i}:')
        print(f'  Surprise: {fcrs.surprises[i]:.3f}')
        print(f'  λ: {fcrs.lambdas[i]:.3f}')
        print(f'  Broadcast: {fcrs.broadcasts[i]}')
    
    # 整体
    high_s = [fcrs.broadcasts[i] for i in range(len(fcrs.surprises)) if fcrs.surprises[i] > 2.0]
    print(f'\nHigh surprise events: {len(high_s)}')
    print(f'High surprise broadcast rate: {np.mean(high_s) if high_s else 0:.1%}')


if __name__ == "__main__":
    main()
