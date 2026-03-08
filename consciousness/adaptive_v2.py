"""
意识研究 Phase 1: 真正的自适应博弈
变化的环境 + 真正的博弈
"""

import numpy as np
import random


class DynamicEnv:
    """动态变化的环境"""
    def __init__(self):
        self.centers = [np.random.randn(10) * 3 for _ in range(3)]
        self.phase = 0
    
    def generate(self):
        # 每50步完全改变环境
        self.phase += 1
        
        if self.phase % 50 == 0:
            # 突然改变！
            self.centers = [np.random.randn(10) * 3 for _ in range(3)]
        
        cls = np.random.randint(0, 3)
        return self.centers[cls] + np.random.randn(10) * 0.1
    
    def surprise(self, x):
        dists = [np.linalg.norm(x - c) for c in self.centers]
        return min(dists)


class ConsciousSystem:
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
        self.awareness = 0.5
    
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
        
        # === 博弈规则 ===
        # 惊奇高 → 需要精确(低λ)
        # 惊奇低 → 可以模糊(高λ)
        
        # 但能量有限！
        
        if surprise > 2.5:
            # 重大变化 - 必须精确
            target_lambda = 0.1
        elif surprise > 1.5:
            # 变化 - 适度精确
            target_lambda = 0.3
        elif surprise > 0.8:
            # 小变化
            target_lambda = 0.5
        else:
            # 无变化 - 模糊省能
            target_lambda = 0.9
        
        # 如果能量低，更倾向模糊
        if self.energy < 30:
            target_lambda = min(0.9, target_lambda + 0.3)
        
        # 渐进调整
        self.lambda_val = 0.7 * self.lambda_val + 0.3 * target_lambda
        
        # 选择
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            # 广播
            broadcast = surprise > self.lambda_val
            
            if broadcast:
                self.energy = max(0, self.energy - 2)
                self.awareness = min(1.0, self.awareness + 0.1)
            else:
                self.energy = min(100, self.energy + 0.5)
                self.awareness = max(0.1, self.awareness - 0.02)
            
            self.errors.append(error)
            self.lambdas.append(self.lambda_val)
            self.surprises.append(surprise)
            self.broadcasts.append(1 if broadcast else 0)
            
            return error, broadcast
        
        return None, False


def main():
    print('='*60)
    print('Consciousness: True Adaptive Game')
    print('='*60)
    
    env = DynamicEnv()
    system = ConsciousSystem()
    
    # 运行
    for _ in range(1000):
        x = env.generate()
        system.step(x, env)
    
    # 结果
    print('\n=== Results ===')
    print(f'Avg λ: {np.mean(system.lambdas):.3f}')
    print(f'Avg Surprise: {np.mean(system.surprises):.3f}')
    print(f'Broadcast: {np.mean(system.broadcasts):.1%}')
    print(f'Avg Error: {np.mean(system.errors):.4f}')
    print(f'Energy: {system.energy:.1f}')
    print(f'Awareness: {system.awareness:.2f}')
    
    # 分析
    print('\n=== Analysis ===')
    
    # 找出phase变化时刻
    broadcasts = system.broadcasts
    surprises = system.surprises
    
    # 高surprise时期
    high_s = [broadcasts[i] for i in range(len(surprises)) if surprises[i] > 2.0]
    med_s = [broadcasts[i] for i in range(len(surprises)) if 1.0 < surprises[i] <= 2.0]
    low_s = [broadcasts[i] for i in range(len(surprises)) if surprises[i] <= 1.0]
    
    print(f'High surprise → Broadcast: {np.mean(high_s):.1%}' if high_s else 'No high surprise')
    print(f'Med surprise → Broadcast: {np.mean(med_s):.1%}' if med_s else 'No med surprise')
    print(f'Low surprise → Broadcast: {np.mean(low_s):.1%}' if low_s else 'No low surprise')
    
    # 判断
    if high_s and np.mean(high_s) > 0.3:
        print('\n[OK] SYSTEM SHOWS ADAPTIVE CONSCIOUSNESS!')
    else:
        print('\n[WARN] Need more tuning')


if __name__ == "__main__":
    main()
