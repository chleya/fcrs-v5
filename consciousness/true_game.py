"""
意识研究 Phase 1: 真正的自适应博弈
让λ和surprise真正博弈
"""

import numpy as np
import random


class Env:
    """变化的环境"""
    def __init__(self, change_rate=0.1):
        self.change_rate = change_rate
        self.centers = [np.random.randn(10) * 3 for _ in range(3)]
        self.step_count = 0
    
    def generate(self):
        # 定期改变环境
        if self.step_count % 50 == 0:
            self.centers = [np.random.randn(10) * 3 for _ in range(3)]
        
        self.step_count += 1
        cls = np.random.randint(0, 3)
        return self.centers[cls] + np.random.randn(10) * 0.1
    
    def surprise(self, x):
        dists = [np.linalg.norm(x - c) for c in self.centers]
        return min(dists)


class ConsciousFCRS:
    """有意识的系统 - 真正的博弈"""
    
    def __init__(self):
        self.dimension = 10
        self.representations = [{'vector': np.random.randn(10) * 0.1} for _ in range(3)]
        
        # 历史
        self.errors = []
        self.lambdas = []
        self.surprises = []
        self.broadcasts = []
        
        # 博弈状态
        self.lambda_val = 0.5  # 初始λ
        self.energy_budget = 100  # 能量预算
        self.awareness = 0.5  # 意识水平
    
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
        
        # === 真正的博弈 ===
        # 1. 惊奇高时，降低λ获得更多细节
        # 2. 惊奇低时，提高λ节省能量
        
        if surprise > 2.0:
            # 重大变化 - 降低λ，精细观察
            target_lambda = 0.1
        elif surprise > 1.0:
            # 中等变化
            target_lambda = 0.4
        elif surprise > 0.5:
            # 小变化
            target_lambda = 0.7
        else:
            # 无变化 - 保持模糊节省能量
            target_lambda = 0.9
        
        # 2. 根据能量预算调整
        if self.energy_budget < 20:
            target_lambda = 0.9  # 能量低，更模糊
        
        # 渐进调整λ
        self.lambda_val = self.lambda_val * 0.8 + target_lambda * 0.2
        
        # 选择
        best = self.select(x)
        
        if best is not None:
            v = self.representations[best]['vector'][:self.dimension]
            error = np.linalg.norm(x[:self.dimension] - v)
            
            # 学习
            self.representations[best]['vector'][:self.dimension] += 0.5 * (x[:self.dimension] - v)
            
            # 广播决策 (GWT核心!)
            broadcast = surprise > self.lambda_val
            
            if broadcast:
                # 消耗能量
                self.energy_budget -= 1
                self.awareness = min(1.0, self.awareness + 0.1)
            else:
                # 恢复能量
                self.energy_budget = min(100, self.energy_budget + 0.5)
                self.awareness = max(0.1, self.awareness - 0.01)
            
            # 记录
            self.errors.append(error)
            self.lambdas.append(self.lambda_val)
            self.surprises.append(surprise)
            self.broadcasts.append(1 if broadcast else 0)
            
            return error, broadcast
        
        return None, False


def main():
    print('='*60)
    print('Consciousness Phase 1: True博弈')
    print('='*60)
    
    # 变化的环境
    env = Env(change_rate=0.1)
    fcrs = ConsciousFCRS()
    
    # 运行
    for _ in range(1000):
        x = env.generate()
        error, broadcast = fcrs.step(x, env)
    
    # 结果
    print('\n=== Results ===')
    print(f'Avg λ: {np.mean(fcrs.lambdas[-100:]):.3f}')
    print(f'Avg Surprise: {np.mean(fcrs.surprises[-100:]):.3f}')
    print(f'Broadcast Rate: {np.mean(fcrs.broadcasts[-100:]):.1%}')
    print(f'Avg Error: {np.mean(fcrs.errors[-100:]):.4f}')
    print(f'Energy: {fcrs.energy_budget:.1f}')
    print(f'Awareness: {fcrs.awareness:.2f}')
    
    # 分析
    print('\n=== Analysis ===')
    
    # 提取高surprise和低surprise时期
    high_surprise = [fcrs.broadcasts[i] for i in range(len(fcrs.surprises)) if fcrs.surprises[i] > 1.5]
    low_surprise = [fcrs.broadcasts[i] for i in range(len(fcrs.surprises)) if fcrs.surprises[i] < 0.5]
    
    if high_surprise:
        print(f'High surprise → Broadcast: {np.mean(high_surprise):.1%}')
    if low_surprise:
        print(f'Low surprise → Broadcast: {np.mean(low_surprise):.1%}')
    
    # 检查博弈
    if high_surprise and low_surprise:
        if np.mean(high_surprise) > np.mean(low_surprise):
            print('\n[OK] System shows ADAPTIVE broadcasting!')
            print('    Higher surprise → More broadcast')
        else:
            print('\n[WARN] No adaptive behavior')


if __name__ == "__main__":
    main()
