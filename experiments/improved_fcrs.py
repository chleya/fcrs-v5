"""
改进版FCRS
问题分析：
1. 选择机制不够有效
2. 学习率太低
3. 没有真正利用多表征
"""

import numpy as np


class ImprovedFCRS:
    """改进的FCRS"""
    
    def __init__(self, lambda_val=0.5):
        self.lambda_val = lambda_val
        self.dimension = 10
        
        # 初始化多个表征
        self.representations = []
        for _ in range(5):  # 5个表征
            self.representations.append({
                'vector': np.random.randn(50) * 0.1,  # 更大的向量
                'count': 0,
                'total_error': 0
            })
    
    def select(self, x):
        """改进的选择机制"""
        best = None
        best_score = -float('inf')
        
        for i, rep in enumerate(self.representations):
            v = rep['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            
            # Cosine similarity
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
    
    def learn(self, x, best):
        """改进的学习"""
        if best is None:
            return
        
        v = self.representations[best]['vector'][:self.dimension]
        x_sub = x[:self.dimension]
        
        # 计算误差
        error = np.linalg.norm(x_sub - v)
        
        # 更新统计
        self.representations[best]['count'] += 1
        self.representations[best]['total_error'] += error
        
        # 学习 - 使用更高的学习率
        learning_rate = 0.5  # 提高！
        self.representations[best]['vector'][:self.dimension] += learning_rate * (x_sub - v)
    
    def step(self, x):
        """一步"""
        # 选择
        best = self.select(x)
        
        if best is not None:
            # 学习
            self.learn(x, best)
            
            # 计算误差
            v = self.representations[best]['vector'][:self.dimension]
            x_sub = x[:self.dimension]
            error = np.linalg.norm(x_sub - v)
            
            return error
        
        return None


class Environment:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.centers = [np.random.randn(50) * 2 for _ in range(n_classes)]
    
    def generate(self):
        cls = np.random.randint(0, self.n_classes)
        return self.centers[cls] + np.random.randn(50) * 0.1, cls


def test_improved():
    """测试改进版"""
    print('='*60)
    print('Improved FCRS Test')
    print('='*60)
    
    env = Environment(n_classes=3)
    fcrs = ImprovedFCRS(lambda_val=0.5)
    
    # 训练
    for step in range(1000):
        x, cls = env.generate()
        fcrs.step(x)
        
        if step % 200 == 0:
            # 测试误差
            test_errors = []
            for _ in range(50):
                x, cls = env.generate()
                err = fcrs.step(x)
                if err:
                    test_errors.append(err)
            
            if test_errors:
                print(f'Step {step}: error={np.mean(test_errors):.4f}')
    
    # 最终测试
    final_errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = fcrs.step(x)
        if err:
            final_errors.append(err)
    
    print(f'\nFinal error: {np.mean(final_errors):.4f}')
    
    # 检查表征使用
    print('\nRepresentation usage:')
    for i, rep in enumerate(fcrs.representations):
        print(f'  Rep {i}: count={rep["count"]}, avg_error={rep["total_error"]/max(1,rep["count"]):.4f}')


def compare():
    """对比测试"""
    print('\n' + '='*60)
    print('Comparison: Improved vs Original')
    print('='*60)
    
    env = Environment(n_classes=3)
    
    # 改进版
    improved = ImprovedFCRS(lambda_val=0.5)
    
    for _ in range(1000):
        x, cls = env.generate()
        improved.step(x)
    
    imp_errors = []
    for _ in range(100):
        x, cls = env.generate()
        err = improved.step(x)
        if err:
            imp_errors.append(err)
    
    print(f'Improved FCRS: {np.mean(imp_errors):.4f}')


if __name__ == "__main__":
    test_improved()
    compare()
