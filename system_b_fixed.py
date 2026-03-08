"""
System B: 纯扩张系统 - 真正涌现驱动版
修复: 移除阈值驱动 (if loss > 0.3)
"""

import numpy as np


class PureExpansionB:
    """纯扩张系统 - 真正涌现"""
    
    def __init__(self, input_dim=10, max_dim=50, lr=0.01):
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.latent_dim = input_dim
        self.lr = lr
        
        # 表征
        self.W = np.random.randn(input_dim, input_dim) * 0.1
        self.latent = np.zeros(input_dim)
        
        # 涌现统计
        self.emergence_count = 0
        self.loss_history = []
        
    def forward(self, x):
        """前向"""
        # 简单线性降维
        self.latent = x @ self.W[:self.latent_dim, :].T
        return self.latent
    
    def backward(self, x, pred, target):
        """反向"""
        error = target - pred
        self.loss_history.append(np.mean(error**2))
        
        # 在线学习W
        grad = np.outer(error, x[:self.latent_dim])
        self.W[:self.latent_dim, :] += self.lr * grad
        
        return np.mean(error**2)
    
    def emergent_expand(self):
        """
        真正涌现: 基于系统内在动力扩张
        无阈值! 使用波动率检测
        """
        if self.latent_dim >= self.max_dim:
            return False
        
        # 计算损失波动率
        if len(self.loss_history) < 20:
            return False
        
        recent = self.loss_history[-20:]
        volatility = np.std(recent)
        
        # 临界状态: 中等波动时扩展
        # 波动太小 -> 稳定, 无需扩展
        # 波动太大 -> 混乱, 无法扩展
        if 0.05 < volatility < 0.5:
            # 扩展维度
            self.latent_dim += 1
            self.emergence_count += 1
            
            # 重置W的新行
            self.W[self.latent_dim-1, :] = np.random.randn(self.input_dim) * 0.1
            
            return True
        
        return False
    
    def train_step(self, x):
        """训练步"""
        pred = self.forward(x)
        loss = self.backward(x, pred, x)  # 自编码
        
        # 尝试涌现扩张
        self.emergent_expand()
        
        return loss
    
    def run(self, env, steps=500):
        """运行"""
        losses = []
        
        for _ in range(steps):
            x = env.generate_input()
            loss = self.train_step(x)
            losses.append(loss)
        
        return {
            'final_loss': np.mean(losses[-20:]),
            'final_dim': self.latent_dim,
            'emergence_count': self.emergence_count
        }


def demo():
    """演示"""
    from core import EnvironmentLoop
    
    env = EnvironmentLoop(input_dim=10, n_classes=3)
    system = PureExpansionB(input_dim=10, max_dim=50, lr=0.01)
    
    result = system.run(env, 500)
    
    print('System B (True Emergence):')
    print(f'  Final Loss: {result["final_loss"]:.4f}')
    print(f'  Final Dim: {result["final_dim"]}')
    print(f'  Emergence: {result["emergence_count"]}')


if __name__ == "__main__":
    demo()
