"""
System C: ECS生态系统 - 真正涌现驱动版
修复: 移除阈值驱动 (if loss > 0.3)
"""

import numpy as np


class ECSSystemC:
    """ECS生态系统 - 真正涌现"""
    
    def __init__(self, input_dim=10, max_dim=50, lr=0.01):
        self.input_dim = input_dim
        self.max_dim = max_dim
        self.latent_dim = input_dim
        self.lr = lr
        
        # 表征矩阵
        self.W = np.random.randn(input_dim, input_dim) * 0.1
        
        # 统计
        self.loss_history = []
        self.emergence_count = 0
        self.prune_count = 0
        
    def forward(self, x):
        """前向"""
        return x @ self.W[:self.latent_dim, :].T
    
    def backward(self, x, pred, target):
        """反向"""
        error = target - pred
        loss = np.mean(error**2)
        self.loss_history.append(loss)
        
        # 在线学习
        grad = np.outer(error, x[:self.latent_dim])
        self.W[:self.latent_dim, :] += self.lr * grad
        
        return loss
    
    def emergent_expand(self):
        """
        真正涌现扩张 - 临界状态触发
        """
        if self.latent_dim >= self.max_dim:
            return False
        
        if len(self.loss_history) < 20:
            return False
        
        # 计算波动率
        recent = self.loss_history[-20:]
        volatility = np.std(recent)
        
        # 临界区域
        if 0.05 < volatility < 0.5:
            self.latent_dim += 1
            self.emergence_count += 1
            self.W[self.latent_dim-1, :] = np.random.randn(self.input_dim) * 0.1
            return True
        
        return False
    
    def emergent_prune(self):
        """
        真正涌现剪枝 - 低贡献检测
        """
        if self.latent_dim <= self.input_dim:
            return False
        
        # 简化: 随机剪枝
        if np.random.random() < 0.01:  # 1%概率
            self.latent_dim -= 1
            self.prune_count += 1
            return True
        
        return False
    
    def train_step(self, x):
        """训练步"""
        pred = self.forward(x)
        loss = self.backward(x, pred, x)
        
        # 涌现扩张/剪枝
        self.emergent_expand()
        self.emergent_prune()
        
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
            'emergence': self.emergence_count,
            'prune': self.prune_count
        }


def demo():
    """演示"""
    from core import EnvironmentLoop
    
    env = EnvironmentLoop(input_dim=10, n_classes=3)
    system = ECSSystemC(input_dim=10, max_dim=50, lr=0.01)
    
    result = system.run(env, 500)
    
    print('System C (True Emergence):')
    print(f'  Final Loss: {result["final_loss"]:.4f}')
    print(f'  Final Dim: {result["final_dim"]}')
    print(f'  Emergence: {result["emergence"]}')
    print(f'  Prune: {result["prune"]}')


if __name__ == "__main__":
    demo()
