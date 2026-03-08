# FCRS-v5.1 系统A: 固定容量

"""
系统A: 固定容量baseline
latent_dim = 8 (固定)
无spawn
无prune
"""

import numpy as np


class SystemA:
    """固定容量系统"""
    
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        
        # 编码器
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        
        # 解码器
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # 指标记录
        self.history = {
            'dim': [],
            'loss': [],
            'entropy': []
        }
    
    def encode(self, x):
        """编码: 9 -> latent_dim"""
        h = np.tanh(np.dot(x, self.encoder_w))
        return h
    
    def decode(self, h):
        """解码: latent_dim -> 9"""
        return np.tanh(np.dot(h, self.decoder_w))
    
    def forward(self, x):
        """前向"""
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    def train(self, x, lr=0.01):
        """训练一步"""
        # 前向
        x_hat, h = self.forward(x)
        
        # 重建损失
        loss = np.mean((x - x_hat) ** 2)
        
        # 反向(简化)
        grad_out = (x_hat - x) / 9
        
        # 解码器梯度
        grad_decoder = np.outer(h, grad_out)
        self.decoder_w -= lr * grad_decoder
        
        # 编码器梯度
        grad_hidden = np.dot(grad_out, self.decoder_w.T)
        grad_encoder = np.outer(x, grad_hidden)
        self.encoder_w -= lr * grad_encoder
        
        return loss, h
    
    def get_active_dims(self, h, threshold=0.1):
        """活跃维度"""
        return np.sum(np.abs(h) > threshold)
    
    def get_entropy(self, h):
        """维度利用率"""
        p = np.abs(h) / (np.sum(np.abs(h)) + 1e-8)
        return -np.sum(p * np.log(p + 1e-8))
    
    def run(self, env, steps=1000):
        """运行环境"""
        for step in range(steps):
            # 获取观察
            obs = env.get_obs()
            
            # 训练
            loss, h = self.train(obs)
            
            # 记录
            if step % 100 == 0:
                active_dims = self.get_active_dims(h)
                entropy = self.get_entropy(h)
                
                self.history['dim'].append(active_dims)
                self.history['loss'].append(loss)
                self.history['entropy'].append(entropy)
                
                print(f"Step {step}: dim={self.latent_dim}, loss={loss:.4f}, entropy={entropy:.4f}")
        
        return self.history


class SimpleEnv:
    """简单环境"""
    
    def __init__(self, complexity='E1'):
        self.complexity = complexity
        self.grid_size = 8
        
        # 根据复杂度设置物体
        if complexity == 'E1':  # 单物体
            self.n_objects = 1
        elif complexity == 'E2':  # 多物体
            self.n_objects = 3
        elif complexity == 'E3':  # 遮挡
            self.n_objects = 5
        elif complexity == 'E4':  # 随机
            self.n_objects = 5
        
        self.objects = self._init_objects()
    
    def _init_objects(self):
        """初始化物体"""
        import random
        return [(random.randint(0, 7), random.randint(0, 7)) for _ in range(self.n_objects)]
    
    def get_obs(self):
        """获取观察(9维)"""
        import random
        
        # 简单: 随机生成观察
        if self.complexity == 'E1':
            return np.random.randn(9) * 0.5
        elif self.complexity == 'E2':
            return np.random.randn(9) * 0.7
        elif self.complexity == 'E3':
            return np.random.randn(9) * 0.9
        elif self.complexity == 'E4':
            return np.random.randn(9) * (0.5 + random.random() * 0.5)


def run_system_A():
    """运行系统A"""
    print("="*50)
    print("系统A: 固定容量 (dim=8)")
    print("="*50)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        print(f"\n--- {env_level} ---")
        
        # 创建系统
        sys_a = SystemA(latent_dim=8)
        
        # 创建环境
        env = SimpleEnv(complexity=env_level)
        
        # 运行
        history = sys_a.run(env, steps=500)
        
        # 记录
        results[env_level] = {
            'final_loss': np.mean(history['loss'][-5:]),
            'avg_dim': np.mean(history['dim'][-5:]),
            'avg_entropy': np.mean(history['entropy'][-5:])
        }
    
    # 打印结果
    print("\n" + "="*50)
    print("系统A结果汇总")
    print("="*50)
    
    for env, metrics in results.items():
        print(f"{env}: loss={metrics['final_loss']:.4f}, dim={metrics['avg_dim']}, entropy={metrics['avg_entropy']:.4f}")
    
    return results


if __name__ == "__main__":
    run_system_A()
