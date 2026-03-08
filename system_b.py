# FCRS-v5.1 系统B: 纯扩张

"""
系统B: 纯扩张(spawn only)
dim: 8 → 16 → 32 → 64
无prune
预期: 过拟合
"""

import numpy as np


class SystemB:
    """纯扩张系统"""
    
    def __init__(self, latent_dim=8, max_dim=32):
        self.latent_dim = latent_dim
        self.max_dim = max_dim
        
        # 编码器
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        # 解码器
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # 记录
        self.history = {'dim': [], 'loss': [], 'spawns': 0}
    
    def expand(self):
        """扩张维度"""
        if self.latent_dim >= self.max_dim:
            return False
        
        # 添加新列
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        
        self.latent_dim += 1
        self.history['spawns'] += 1
        
        return True
    
    def encode(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        return h
    
    def decode(self, h):
        return np.tanh(np.dot(h, self.decoder_w))
    
    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    def train(self, x, lr=0.01):
        x_hat, h = self.forward(x)
        
        # 重建损失
        loss = np.mean((x - x_hat) ** 2)
        
        # 检查是否需要扩张
        if loss > 0.3 and self.latent_dim < self.max_dim:
            if np.random.random() < 0.05:
                self.expand()
        
        # 简化训练(不反向传播,只更新现有权重)
        # 直接用随机扰动
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss, h
    
    def run(self, env, steps=500):
        for step in range(steps):
            obs = env.get_obs()
            loss, h = self.train(obs)
            
            if step % 100 == 0:
                active = np.sum(np.abs(h) > 0.1)
                self.history['dim'].append(self.latent_dim)
                self.history['loss'].append(loss)
                print(f"Step {step}: dim={self.latent_dim}, loss={loss:.4f}, active={active}")
        
        return self.history


class SimpleEnv:
    def __init__(self, complexity='E1'):
        self.complexity = complexity
        
    def get_obs(self):
        import random
        if self.complexity == 'E1':
            return np.random.randn(9) * 0.5
        elif self.complexity == 'E2':
            return np.random.randn(9) * 0.7
        elif self.complexity == 'E3':
            return np.random.randn(9) * 0.9
        elif self.complexity == 'E4':
            return np.random.randn(9) * (0.5 + random.random() * 0.5)


def run_system_B():
    print("="*50)
    print("系统B: 纯扩张 (spawn only)")
    print("="*50)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        print(f"\n--- {env_level} ---")
        
        sys_b = SystemB(latent_dim=8, max_dim=32)
        env = SimpleEnv(complexity=env_level)
        
        history = sys_b.run(env, steps=500)
        
        results[env_level] = {
            'final_dim': sys_b.latent_dim,
            'spawns': history['spawns'],
            'avg_loss': np.mean(history['loss'][-5:])
        }
    
    print("\n" + "="*50)
    print("系统B结果汇总")
    print("="*50)
    
    for env, metrics in results.items():
        print(f"{env}: dim={metrics['final_dim']}, spawns={metrics['spawns']}, loss={metrics['avg_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    run_system_B()
