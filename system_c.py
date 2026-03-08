# FCRS-v5.1 系统C: ECS生态

"""
系统C: 扩张+竞争+淘汰(ECS)
预期: dim→8→12→16→17(稳定)
"""

import numpy as np


class SystemC:
    """ECS生态系统"""
    
    def __init__(self, latent_dim=8, max_dim=32):
        self.latent_dim = latent_dim
        self.max_dim = max_dim
        
        # 权重
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # 维度重要性分数
        self.importance = np.ones(latent_dim)
        
        # 记录
        self.history = {'dim': [], 'loss': [], 'spawns': 0, 'prunes': 0}
    
    def expand(self):
        """扩张"""
        if self.latent_dim >= self.max_dim:
            return False
        
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        
        self.importance = np.append(self.importance, 1.0)
        
        self.latent_dim += 1
        self.history['spawns'] += 1
        
        return True
    
    def compete(self, h):
        """竞争: Top-K gating"""
        k = max(2, self.latent_dim // 4)  # 25%激活
        activation = np.abs(h)
        
        # 记录重要性
        self.importance = 0.9 * self.importance + 0.1 * activation
    
    def prune(self):
        """淘汰: 低重要性维度"""
        if self.latent_dim <= 4:  # 最小4维
            return False
        
        # 找到最低重要性维度
        min_idx = np.argmin(self.importance)
        
        if self.importance[min_idx] < 0.1:  # 阈值
            # 删除该维度
            mask = np.ones(self.latent_dim, dtype=bool)
            mask[min_idx] = False
            
            self.encoder_w = self.encoder_w[:, mask]
            self.decoder_w = self.decoder_w[mask, :]
            self.importance = self.importance[mask]
            
            self.latent_dim -= 1
            self.history['prunes'] += 1
            
            return True
        
        return False
    
    def encode(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        return h
    
    def decode(self, h):
        return np.tanh(np.dot(h, self.decoder_w))
    
    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
    
    def train(self, x):
        x_hat, h = self.forward(x)
        loss = np.mean((x - x_hat) ** 2)
        
        # 竞争
        self.compete(h)
        
        # 扩张: loss高且维度饱和
        if loss > 0.3 and self.latent_dim < self.max_dim:
            if np.random.random() < 0.05:
                self.expand()
        
        # 淘汰: 每50步检查
        if np.random.random() < 0.02:
            self.prune()
        
        # 简化训练
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
                print(f"Step {step}: dim={self.latent_dim}, loss={loss:.4f}, spawns={self.history['spawns']}, prunes={self.history['prunes']}")
        
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


def run_system_C():
    print("="*50)
    print("系统C: ECS生态")
    print("="*50)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        print(f"\n--- {env_level} ---")
        
        sys_c = SystemC(latent_dim=8, max_dim=32)
        env = SimpleEnv(complexity=env_level)
        
        history = sys_c.run(env, steps=500)
        
        results[env_level] = {
            'final_dim': sys_c.latent_dim,
            'spawns': history['spawns'],
            'prunes': history['prunes'],
            'avg_loss': np.mean(history['loss'][-5:])
        }
    
    print("\n" + "="*50)
    print("系统C结果汇总")
    print("="*50)
    
    for env, metrics in results.items():
        print(f"{env}: dim={metrics['final_dim']}, spawns={metrics['spawns']}, prunes={metrics['prunes']}, loss={metrics['avg_loss']:.4f}")
    
    return results


if __name__ == "__main__":
    run_system_C()
