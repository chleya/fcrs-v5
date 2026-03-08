# FCRS-v5 Experiment 4.3: 截断类型对比

"""
Experiment 4.3: 截断类型 vs 效果

截断类型:
- 无截断: 无限扩张
- 容量截断: max_dims=B
- 信息截断: top-k
- 能量截断: 阈值门限
- 组合截断: 容量+信息+能量
"""

import numpy as np


class System_NoTruncation:
    """无截断系统"""
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.history = {'dim': []}
    
    def expand(self):
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.latent_dim += 1
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        if loss > 0.3 and np.random.random() < 0.1:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


class System_CapacityTruncation:
    """容量截断"""
    def __init__(self, latent_dim=8, max_budget=16):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.history = {'dim': []}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
            return False
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.latent_dim += 1
        return True
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        if loss > 0.3 and np.random.random() < 0.1:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


class System_InfoTruncation:
    """信息截断 (top-k)"""
    def __init__(self, latent_dim=8, max_budget=16, k=2):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.k = k
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.importance = np.ones(latent_dim)
        self.history = {'dim': []}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
            return False
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.importance = np.append(self.importance, 1.0)
        self.latent_dim += 1
        return True
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        # Top-k competition
        k = min(self.k, self.latent_dim)
        top_k_idx = np.argsort(np.abs(h))[-k:]
        new_imp = np.zeros_like(self.importance)
        new_imp[top_k_idx] = np.abs(h[top_k_idx])
        self.importance = 0.9 * self.importance + 0.1 * new_imp
        
        if loss > 0.3 and np.random.random() < 0.1:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


class System_EnergyTruncation:
    """能量截断 (阈值门限)"""
    def __init__(self, latent_dim=8, max_budget=16, threshold=0.3):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.threshold = threshold
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.history = {'dim': []}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
            return False
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.latent_dim += 1
        return True
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        
        # 能量截断: 只传递超过阈值的信号
        h_truncated = h * (np.abs(h) > self.threshold)
        
        x_hat = np.tanh(np.dot(h_truncated, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        if loss > 0.3 and np.random.random() < 0.1:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


class System_CombinedTruncation:
    """组合截断 (容量+信息+能量)"""
    def __init__(self, latent_dim=8, max_budget=16, k=2, threshold=0.3):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.k = k
        self.threshold = threshold
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.importance = np.ones(latent_dim)
        self.history = {'dim': []}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
            return False
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.importance = np.append(self.importance, 1.0)
        self.latent_dim += 1
        return True
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        
        # 1. 能量截断
        h = h * (np.abs(h) > self.threshold)
        
        # 2. 信息截断 (top-k)
        k = min(self.k, self.latent_dim)
        top_k_idx = np.argsort(np.abs(h))[-k:]
        new_imp = np.zeros_like(self.importance)
        new_imp[top_k_idx] = np.abs(h[top_k_idx])
        self.importance = 0.9 * self.importance + 0.1 * new_imp
        
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        # 3. 容量截断在expand中
        
        if loss > 0.3 and np.random.random() < 0.1:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


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


def run_system(SystemClass, **kwargs):
    np.random.seed(0)
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        sys = SystemClass(**kwargs)
        env = SimpleEnv(complexity=env_level)
        
        for step in range(500):
            obs = env.get_obs()
            sys.train(obs)
        
        results[env_level] = {
            'final_dim': sys.latent_dim,
            'avg_dim': np.mean(sys.history['dim']) if sys.history['dim'] else sys.latent_dim
        }
    
    return results


def main():
    print("="*60)
    print("Experiment 4.3: 截断类型对比")
    print("="*60)
    
    systems = [
        ("无截断", System_NoTruncation, {}),
        ("容量截断", System_CapacityTruncation, {'max_budget': 16}),
        ("信息截断", System_InfoTruncation, {'max_budget': 16, 'k': 2}),
        ("能量截断", System_EnergyTruncation, {'max_budget': 16, 'threshold': 0.3}),
        ("组合截断", System_CombinedTruncation, {'max_budget': 16, 'k': 2, 'threshold': 0.3}),
    ]
    
    all_results = {}
    
    for name, SystemClass, kwargs in systems:
        print(f"\n--- {name} ---")
        results = run_system(SystemClass, latent_dim=8, **kwargs)
        all_results[name] = results
        
        dims = [results[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        print(f"  Dims: {dims}, Avg: {np.mean(dims):.1f}")
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n截断类型 | E1 | E2 | E3 | E4 | AvgDim")
    print("-"*50)
    
    for name, results in all_results.items():
        dims = [results[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        print(f"{name:^10} | {dims[0]:>2} | {dims[1]:>2} | {dims[2]:>2} | {dims[3]:>2} | {np.mean(dims):>5.1f}")
    
    return all_results


if __name__ == "__main__":
    main()
