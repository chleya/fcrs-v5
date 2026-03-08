# FCRS-v5 Experiment 4.1: 截断强度验证

"""
Experiment 4.1: Budget强度 vs 维度稳定性

变量: Budget B ∈ {8, 12, 16, 24, 32}
控制: k=2 (固定竞争)
"""

import numpy as np


class SystemC_Truncation:
    """截断机制系统"""
    
    def __init__(self, latent_dim=8, max_budget=16, k_active=2):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.k_active = k_active
        
        # 权重
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # 重要性
        self.importance = np.ones(latent_dim)
        
        # 记录
        self.history = {'dim': [], 'loss': [], 'spawns': 0, 'prunes': 0}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
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
        """信息截断: top-k"""
        k = min(self.k_active, self.latent_dim)
        top_k_indices = np.argsort(np.abs(h))[-k:]
        
        new_importance = np.zeros_like(self.importance)
        new_importance[top_k_indices] = np.abs(h[top_k_indices])
        
        self.importance = 0.9 * self.importance + 0.1 * new_importance
    
    def prune(self):
        if self.latent_dim <= 4:
            return False
        
        min_idx = np.argmin(self.importance)
        
        if self.importance[min_idx] < 0.05:  # 更严格阈值
            mask = np.ones(self.latent_dim, dtype=bool)
            mask[min_idx] = False
            
            self.encoder_w = self.encoder_w[:, mask]
            self.decoder_w = self.decoder_w[mask, :]
            self.importance = self.importance[mask]
            
            self.latent_dim -= 1
            self.history['prunes'] += 1
            return True
        return False
    
    def forward(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        return x_hat, h
    
    def train(self, x):
        x_hat, h = self.forward(x)
        loss = np.mean((x - x_hat) ** 2)
        
        # 信息截断
        self.compete(h)
        
        # 能量截断: 只有高loss才扩张
        if loss > 0.5 and self.latent_dim < self.max_budget:
            if np.random.random() < 0.03:
                self.expand()
        
        # 淘汰
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
                self.history['dim'].append(self.latent_dim)
                self.history['loss'].append(loss)
        
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


def run_experiment(budget, seed=0):
    np.random.seed(seed)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        sys_c = SystemC_Truncation(latent_dim=8, max_budget=budget, k_active=2)
        env = SimpleEnv(complexity=env_level)
        
        history = sys_c.run(env, steps=500)
        
        results[env_level] = {
            'final_dim': sys_c.latent_dim,
            'budget': budget,
            'avg_loss': np.mean(history['loss'][-10:]),
            'dim_std': np.std(history['dim']),
            'spawns': history['spawns'],
            'prunes': history['prunes']
        }
    
    return results


def main():
    print("="*60)
    print("Experiment 4.1: 截断强度 vs 维度稳定性")
    print("Budget B ∈ {8, 12, 16, 24, 32}, k=2")
    print("="*60)
    
    budgets = [8, 12, 16, 24, 32]
    all_results = {}
    
    for budget in budgets:
        print(f"\n--- Budget = {budget} ---")
        result = run_experiment(budget, seed=0)
        all_results[budget] = result
        
        dims = [result[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        losses = [result[e]['avg_loss'] for e in ['E1','E2','E3','E4']]
        
        print(f"  Dims: {dims}, Avg: {np.mean(dims):.1f}")
        print(f"  Loss: {[f'{l:.3f}' for l in losses]}, Avg: {np.mean(losses):.3f}")
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\nBudget | E1 | E2 | E3 | E4 | AvgDim | AvgLoss")
    print("-"*55)
    
    for budget in budgets:
        r = all_results[budget]
        dims = [r[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        losses = [r[e]['avg_loss'] for e in ['E1','E2','E3','E4']]
        
        print(f"  {budget:>4} | {dims[0]:>2} | {dims[1]:>2} | {dims[2]:>2} | {dims[3]:>2} | {np.mean(dims):>6.1f} | {np.mean(losses):>7.3f}")
    
    return all_results


if __name__ == "__main__":
    main()
