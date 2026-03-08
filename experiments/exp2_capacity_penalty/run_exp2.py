# FCRS-v5 Experiment 2: Capacity Penalty

"""
Experiment 2: Capacity Penalty System

Loss_total = Loss_task + λ * dimension_count
"""

import numpy as np


class SystemC_Penalty:
    """ECS + Capacity Penalty"""
    
    def __init__(self, latent_dim=8, max_dim=32, lambda_penalty=0.01):
        self.latent_dim = latent_dim
        self.max_dim = max_dim
        self.lambda_penalty = lambda_penalty
        
        # 权重
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # 重要性
        self.importance = np.ones(latent_dim)
        
        # 记录
        self.history = {'dim': [], 'loss': [], 'task_loss': [], 'penalty': [], 'spawns': 0, 'prunes': 0}
    
    def expand(self):
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
        k = max(2, self.latent_dim // 4)
        self.importance = 0.9 * self.importance + 0.1 * np.abs(h)
    
    def prune(self):
        if self.latent_dim <= 4:
            return False
        
        min_idx = np.argmin(self.importance)
        
        if self.importance[min_idx] < 0.1:
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
        
        # Task loss
        task_loss = np.mean((x - x_hat) ** 2)
        
        # Capacity penalty
        active_dims = np.sum(self.importance > 0.1)
        capacity_penalty = self.lambda_penalty * active_dims
        
        # Total loss
        loss = task_loss + capacity_penalty
        
        # 竞争
        self.compete(h)
        
        # 扩张
        if task_loss > 0.3 and self.latent_dim < self.max_dim:
            if np.random.random() < 0.05:
                self.expand()
        
        # 淘汰
        if np.random.random() < 0.02:
            self.prune()
        
        # 简化训练
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss, task_loss, capacity_penalty, h
    
    def run(self, env, steps=500):
        for step in range(steps):
            obs = env.get_obs()
            loss, task_loss, penalty, h = self.train(obs)
            
            if step % 100 == 0:
                active = np.sum(np.abs(h) > 0.1)
                self.history['dim'].append(self.latent_dim)
                self.history['loss'].append(loss)
                self.history['task_loss'].append(task_loss)
                self.history['penalty'].append(penalty)
                print(f"Step {step}: dim={self.latent_dim}, loss={loss:.4f}, task={task_loss:.4f}, penalty={penalty:.4f}")
        
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


def run_experiment(lambda_penalty, seed=0):
    """运行单个实验"""
    np.random.seed(seed)
    
    print(f"\n{'='*50}")
    print(f"λ = {lambda_penalty}, seed = {seed}")
    print('='*50)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        print(f"\n--- {env_level} ---")
        
        sys_c = SystemC_Penalty(latent_dim=8, max_dim=32, lambda_penalty=lambda_penalty)
        env = SimpleEnv(complexity=env_level)
        
        history = sys_c.run(env, steps=500)
        
        results[env_level] = {
            'final_dim': sys_c.latent_dim,
            'spawns': history['spawns'],
            'prunes': history['prunes'],
            'avg_loss': np.mean(history['loss'][-5:]),
            'avg_task_loss': np.mean(history['task_loss'][-5:]),
            'avg_penalty': np.mean(history['penalty'][-5:])
        }
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("FCRS-v5 Experiment 2: Capacity Penalty")
    print("="*60)
    
    # λ 矩阵
    lambdas = [0, 0.001, 0.01, 0.1]
    seeds = [0, 1, 2]
    
    all_results = {}
    
    for lam in lambdas:
        lam_results = []
        for seed in seeds:
            result = run_experiment(lam, seed)
            lam_results.append(result)
        all_results[lam] = lam_results
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for lam in lambdas:
        print(f"\nλ = {lam}:")
        avg_dims = []
        for env in ['E1', 'E2', 'E3', 'E4']:
            dims = [r[env]['final_dim'] for r in all_results[lam]]
            avg = np.mean(dims)
            avg_dims.append(avg)
            print(f"  {env}: dim = {avg:.1f}")
    
    return all_results


if __name__ == "__main__":
    main()
