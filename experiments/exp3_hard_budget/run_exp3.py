# FCRS-v5 Experiment 3: Hard Budget + Stricter Competition

"""
Experiment 3: Hard Budget System

H3: Hard capacity budget + stricter competition stabilizes scale
"""

import numpy as np


class SystemC_Hard:
    """ECS with Hard Budget + Stricter Competition"""
    
    def __init__(self, latent_dim=8, max_budget=16, k_active=2):
        self.latent_dim = latent_dim
        self.max_budget = max_budget  # Hard budget
        self.k_active = k_active  # Stricter competition
        
        # Weights
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        
        # Importance
        self.importance = np.ones(latent_dim)
        
        # History
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
        """Stricter competition: only top-k matter"""
        # Update importance based on top-k activation
        k = min(self.k_active, self.latent_dim)
        top_k_indices = np.argsort(np.abs(h))[-k:]
        
        new_importance = np.zeros_like(self.importance)
        new_importance[top_k_indices] = np.abs(h[top_k_indices])
        
        self.importance = 0.9 * self.importance + 0.1 * new_importance
    
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
        loss = np.mean((x - x_hat) ** 2)
        
        # Stricter competition
        self.compete(h)
        
        # Expand: only if under budget
        if loss > 0.3 and self.latent_dim < self.max_budget:
            if np.random.random() < 0.05:
                self.expand()
        
        # Prune
        if np.random.random() < 0.02:
            self.prune()
        
        # Training
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
                print(f"Step {step}: dim={self.latent_dim}/{self.max_budget}, loss={loss:.4f}")
        
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


def run_experiment(budget, k_active, seed=0):
    np.random.seed(seed)
    
    print(f"\n{'='*50}")
    print(f"Budget={budget}, k={k_active}, seed={seed}")
    print('='*50)
    
    results = {}
    
    for env_level in ['E1', 'E2', 'E3', 'E4']:
        print(f"\n--- {env_level} ---")
        
        sys_c = SystemC_Hard(latent_dim=8, max_budget=budget, k_active=k_active)
        env = SimpleEnv(complexity=env_level)
        
        history = sys_c.run(env, steps=500)
        
        results[env_level] = {
            'final_dim': sys_c.latent_dim,
            'budget': budget,
            'avg_loss': np.mean(history['loss'][-5:])
        }
    
    return results


def main():
    print("="*60)
    print("FCRS-v5 Experiment 3: Hard Budget")
    print("="*60)
    
    # Configurations
    configs = [
        (32, 8),   # Baseline (no limit, loose competition)
        (16, 2),   # Hard budget, strict competition
        (24, 4),   # Medium budget
    ]
    
    all_results = {}
    
    for budget, k in configs:
        result = run_experiment(budget, k, seed=0)
        all_results[(budget, k)] = result
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for (budget, k), result in all_results.items():
        print(f"\nBudget={budget}, k={k}:")
        dims = [result[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        print(f"  Dims: {dims}")
        print(f"  Avg: {np.mean(dims):.1f}")
    
    return all_results


if __name__ == "__main__":
    main()
