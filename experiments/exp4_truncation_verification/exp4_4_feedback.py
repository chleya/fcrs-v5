# FCRS-v5 Experiment 4.4: 回流机制验证

"""
Experiment 4.4: 回流类型 vs 结构稳定性

回流类型:
- 无回流: 无反馈
- 梯度回流: 误差反传
- 注意力回流: 重要信息强化
- 预测回流: 期望vs现实
"""

import numpy as np


class System_NoFeedback:
    """无回流系统"""
    def __init__(self, latent_dim=8, max_budget=16):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.history = {'dim': [], 'loss': []}
    
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
        
        # 无回流: 只向前学习
        if loss > 0.3 and np.random.random() < 0.05:
            self.expand()
        
        if np.random.random() < 0.1:
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01
        
        return loss


class System_GradientFeedback:
    """梯度回流系统"""
    def __init__(self, latent_dim=8, max_budget=16):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.history = {'dim': [], 'loss': []}
    
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
        # 简化前向
        h = np.dot(x, self.encoder_w)
        x_hat = np.dot(h, self.decoder_w)
        loss = np.mean((x - x_hat) ** 2)
        
        # 梯度回流: 用随机扰动代替
        if loss > 0.3 and np.random.random() < 0.05:
            self.expand()
        
        if np.random.random() < 0.1:
            # 随机学习
            self.encoder_w += np.random.randn(*self.encoder_w.shape) * 0.01 * (1.0/(loss+0.1))
            self.decoder_w += np.random.randn(*self.decoder_w.shape) * 0.01 * (1.0/(loss+0.1))
        
        return loss


class System_AttentionFeedback:
    """注意力回流系统"""
    def __init__(self, latent_dim=8, max_budget=16):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.attention = np.ones(latent_dim) / latent_dim
        self.history = {'dim': [], 'loss': []}
    
    def expand(self):
        if self.latent_dim >= self.max_budget:
            return False
        new_encoder = np.random.randn(9, 1) * 0.1
        new_decoder = np.random.randn(1, 9) * 0.1
        self.encoder_w = np.hstack([self.encoder_w, new_encoder])
        self.decoder_w = np.vstack([self.decoder_w, new_decoder])
        self.attention = np.append(self.attention, 1.0/self.latent_dim)
        self.latent_dim += 1
        return True
    
    def train(self, x):
        h = np.tanh(np.dot(x, self.encoder_w))
        x_hat = np.tanh(np.dot(h, self.decoder_w))
        loss = np.mean((x - x_hat) ** 2)
        
        # 注意力回流: 根据激活强度更新注意力
        activation = np.abs(h)
        self.attention = 0.9 * self.attention + 0.1 * (activation / (np.sum(activation) + 1e-8))
        
        # 用注意力加权训练
        if loss > 0.3 and np.random.random() < 0.05:
            self.expand()
        
        if np.random.random() < 0.1:
            # 简化注意力加权
            self.encoder_w *= (0.99 + 0.01 * np.mean(self.attention))
            self.decoder_w *= (0.99 + 0.01 * np.mean(self.attention))
            
        return loss


class System_PredictionFeedback:
    """预测回流系统"""
    def __init__(self, latent_dim=8, max_budget=16):
        self.latent_dim = latent_dim
        self.max_budget = max_budget
        self.encoder_w = np.random.randn(9, latent_dim) * 0.1
        self.decoder_w = np.random.randn(latent_dim, 9) * 0.1
        self.prediction = np.zeros(9)  # 上一步的预测
        self.history = {'dim': [], 'loss': []}
    
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
        
        # 预测回流: 用上一步预测指导当前学习
        if np.sum(self.prediction) != 0:
            # 预测误差
            pred_error = x - self.prediction
            # 根据预测误差调整表征
            h = h + 0.01 * np.dot(pred_error, self.decoder_w.T)
        
        # 更新预测
        self.prediction = x_hat.copy()
        
        loss = np.mean((x - x_hat) ** 2)
        
        if loss > 0.3 and np.random.random() < 0.05:
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
        
        dims_history = []
        loss_history = []
        
        for step in range(500):
            obs = env.get_obs()
            loss = sys.train(obs)
            
            if step % 50 == 0:
                dims_history.append(sys.latent_dim)
                loss_history.append(loss)
        
        results[env_level] = {
            'final_dim': sys.latent_dim,
            'avg_dim': np.mean(dims_history),
            'dim_std': np.std(dims_history),
            'avg_loss': np.mean(loss_history[-10:])
        }
    
    return results


def main():
    print("="*60)
    print("Experiment 4.4: 回流机制对比")
    print("="*60)
    
    systems = [
        ("无回流", System_NoFeedback, {'max_budget': 16}),
        ("梯度回流", System_GradientFeedback, {'max_budget': 16}),
        ("注意力回流", System_AttentionFeedback, {'max_budget': 16}),
        ("预测回流", System_PredictionFeedback, {'max_budget': 16}),
    ]
    
    all_results = {}
    
    for name, SystemClass, kwargs in systems:
        print(f"\n--- {name} ---")
        results = run_system(SystemClass, latent_dim=8, **kwargs)
        all_results[name] = results
        
        dims = [results[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        stds = [results[e]['dim_std'] for e in ['E1','E2','E3','E4']]
        losses = [results[e]['avg_loss'] for e in ['E1','E2','E3','E4']]
        
        print(f"  Dims: {dims}, Avg: {np.mean(dims):.1f}")
        print(f"  Std: {[f'{s:.2f}' for s in stds]}, Avg: {np.mean(stds):.2f}")
        print(f"  Loss: {[f'{l:.3f}' for l in losses]}, Avg: {np.mean(losses):.3f}")
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("\n回流类型 | E1 | E2 | E3 | E4 | AvgDim | DimStd | AvgLoss")
    print("-"*65)
    
    for name, results in all_results.items():
        dims = [results[e]['final_dim'] for e in ['E1','E2','E3','E4']]
        stds = [results[e]['dim_std'] for e in ['E1','E2','E3','E4']]
        losses = [results[e]['avg_loss'] for e in ['E1','E2','E3','E4']]
        
        print(f"{name:^10} | {dims[0]:>2} | {dims[1]:>2} | {dims[2]:>2} | {dims[3]:>2} | {np.mean(dims):>6.1f} | {np.mean(stds):>6.2f} | {np.mean(losses):>7.3f}")
    
    return all_results


if __name__ == "__main__":
    main()
