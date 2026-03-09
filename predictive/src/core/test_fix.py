# -*- coding: utf-8 -*-
"""
测试核心修复
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from core_predictive import *

def test():
    print("="*60)
    print("Core Fix Test")
    print("="*60)
    
    results = {'pred': [], 'recon': [], 'random': []}
    
    for run in range(20):
        np.random.seed(run * 100)
        
        # 预测系统
        Config.set(LEARNING_RATE=0.01, EXPLORATION_RATE=0.1)
        env = SimpleEnv(10, 5, 0.3)
        sys = PredictiveSystem()
        
        for _ in range(500):
            x = env.generate_input()
            sys.step(x)
        
        results['pred'].append(np.mean(sys.recon_errors[-100:]))
        
        # 重构基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        baseline = ReconstructionBaseline(10, 3, 0.01)
        
        for _ in range(500):
            x = env.generate_input()
            baseline.step(x)
        
        results['recon'].append(np.mean(baseline.errors[-100:]))
        
        # 随机基线
        np.random.seed(run * 100)
        env = SimpleEnv(10, 5, 0.3)
        rand = RandomBaseline2(10, 3)
        
        for _ in range(500):
            x = env.generate_input()
            rand.step(x)
        
        results['random'].append(np.mean(rand.errors[-100:]))
    
    print(f"\nPredictive:   {np.mean(results['pred']):.4f} +/- {np.std(results['pred']):.4f}")
    print(f"Reconstruction: {np.mean(results['recon']):.4f} +/- {np.std(results['recon']):.4f}")
    print(f"Random:       {np.mean(results['random']):.4f} +/- {np.std(results['random']):.4f}")
    
    pred_vs_recon = (np.mean(results['recon']) - np.mean(results['pred'])) / np.mean(results['recon']) * 100
    pred_vs_random = (np.mean(results['random']) - np.mean(results['pred'])) / np.mean(results['random']) * 100
    
    print(f"\nPredictive vs Reconstruction: {pred_vs_recon:+.1f}%")
    print(f"Predictive vs Random: {pred_vs_random:+.1f}%")
    
    if pred_vs_recon > 5:
        print("\n[OK] Predictive > Reconstruction!")
    elif pred_vs_recon > 0:
        print("\n[~] Predictive > Reconstruction (marginal)")
    else:
        print("\n[X] Predictive <= Reconstruction")

if __name__ == "__main__":
    test()
