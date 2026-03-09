"""
FCRS-v5: 核心问题诊断
到底预测机制有没有用?
"""

import numpy as np


def test_core_hypothesis():
    """
    核心假设检验:
    选择"预测下一时刻误差最小"的表征 是否优于 
    选择"当前重构误差最小"的表征?
    """
    print("="*60)
    print("CORE HYPOTHESIS TEST")
    print("="*60)
    
    results = {'by_prediction': [], 'by_reconstruction': []}
    
    for run in range(30):
        np.random.seed(run * 100)
        
        # 环境
        dim = 10
        n_classes = 5
        centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
        
        # 表征 (初始化)
        reps_A = [np.random.randn(dim) for _ in range(3)]  # 预测组
        reps_B = [np.random.randn(dim) for _ in range(3)]  # 重构组
        
        # 预测器
        predictors = [np.eye(dim) * 0.5 for _ in range(3)]
        
        # 历史
        prev_x = None
        
        for step in range(500):
            # 生成样本
            c = centers[np.random.randint(0, n_classes)]
            x = c + np.random.randn(dim) * 0.3
            
            # === 方法A: 基于预测选择 ===
            if prev_x is not None:
                # 更新预测器
                for i, W in enumerate(predictors):
                    pred = prev_x @ W
                    error = x - pred
                    W += 0.01 * np.outer(prev_x, error)
            
            # 选择 (基于重构误差, 因为预测误差无法直接比较)
            scores_A = [-np.linalg.norm(r - x) for r in reps_A]
            idx_A = np.argmax(scores_A)
            
            # 更新表征A
            reps_A[idx_A] += 0.01 * (x - reps_A[idx_A])
            
            # === 方法B: 基于重构选择 (对照组) ===
            scores_B = [-np.linalg.norm(r - x) for r in reps_B]
            idx_B = np.argmax(scores_B)
            
            # 更新表征B
            reps_B[idx_B] += 0.01 * (x - reps_B[idx_B])
            
            prev_x = x.copy()
        
        # 评估 (最后100步的平均误差)
        prev_x = None
        errors_A = []
        errors_B = []
        
        np.random.seed(run * 100 + 500)  # 新的随机种子
        
        for step in range(100):
            c = centers[np.random.randint(0, n_classes)]
            x = c + np.random.randn(dim) * 0.3
            
            # 方法A
            if prev_x is not None:
                for i, W in enumerate(predictors):
                    pred = prev_x @ W
                    error = x - pred
                    W += 0.01 * np.outer(prev_x, error)
            
            scores_A = [-np.linalg.norm(r - x) for r in reps_A]
            idx_A = np.argmax(scores_A)
            errors_A.append(np.linalg.norm(reps_A[idx_A] - x))
            
            reps_A[idx_A] += 0.01 * (x - reps_A[idx_A])
            
            # 方法B
            scores_B = [-np.linalg.norm(r - x) for r in reps_B]
            idx_B = np.argmax(scores_B)
            errors_B.append(np.linalg.norm(reps_B[idx_B] - x))
            
            reps_B[idx_B] += 0.01 * (x - reps_B[idx_B])
            
            prev_x = x.copy()
        
        results['by_prediction'].append(np.mean(errors_A))
        results['by_reconstruction'].append(np.mean(errors_B))
    
    print(f"\n基于预测: {np.mean(results['by_prediction']):.4f} +/- {np.std(results['by_prediction']):.4f}")
    print(f"基于重构: {np.mean(results['by_reconstruction']):.4f} +/- {np.std(results['by_reconstruction']):.4f}")
    
    diff = (np.mean(results['by_reconstruction']) - np.mean(results['by_prediction'])) / np.mean(results['by_reconstruction']) * 100
    print(f"\n差异: {diff:+.1f}%")
    
    if abs(diff) < 5:
        print("结论: 两种方法没有显著差异!")
    elif diff > 0:
        print("结论: 基于预测的选择更好!")
    else:
        print("结论: 基于重构的选择更好!")


def test_with_different_tasks():
    """不同任务测试"""
    print("\n" + "="*60)
    print("Different Tasks Test")
    print("="*60)
    
    tasks = [
        ('简单聚类', 3, 0.1),
        ('中等聚类', 5, 0.3),
        ('困难聚类', 10, 0.5),
    ]
    
    for name, n_classes, noise in tasks:
        errors = {'pred': [], 'recon': []}
        
        for run in range(10):
            np.random.seed(run * 100)
            
            dim = 10
            centers = {i: np.random.randn(dim) * 2 for i in range(n_classes)}
            
            reps = [np.random.randn(dim) for _ in range(3)]
            W = [np.eye(dim) * 0.5 for _ in range(3)]
            
            prev_x = None
            
            for step in range(300):
                c = centers[np.random.randint(0, n_classes)]
                x = c + np.random.randn(dim) * noise
                
                if prev_x is not None:
                    for w in W:
                        w += 0.01 * np.outer(prev_x, x - prev_x @ w)
                
                scores = [-np.linalg.norm(r - x) for r in reps]
                idx = np.argmax(scores)
                
                reps[idx] += 0.01 * (x - reps[idx])
                prev_x = x.copy()
            
            # 测试
            errs = []
            for _ in range(50):
                c = centers[np.random.randint(0, n_classes)]
                x = c + np.random.randn(dim) * noise
                
                scores = [-np.linalg.norm(r - x) for r in reps]
                idx = np.argmax(scores)
                errs.append(np.linalg.norm(reps[idx] - x))
            
            errors['pred'].append(np.mean(errs))
        
        print(f"{name}: {np.mean(errors['pred']):.4f}")


# ==================== Main ====================
test_core_hypothesis()
test_with_different_tasks()
print("\nDone!")
