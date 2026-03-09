"""
FCRS-v5: 深入诊断
检查每个组件是否正常工作
"""

import numpy as np


class DiagnosticSystem:
    """带诊断的系统"""
    
    def __init__(self, dim=10):
        self.dim = dim
        
        # 组件1: 表征
        self.reps = [np.random.randn(dim) for _ in range(3)]
        
        # 组件2: 预测器
        self.predictors = []
        for _ in range(3):
            self.predictors.append({
                'W': np.eye(dim) * 0.5,
                'errors': [],
                'update_count': 0
            })
        
        # 组件3: 选择
        self.selection_history = []
        
        # 组件4: 因果
        self.causal_history = []
        
        # 诊断
        self.diagnostics = {
            'predictor_improving': [],
            'selection_working': [],
            'causal_recording': []
        }
    
    def check_predictor(self, rep_idx, f_curr, f_next):
        """检查预测器"""
        p = self.predictors[rep_idx]
        
        # 预测
        pred = f_curr @ p['W']
        
        # 误差
        error = np.linalg.norm(f_next - pred)
        p['errors'].append(error)
        p['update_count'] += 1
        
        # 检查是否改进
        if len(p['errors']) > 20:
            early = np.mean(p['errors'][:10])
            late = np.mean(p['errors'][-10:])
            improving = (early - late) / (early + 1e-8)
            self.diagnostics['predictor_improving'].append(improving)
        
        return error
    
    def check_selection(self, rep_idx, x):
        """检查选择"""
        self.selection_history.append(rep_idx)
        
        # 检查选择是否有偏好
        if len(self.selection_history) > 50:
            counts = [self.selection_history.count(i) for i in range(3)]
            entropy = -sum((c/len(self.selection_history) * np.log(c/len(self.selection_history) + 1e-8) 
                         for c in counts if c > 0))
            self.diagnostics['selection_working'].append(entropy)
    
    def check_causal(self, action, effect):
        """检查因果"""
        self.causal_history.append((action, effect))
        
        if len(self.causal_history) > 50:
            # 检查是否有多样性
            actions = [a for a, e in self.causal_history[-50:]]
            avg_norm = np.mean([np.linalg.norm(a) for a in actions])
            self.diagnostics['causal_recording'].append(avg_norm)
    
    def get_diagnostics(self):
        """获取诊断报告"""
        report = {}
        
        # 预测器
        if self.diagnostics['predictor_improving']:
            avg_imp = np.mean(self.diagnostics['predictor_improving'])
            report['predictor'] = f"改进: {avg_imp:+.2%}"
        else:
            report['predictor'] = "无数据"
        
        # 选择
        if self.diagnostics['selection_working']:
            avg_ent = np.mean(self.diagnostics['selection_working'])
            report['selection'] = f"熵: {avg_ent:.2f}"
        else:
            report['selection'] = "无数据"
        
        # 因果
        if self.diagnostics['causal_recording']:
            avg = np.mean(self.diagnostics['causal_recording'])
            report['causal'] = f"动作范数: {avg:.2f}"
        else:
            report['causal'] = "无数据"
        
        return report


def test_diagnostics():
    """诊断测试"""
    print("="*60)
    print("DIAGNOSTIC TEST")
    print("="*60)
    
    sys = DiagnosticSystem(10)
    
    # 训练
    for step in range(500):
        # 随机输入
        x = np.random.randn(10)
        
        # 表征
        best_idx = 0
        best_err = float('inf')
        
        for i, rep in enumerate(sys.reps):
            err = np.linalg.norm(rep - x)
            if err < best_err:
                best_err = err
                best_idx = i
        
        # 检查选择
        sys.check_selection(best_idx, x)
        
        # 模拟预测器
        f_curr = sys.reps[best_idx]
        f_next = x + np.random.randn(10) * 0.1
        pred_err = sys.check_predictor(best_idx, f_curr, f_next)
        
        # 模拟因果
        action = np.random.randn(10)
        effect = np.random.randn(10)
        sys.check_causal(action, effect)
        
        # 更新表征
        sys.reps[best_idx] += 0.01 * (x - sys.reps[best_idx])
        
        if (step + 1) % 100 == 0:
            print(f"\nStep {step+1}:")
            diag = sys.get_diagnostics()
            for k, v in diag.items():
                print(f"  {k}: {v}")


def test_selection_bias():
    """测试选择偏差"""
    print("\n" + "="*60)
    print("SELECTION BIAS TEST")
    print("="*60)
    
    # 创建有明显差异的表征
    reps = [
        np.array([1.0, 0.0, 0.0] + [0]*7),  # 接近x轴
        np.array([0.0, 1.0, 0.0] + [0]*7),  # 接近y轴  
        np.array([0.0, 0.0, 1.0] + [0]*7),  # 接近z轴
    ]
    
    # 测试: 总是选择第一个
    selection = [0] * 100
    
    # 正确的选择应该是根据输入决定
    for _ in range(100):
        x = np.array([1.0, 0.0, 0.0] + [0]*7)  # x方向输入
        scores = [-np.linalg.norm(r - x) for r in reps]
        best = np.argmax(scores)
        selection[best] += 1
    
    print(f"Selection counts: {selection}")
    print("如果总是选第一个，则有偏差")


def test_predictor_learning():
    """测试预测器学习"""
    print("\n" + "="*60)
    print("PREDICTOR LEARNING TEST")
    print("="*60)
    
    # 简单线性系统: y = 2x
    W_true = np.eye(10) * 2.0
    
    # 预测器
    W = np.eye(10) * 0.1
    
    errors = []
    
    for step in range(200):
        x = np.random.randn(10)
        y = x @ W_true  # 真实输出
        
        pred = x @ W
        error = np.linalg.norm(y - pred)
        errors.append(error)
        
        # 更新
        W += 0.01 * np.outer(x, y - pred)
        
        if step % 50 == 0:
            print(f"Step {step}: error = {error:.4f}")
    
    print(f"\nInitial error: {errors[0]:.4f}")
    print(f"Final error: {errors[-1]:.4f}")
    print(f"Improvement: {(errors[0] - errors[-1])/errors[0]:.1%}")
    
    if errors[-1] < errors[0] * 0.5:
        print("预测器能学习!")
    else:
        print("预测器无法学习!")


def test_compression_effect():
    """测试压缩效果"""
    print("\n" + "="*60)
    print("COMPRESSION EFFECT TEST")
    print("="*60)
    
    # 不压缩
    errors_no_compress = []
    for run in range(10):
        reps = [np.random.randn(10) for _ in range(3)]
        
        errs = []
        for _ in range(200):
            x = np.random.randn(10) * 2
            best = min(reps, key=lambda r: np.linalg.norm(r - x))
            errs.append(np.linalg.norm(best - x))
            
            best += 0.01 * (x - best)
        
        errors_no_compress.append(np.mean(errs))
    
    # 压缩
    errors_compress = []
    for run in range(10):
        W = np.random.randn(10, 3) * 0.1
        
        errs = []
        for _ in range(200):
            x = np.random.randn(10) * 2
            f = x @ W
            f = np.maximum(0, f)
            
            # 表征是压缩后的
            best = min(f, key=lambda r: np.linalg.norm(r - f))
            errs.append(np.linalg.norm(best - f))
            
            # 更新W
            W += 0.01 * np.outer(x - W @ best, f - best)
        
        errors_compress.append(np.mean(errs))
    
    print(f"No compression: {np.mean(errors_no_compress):.4f}")
    print(f"Compression: {np.mean(errors_compress):.4f}")
    
    if np.mean(errors_compress) < np.mean(errors_no_compress):
        print("压缩有帮助!")
    else:
        print("压缩没有帮助!")


# ==================== Main ====================
test_diagnostics()
test_selection_bias()
test_predictor_learning()
test_compression_effect()

print("\n" + "="*60)
print("DIAGNOSTICS COMPLETE")
print("="*60)
