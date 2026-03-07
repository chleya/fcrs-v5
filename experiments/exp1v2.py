"""
实验1v2：原型表征的涌现（改进版）

调整：
1. 降低容量N=10 → 增加竞争压力
2. 改进适应度计算
3. 增加训练步数
"""

import numpy as np
import json


class Representation:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector
        self.fitness_history = []
        self.activation_count = 0
        self.age = 0
    
    @property
    def reuse_frequency(self):
        return self.activation_count / max(1, self.age)


class Environment:
    """环境：3个类别，不同均值"""
    def __init__(self, num_classes=3, dim=10):
        self.num_classes = num_classes
        self.dim = dim
        self.centers = {i: np.random.randn(dim) * 3 for i in range(num_classes)}
    
    def generate(self):
        cls = np.random.randint(0, self.num_classes)
        x = self.centers[cls] + np.random.randn(self.dim) * 0.3
        return x, cls


class FCRSystem:
    def __init__(self, capacity=10, dim=10):
        self.capacity = capacity
        self.dim = dim
        self.reps = []
        self.next_id = 0
    
    def add(self, vec):
        rep = Representation(self.next_id, vec.copy())
        self.next_id += 1
        self.reps.append(rep)
        return rep
    
    def select(self, x):
        if not self.reps:
            return None
        # 改进：选择与输入最接近的
        best = None
        best_dist = float('inf')
        for r in self.reps:
            d = np.linalg.norm(r.vector - x)
            if d < best_dist:
                best_dist = d
                best = r
        return best
    
    def persistence(self, r):
        avg_fit = np.mean(r.fitness_history[-5:]) if r.fitness_history else -1
        reuse = r.reuse_frequency
        cost = r.age * 0.01
        return avg_fit * 0.5 + reuse * 0.3 - cost
    
    def step(self, x, cls):
        # 年龄+1
        for r in self.reps:
            r.age += 1
        
        # 选择
        active = self.select(x)
        
        if active is None:
            self.add(x)
            return
        
        # 更新
        active.activation_count += 1
        error = np.linalg.norm(active.vector - x)
        fitness = -error
        active.fitness_history.append(fitness)
        
        # 变异
        if np.random.random() < 0.05:  # 降低变异率
            # 新表征
            new_vec = active.vector + np.random.randn(self.dim) * 0.3
            
            # 淘汰一个
            if len(self.reps) >= self.capacity:
                worst = min(self.reps, key=lambda r: self.persistence(r))
                self.reps.remove(worst)
            
            self.add(new_vec)


def silhouette(vectors, labels):
    """简化轮廓系数"""
    n = len(vectors)
    if n < 2:
        return 0
    
    scores = []
    for i, v in enumerate(vectors):
        # 类内距离
        same = [vectors[j] for j in range(n) if labels[j] == labels[i] and j != i]
        a = np.mean([np.linalg.norm(v - s) for s in same]) if same else 0
        
        # 类间最小距离
        other_labels = set(labels) - {labels[i]}
        b = float('inf')
        for ol in other_labels:
            others = [vectors[j] for j in range(n) if labels[j] == ol]
            d = np.mean([np.linalg.norm(v - o) for o in others])
            b = min(b, d)
        
        if max(a, b) > 0:
            scores.append((b - a) / max(a, b))
    
    return np.mean(scores) if scores else 0


def run():
    print("=" * 50)
    print("实验1v2: 原型涌现（改进版）")
    print("=" * 50)
    
    # 配置
    CAPACITY = 10  # 大幅降低
    DIM = 10
    CLASSES = 3
    STEPS = 3000
    
    env = Environment(CLASSES, DIM)
    sys = FCRSystem(CAPACITY, DIM)
    
    print(f"配置: 容量={CAPACITY}, 维度={DIM}, 类别={CLASSES}, 步数={STEPS}")
    
    history = {'steps': [], 'sil': [], 'pool': []}
    class_history = []
    
    for s in range(STEPS):
        x, c = env.generate()
        sys.step(x, c)
        class_history.append(c)
        
        if s % 300 == 0 and len(sys.reps) >= CLASSES:
            vecs = np.array([r.vector for r in sys.reps])
            # 用表征被激活时的类别作为标签
            labels = class_history[-len(sys.reps):]
            sil = silhouette(vecs, labels)
            
            history['steps'].append(s)
            history['sil'].append(sil)
            history['pool'].append(len(sys.reps))
            
            print(f"  Step {s}: 轮廓={sil:.3f}, 池={len(sys.reps)}")
    
    final_sil = history['sil'][-1] if history['sil'] else 0
    
    print(f"\n结果:")
    print(f"  最终轮廓系数: {final_sil:.3f}")
    print(f"  表征数量: {len(sys.reps)}")
    
    if final_sil > 0:
        print("v 验证通过")
    else:
        print("X 验证未通过")
    
    return {
        'config': {'capacity': CAPACITY, 'dim': DIM, 'classes': CLASSES, 'steps': STEPS},
        'history': history,
        'final_sil': final_sil,
        'success': final_sil > 0
    }


if __name__ == "__main__":
    result = run()
    
    with open('F:/fcrs-v5/experiments/exp1v2_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print("\n结果已保存")
