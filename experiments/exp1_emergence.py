"""
实验1：原型表征的涌现
简化版（不依赖sklearn）
"""

import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Representation:
    id: int
    vector: np.ndarray
    fitness_history: list = None
    activation_count: int = 0
    age: int = 0
    
    def __post_init__(self):
        if self.fitness_history is None:
            self.fitness_history = []
    
    @property
    def reuse_frequency(self):
        if self.age == 0:
            return 0.0
        return self.activation_count / self.age


class SimpleClustering:
    """简化聚类分析"""
    
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        
    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        n = len(vectors)
        if n < self.n_clusters:
            return np.arange(n)
        
        # 简化K-means
        centroids = vectors[np.random.choice(n, self.n_clusters, replace=False)]
        
        for _ in range(10):  # 10次迭代
            # 分配
            labels = np.zeros(n, dtype=int)
            for i, v in enumerate(vectors):
                dists = [np.linalg.norm(v - c) for c in centroids]
                labels[i] = np.argmin(dists)
            
            # 更新
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    centroids[k] = np.mean(vectors[mask], axis=0)
        
        return labels
    
    def silhouette(self, vectors: np.ndarray, labels: np.ndarray) -> float:
        """简化轮廓系数"""
        n = len(vectors)
        if n < 2:
            return 0
        
        scores = []
        for i, v in enumerate(vectors):
            same = labels[i]
            others = [labels[j] for j in range(n) if j != i]
            
            a = np.mean([np.linalg.norm(v - vectors[j]) 
                       for j in range(n) if labels[j] == same and j != i] or [0])
            b = np.min([np.mean([np.linalg.norm(v - vectors[j]) 
                               for j in range(n) if labels[j] == o]) for o in set(others)] or [0])
            
            if max(a, b) > 0:
                scores.append((b - a) / max(a, b))
        
        return np.mean(scores) if scores else 0


class EnvironmentWithClasses:
    """带类别的环境"""
    
    def __init__(self, num_classes: int, input_dim: int):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.class_means = {i: np.random.randn(input_dim) * 2 
                           for i in range(num_classes)}
    
    def generate_input(self):
        cls = np.random.randint(0, self.num_classes)
        x = self.class_means[cls] + np.random.randn(self.input_dim) * 0.5
        return x, cls


class FCRSystem:
    """有限竞争表征系统"""
    
    def __init__(self, pool_capacity: int = 50, vector_dim: int = 20):
        self.pool_capacity = pool_capacity
        self.vector_dim = vector_dim
        self.representations = []
        self.next_id = 0
        self.step_count = 0
        self.alpha, self.beta, self.gamma = 0.5, 0.3, 0.2
    
    def add(self, vector: np.ndarray):
        rep = Representation(self.next_id, vector.copy())
        self.next_id += 1
        self.representations.append(rep)
        return rep
    
    def select(self, x: np.ndarray):
        if not self.representations:
            return None
        
        best = None
        best_score = float('-inf')
        
        for rep in self.representations:
            score = np.dot(rep.vector, x)
            if score > best_score:
                best_score = score
                best = rep
        
        return best
    
    def calculate_persistence(self, rep: Representation) -> float:
        avg_fitness = np.mean(rep.fitness_history[-10:]) if rep.fitness_history else 0
        reuse = rep.reuse_frequency
        cost = rep.vector.shape[0] / 100.0
        return self.alpha * avg_fitness + self.beta * reuse - self.gamma * cost
    
    def select_for_deletion(self):
        if len(self.representations) < self.pool_capacity:
            return None
        
        worst = None
        worst_p = float('inf')
        
        for rep in self.representations:
            p = self.calculate_persistence(rep)
            if p < worst_p:
                worst_p = p
                worst = rep
        
        return worst
    
    def step(self, x: np.ndarray, cls: int):
        self.step_count += 1
        
        active = self.select(x)
        
        if active is None:
            self.add(x)
            return cls
        
        active.activation_count += 1
        
        error = np.linalg.norm(active.vector - x)
        fitness = -error
        active.fitness_history.append(fitness)
        
        # 年龄
        for rep in self.representations:
            rep.age += 1
        
        # 变异
        if np.random.random() < 0.1:
            strength = 0.1 / (1 + active.age * 0.1)
            new_vec = active.vector + np.random.randn(*active.vector.shape) * strength
            
            to_del = self.select_for_deletion()
            if to_del:
                self.representations.remove(to_del)
            
            self.add(new_vec)
        
        return cls


def run_experiment_1(pool_capacity=30, vector_dim=20, num_classes=3, total_steps=2000):
    """运行实验1"""
    print("=" * 60)
    print("实验1：原型表征的涌现")
    print("=" * 60)
    
    env = EnvironmentWithClasses(num_classes, vector_dim)
    system = FCRSystem(pool_capacity, vector_dim)
    cluster = SimpleClustering(num_classes)
    
    history = {'steps': [], 'ari': [], 'pool_size': []}
    rep_classes = []  # 每个表征对应的类别
    
    print(f"\n配置: 容量={pool_capacity}, 维度={vector_dim}, 类别={num_classes}, 步数={total_steps}")
    
    for step in range(total_steps):
        x, cls = env.generate_input()
        rep_cls = system.step(x, cls)
        rep_classes.append(rep_cls)
        
        if step % 200 == 0 and len(system.representations) >= num_classes:
            # 聚类分析
            vectors = np.array([r.vector for r in system.representations])
            labels = cluster.fit_predict(vectors)
            sil = cluster.silhouette(vectors, labels)
            
            history['steps'].append(step)
            history['ari'].append(sil)
            history['pool_size'].append(len(system.representations))
            
            print(f"  Step {step}: 轮廓系数={sil:.3f}, 池大小={len(system.representations)}")
    
    final_sil = history['ari'][-1] if history['ari'] else 0
    
    print(f"\n结果:")
    print(f"  最终轮廓系数: {final_sil:.3f}")
    print(f"  表征数量: {len(system.representations)}")
    
    if final_sil > 0.2:
        print("v 原型涌现验证通过")
    else:
        print("X 原型涌现验证未通过")
    
    return {
        'experiment': 'phenomenon_1',
        'config': {'pool': pool_capacity, 'dim': vector_dim, 'classes': num_classes, 'steps': total_steps},
        'history': history,
        'final_silhouette': final_sil,
        'success': final_sil > 0.2
    }


if __name__ == "__main__":
    result = run_experiment_1()
    
    # 保存
    import os
    os.makedirs('F:/fcrs-v5/experiments', exist_ok=True)
    with open('F:/fcrs-v5/experiments/exp1_result.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n结果已保存")
