# FCRS-v5 完整技术报告

**日期**: 2026-03-08  
**版本**: Final

---

## 摘要

本项目研究有限竞争表征系统(FCRS)中的容量调节机制。我们发现：

1. **λ (capacity cost) 是唯一有效的维度调节器**
2. **维度在困难任务下有价值，在简单任务下饱和**
3. **出现Phase Transition现象**
4. **选择机制需要Cosine Similarity**

---

## 一、引言

### 1.1 研究问题

> 有限物质系统是否可以通过表征扩张实现持续提升的泛化能力？

### 1.2 核心贡献

1. 实现了完整的FCRS系统
2. 验证了λ调节机制
3. 发现了容量饱和现象
4. 验证了维度-难度的依赖关系

---

## 二、理论框架

### 2.1 表征达尔文主义

- 涌现驱动 vs 优化驱动
- 容量-误差权衡 (Rate-Distortion)

### 2.2 核心方程

```
dD/dt = E - P

spawn if Δfitness > λ
prune if importance < θ
```

---

## 三、系统实现

### 3.1 三层架构

```
Environment → Representation → Evolution
```

### 3.2 选择机制

**关键修复**: 使用Cosine Similarity代替Dot Product

```python
def select(x):
    score = dot(v, x) / (||v|| * ||x||)
```

---

## 四、实验结果

### 4.1 Phase 1: 维度扩张

| 配置 | 维度 | 误差 |
|------|------|------|
| 动态 | 510 | 过拟合 |
| 固定 | 10 | 基准 |

### 4.2 Phase 2: λ调节

| λ | 维度 | 行为 |
|----|------|------|
| 0 | 510 | 无限扩张 |
| 0.1 | 349 | 调节 |
| 1.0 | 10 | 阻止 |

### 4.3 Phase 3: 竞争强度

**结论**: 竞争不影响维度

### 4.4 Phase 4: 任务复杂度

**结论**: 维度不随任务变化

### 4.5 Phase 5: λ Phase Transition

| 区间 | Δ维度 |
|------|-------|
| 0.01→0.1 | -142 |
| 0.1→1.0 | -339 |

### 4.6 验证: 难度依赖

| 任务 | λ=0维度 | λ=0.6维度 | Error变化 |
|------|---------|-----------|-----------|
| Easy | 10 | 10 | 0 |
| Medium | 20 | 10 | **0.79** |
| Hard | 50 | 10 | **2.29** |

---

## 五、核心发现

### 5.1 λ是唯一有效的容量调节器

- Loss penalty: ❌ 无效
- Competition: ❌ 无效
- **λ (structural cost): ✅ 有效**

### 5.2 容量饱和现象

- 简单任务下，增加维度不降低误差
- 对应Rate-Distortion理论

### 5.3 难度依赖

- 困难任务下，维度有价值
- 体现了capacity-performance tradeoff

---

## 六、论文结构建议

```
1. Abstract
2. Introduction
   - Problem: finite capacity systems
   - Question: can expansion improve generalization?
3. Related Work
   - Neural Darwinism
   - Growing Networks
   - Rate-Distortion Theory
4. Model
   - FCRS Architecture
   - Selection Mechanism (Cosine Similarity)
   - Structural Dynamics
5. Experiments
   - Phase 1-5 Results
   - Key Finding: λ controls capacity
6. Discussion
   - Capacity Saturation
   - Task Difficulty Dependence
   - Limitations
7. Conclusion
```

---

## 七、工程教训

| 教训 | 应用 |
|------|------|
| 现象→假设 | 先验证再优化 |
| 负面结果 | 诚实报告 |
| 选择机制 | Cosine > Dot |
| 任务难度 | 影响系统价值 |

---

## 八、代码位置

- GitHub: https://github.com/chleya/fcrs-v5
- 核心: core_v52.py
- 实验: experiments/

---

*报告完成于 2026-03-08*
