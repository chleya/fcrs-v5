# FCRS-v5.1 最小可验证实验蓝图

**目标**: 用最小系统验证核心假设

---

## 1️⃣ 研究假设

**H**: 在环境复杂度增加时，扩张+竞争+淘汰(ECS)机制可以使表征维度自组织到稳定规模，并提升跨环境泛化能力。

**形式化**:
```
Environment complexity ↑
→ representation dimension grows
→ but stabilizes via competition & pruning
→ generalization performance improves
```

---

## 2️⃣ 三个对照系统

| 系统 | 机制 | 预期 |
|------|------|------|
| A | 固定容量(dim=8) | 维度不变，复杂环境饱和 |
| B | 纯扩张(spawn only) | 8→16→32→64，过拟合 |
| C | ECS(spawn+compete+prune) | 8→12→16→17→稳定 |

### 系统A: 固定容量
```python
latent_dim = 8
spawn = False
prune = False
```

### 系统B: 扩张系统
```python
if residual > threshold:
    spawn_new_dim()
```

### 系统C: 生态系统
```python
# 扩张
if residual > threshold and capacity_saturated:
    spawn_new_dim()

# 竞争
active_dims = top_k(latent, k=4)

# 淘汰
if importance_i < τ for T steps:
    prune_dim(i)
```

---

## 3️⃣ 环境复杂度阶梯

| 环境 | 复杂度 | 内容 |
|------|--------|------|
| E1 | 低 | 单物体 |
| E2 | 中 | 多物体 |
| E3 | 高 | 遮挡 |
| E4 | 更高 | 物理随机化 |

---

## 4️⃣ 训练流程

```
For each system in [A, B, C]:
    For each env_level in [E1, E2, E3, E4]:
        train(agent)        # 不重新初始化
        evaluate(agent)     # 记录指标
```

---

## 5️⃣ 三个必须指标

| 指标 | 说明 |
|------|------|
| 泛化误差 | 在unseen参数环境的预测误差 |
| 活跃维度数量 | activation > ε 的维度数 |
| 维度利用率 | dimension_entropy |

---

## 6️⃣ 关键曲线（论文核心）

### 图1: 环境复杂度 vs 泛化误差
```
error
│
│ A ───────
│ B ──╮
│ C ──╯
└──────── complexity
```

### 图2: 环境复杂度 vs 维度数量
```
dims
│ B ──────↗
│
│ C ──╮
│ ╰───
│
│ A ─────
└──────── complexity
```

### 图3: 维度利用率
```
entropy
│
│ B ↑↑↑
│
│ C stable
│
│ A low
└──────── time
```

---

## 7️⃣ 竞争机制

**Top-K Gating**:
```python
active_dims = top_k(latent, k=4)
```

---

## 8️⃣ 重要性评分

```python
importance_i = mean(|activation_i|)
# 或
importance_i = ∂loss / ∂dim_i
```

---

## 9️⃣ Prune规则

```python
if importance_i < τ for T steps:
    remove_dim(i)

# 推荐: T=500, τ=0.01
```

---

## 🔟 Spawn规则

```python
if residual_error > threshold AND latent_rank ≈ capacity:
    spawn_new_dim()
```

---

## 1️⃣1️⃣ 成功标准

- C系统泛化误差最低
- 维度数量稳定(不膨胀)

→ **假设H被支持**

---

## 1️⃣2️⃣ 失败标准

- C ≈ B (无差异)
- 维度持续膨胀

→ **假设H不成立**

---

## 1️⃣3️⃣ 最小实现规模

```
latent_dim_max = 32
training_steps = 200k
env_size = small
```

几天内可完成实验。

---

## 1️⃣4️⃣ 论文标题(参考)

> Structural Capacity Growth via Expansion-Competition-Selection Dynamics

或

> Do Representations Self-Organize Their Capacity?

---

## 核心一句话

**实验不是要证明"系统可以增加维度"，而是证明"结构规模会在环境复杂度下自组织到稳定点"。**

---

*最小可验证实验蓝图 v1.0*
