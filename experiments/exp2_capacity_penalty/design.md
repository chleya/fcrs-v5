# Experiment 2: Capacity Cost Test

**Date**: 2026-03-08  
**Hypothesis**: H1 - 缺乏容量成本  
**Objective**: 验证添加容量惩罚是否能稳定表征维度

---

## 1. 唯一变量

| 变量 | 值 |
|------|-----|
| λ (capacity penalty) | {0, 0.001, 0.01, 0.1} |

---

## 2. 固定参数 (从Exp1冻结)

| 参数 | 值 |
|------|-----|
| spawn_threshold | 同Exp1 |
| prune_threshold | 同Exp1 |
| competition | 同Exp1 |
| environment | 同Exp1 |
| seeds | {0, 1, 2} |

---

## 3. Capacity Cost 定义

```
Loss_total = Loss_task + λ * active_dimension_count
```

其中:
- `active_dimension_count`: importance > 0.01 的维度数

---

## 4. 实验矩阵

| λ | seeds | 总运行数 |
|----|-------|---------|
| 0 | 0,1,2 | 3 |
| 0.001 | 0,1,2 | 3 |
| 0.01 | 0,1,2 | 3 |
| 0.1 | 0,1,2 | 3 |

**总计**: 12 runs

---

## 5. 记录指标

| 指标 | 说明 |
|------|------|
| loss | 任务误差 |
| dimension | 总维度数 |
| active_dimension | 有效维度数 |
| capacity_penalty | λ * dimension |
| spawn_count | 新表征生成数 |
| prune_count | 表征修剪数 |
| entropy | 表征分布熵 |

---

## 6. 预期结果

### Case A (理想)
- λ ↑ → dimension ↓
- loss 在中间λ最优

### Case B
- dimension ↓ → loss ↑ (容量受限)

### Case C
- dimension 不变 (惩罚太弱)

---

## 7. 成功标准

dimension 稳定 AND loss 不显著增加

---

*Design v1.0 - 2026-03-08*
