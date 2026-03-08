# Experiment 2a: Objective Penalty Only

**Date**: 2026-03-08  
**Hypothesis**: H1 - 缺乏容量成本  
**Design**: penalty only in loss function

---

## 结果

| λ | task_loss | penalty | dimension |
|----|-----------|---------|-----------|
| 0 | 1.40 | 0.00 | 50.0 |
| 0.001 | 1.40 | 0.01 | 50.0 |
| 0.01 | 1.40 | 0.05 | 50.0 |
| 0.1 | 1.40 | 0.50 | 50.0 |

---

## 发现

> **目标函数中的惩罚不影响结构扩张**

所有λ值下，维度都是50.0，惩罚只影响损失计算。

---

## 结论

**Objective-level penalty ≠ Structural regulation**

这说明：
- loss function中的惩罚无法调节architecture dynamics
- 涌现机制独立于目标函数

---

## Exp2b 计划

**Design**: penalty coupled to emergence rule

**规则**:
```
spawn if Δfitness > λ
prune if importance < λ
```

**预期**: 
λ dimension
0 50
0.01 40
0.1 20
1 5

---

*Created: 2026-03-08*
