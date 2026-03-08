# Experiment 3: Competition × Capacity Cost

**Date**: 2026-03-08  
**Hypothesis**: H3 - 竞争机制与容量成本的交互  
**Design**: 2D grid - λ × elimination_rate

---

## 1. 实验矩阵

| λ | elimination_rate | dimension |
|----|-----------------|-----------|
| 0.01 | 0.1 | ? |
| 0.01 | 0.3 | ? |
| 0.01 | 0.5 | ? |
| 0.01 | 0.9 | ? |
| 0.1 | 0.1 | ? |
| 0.1 | 0.3 | ? |
| 0.1 | 0.5 | ? |
| 0.1 | 0.9 | ? |
| 1.0 | 0.1 | ? |
| 1.0 | 0.3 | ? |
| 1.0 | 0.5 | ? |
| 1.0 | 0.9 | ? |

---

## 2. 理论预测

| 区域 | 预期行为 |
|------|---------|
| λ小 + 竞争小 | dimension very large (runaway) |
| λ小 + 竞争大 | dimension moderate |
| λ大 | dimension small (cost主导) |

---

## 3. 预期结果

可能出现 two-regime system：
- cost-dominated: λ控制
- competition-dominated: 修剪控制

---

## 4. 指标

- final_dimension
- expansion_rate (spawn/step)
- pruning_rate (elim/step)

---

*Design v2.0 - 2026-03-08*
