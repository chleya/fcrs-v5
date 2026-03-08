# Experiment 5: λ Phase Transition

**Date**: 2026-03-08  
**Question**: 当λ变化时，capacity是否出现phase transition？

---

## 1. 实验设计

固定 n_classes = 50 (最难任务)

变量: λ ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}

---

## 2. 理论预测

可能出现:
- λ大 → dimension小
- λ小 → dimension大
- **关键**: 可能在某个λ值出现**sharp jump**

---

## 3. 预期结果

| 区域 | 行为 |
|------|------|
| λ > λc | dimension ≈ 10 (blocked) |
| λ ≈ λc | **phase transition** |
| λ < λc | dimension ≈ runaway |

---

*Design v1.0 - 2026-03-08*
