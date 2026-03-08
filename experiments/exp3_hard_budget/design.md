# EXPERIMENT 3: Hard Budget + Stricter Competition

## 1. Motivation

Experiment 1 & 2 showed:
- ECS improves performance ✓
- Soft penalty does NOT stabilize structure ✗

**Core insight**: Soft constraints are insufficient. Need hard budget.

---

## 2. Hypothesis

**H3**: Hard capacity budget + stricter competition will stabilize representational scale.

**Mechanisms**:
1. Hard budget: `total_dims ≤ B`
2. Stricter competition: `top-k` with `k = 2` (not `dim/4`)

---

## 3. System Design

### System C2: Hard Budget ECS

```python
# Hard budget
MAX_DIMS = 16  # Fixed budget

# Stricter competition  
K = 2  # Only 2 dims active

# When spawning:
if current_dims < MAX_DIMS and residual > threshold:
    spawn()

# When pruning:
if importance < threshold:
    prune()
```

---

## 4. Variables

| System | Budget | Competition |
|--------|--------|-------------|
| C-baseline | ∞ | k=dim/4 |
| C-hard | 16 | k=2 |
| C-medium | 24 | k=4 |

---

## 5. Expected Results

```
dimension
│
│ C-baseline ─────────────
│
│ C-hard    ───
│
│ C-medium  ──────
│
└─────────────────────
  complexity
```

Hard budget should show plateau.

---

## 6. Success Criteria

- Dimension stabilizes at budget level
- Performance does not collapse

---

## 7. Failure Criteria

- Dimension exceeds budget (mechanism broken)
- Performance collapses

---

*Experiment 3 Design v1.0*
