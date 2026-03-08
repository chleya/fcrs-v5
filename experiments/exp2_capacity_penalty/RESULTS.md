# Experiment 2 Results

## Summary

### λ vs Average Dimension (E1-E4)

| λ | E1 | E2 | E3 | E4 | Avg |
|---|-----|-----|-----|-----|-----|
| 0 | 9.3 | 27.3 | 28.3 | 24.7 | 22.4 |
| 0.001 | 9.3 | 27.3 | 28.3 | 26.0 | 22.7 |
| 0.01 | 9.3 | 27.3 | 28.3 | 21.0 | 21.5 |
| 0.1 | 9.3 | 27.3 | 28.3 | 25.0 | 22.5 |

---

## Key Observations

### 1. Low Complexity (E1)
- Dimension stable at ~9.3 across all λ
- Capacity penalty has NO effect in simple environments

### 2. Medium Complexity (E2)
- Dimension grows to ~27.3
- No regulation effect observed

### 3. High Complexity (E3)
- Dimension reaches ~28.3
- Maximum capacity reached (near max_dim=32)

### 4. Complex (E4)
- Some variation: 21-26
- λ=0.01 shows lowest dimension (21.0)
- BUT: effect is small and inconsistent

---

## Interpretation

### Hypothesis H2: NOT SUPPORTED

Capacity penalty alone does NOT stabilize representational capacity.

```
dimension
│
│ λ=0    ─────────
│
│ λ=0.001 ─────────
│
│ λ=0.01 ────────
│
│ λ=0.1  ─────────
└─────────────────────
  complexity
```

The curves overlap - no meaningful difference.

---

## Possible Reasons

1. **Penalty too weak**: λ ∈ {0.001, 0.01} may be too small
2. **System still under capacity**: More dimensions → lower loss
3. **Missing resource constraint**: Need hard budget, not soft penalty
4. **Competition mechanism**: Top-K may need stricter k

---

## Recommendations for Next Experiment

### Option 1: Hard Budget
```
total_capacity ≤ B
```

### Option 2: Stronger Competition
```
k = 2  # instead of k = dim/4
```

### Option 3: L1 Activation Penalty
```
loss = task_loss + λ * sum(|activation|)
```

---

## Conclusion

Experiment 2 shows that:

- Soft capacity penalty is insufficient
- Structural stabilization requires stronger mechanisms
- The research question remains open

---

*Results recorded: 2026-03-08*
