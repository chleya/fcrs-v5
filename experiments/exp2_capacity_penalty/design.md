# EXPERIMENT 2: Capacity Regulation

## 1. Motivation

Experiment 1 showed three key findings:

| Finding | Evidence |
|---------|----------|
| Capacity expansion improves performance | C < B < A in complex environments |
| Pure expansion causes continuous growth | B: 8→14→24→25 |
| ECS improves performance but does not stabilize | C: 8→27→29→28 |

**Conclusion**: Expansion alone is insufficient for regulating representational capacity.

**Therefore**: We investigate capacity regulation mechanisms.

---

## 2. Hypothesis

**H2**: Adding a capacity cost will regulate representational growth and lead to a stable structural scale.

**Formally**:
```
Loss_total = Loss_task + λ * dimension_count
```

Where λ controls capacity pressure.

**Predictions**:

| λ | Expected behavior |
|---|-------------------|
| 0 | Uncontrolled growth |
| small | Moderated growth |
| moderate | Stable capacity |
| large | Under-capacity |

---

## 3. System Definition

### Base System
System C (ECS)

### Components
- spawn → competition → selection → prune

### Extended System
C + capacity penalty

---

## 4. Experimental Variables

**Independent variable**: λ ∈ {0, 0.001, 0.01, 0.1}

| λ | Meaning | Expected |
|---|---------|----------|
| 0 | baseline ECS | uncontrolled |
| 0.001 | weak pressure | moderated |
| 0.01 | balanced | stable |
| 0.1 | strong | under-capacity |

---

## 5. Environment Setup

Same as Experiment 1:

| Env | Complexity |
|-----|------------|
| E1 | single object |
| E2 | multiple objects |
| E3 | occlusion |
| E4 | randomized physics |

**Important**: Model parameters are NOT reset between environments.

---

## 6. Metrics

### 6.1 Task Loss
`task_loss` - Measured on test environments.

### 6.2 Representation Size
`dimension_count` - Active latent dimensions.

### 6.3 Dimension Utilization
`dimension_entropy` - Measures representation efficiency.

### 6.4 Structural Dynamics
- `spawn_events`
- `prune_events`

---

## 7. Expected Results

### Figure 1: complexity vs loss
Expectation: moderate λ performs best.

### Figure 2: complexity vs dimension
Expected pattern:
```
dimension
│
│ /\ 
│ / \___
│
└────────────
```

Meaning: growth → stabilization

### Figure 3: λ vs dimension
Expected:
```
dimension
│
│ λ=0 ↑
│
│ λ=0.001
│
│ λ=0.01 ← stable
│
│ λ=0.1 ↓ under
└───────────
```

---

## 8. Success Criteria

**H2 is supported if**:
- dimension growth stabilizes
- AND generalization performance remains high

---

## 9. Failure Criteria

**H2 is rejected if**:
- capacity still grows indefinitely
- OR performance collapses under penalty

---

## 10. Research Question

This experiment addresses:

> What mechanism regulates representational capacity?

---

## 11. Directory Structure

```
experiments/
├── exp1_baseline/
│   ├── system_a.py
│   ├── system_b.py
│   └── system_c.py
└── exp2_capacity_penalty/
    ├── design.md
    └── system_c_penalty.py
```

---

*Experiment 2 Design v1.0*
