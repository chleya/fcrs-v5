# FCRS (Fixed Capacity Representation System)

An intelligent system combining representation competition with online learning, built on the core framework of **Compression → Prediction → Selective Direction** for resource-constrained autonomous intelligence.

## Quick Start

```python
from fcrs import FCRS

fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)

for i in range(1000):
    x = your_input_data
    fcrs.step(x)

print(fcrs.get_avg_error())
```

### One-Click Reproduce Core Results

```bash
python run_experiment.py
```

This will reproduce the core milestone results: prediction-oriented selection achieves +24% success rate improvement over reconstruction-oriented selection in multi-step decision tasks.

## Core Results

### v5.3.1 (Multi-step Sequential Decision Task)

| Selection Mode | Success Rate | Cumulative Reward |
| -------------------- | ------------ | ----------------- |
| Prediction-Oriented | **56.0%** | -9.13 |
| Reconstruction-Oriented | 32.0% | -11.26 |
| Random Selection | 36.0% | -12.09 |

**+24.0% success rate lift in multi-step decision tasks!**

### v5.4.0 - RL Baseline Comparison

#### Task: Grid World (One-Hot)

| Method | Success Rate | Notes |
|--------|--------------|-------|
| Q-Learning | 100% | Best for discrete exact state |
| Random | 14% | Baseline |
| FCRS | 0% | Not suitable for low-dim exact |

#### Task: Multi-Step Grid World

| Method | Success Rate | Notes |
|--------|--------------|-------|
| **FCRS (Prediction)** | **56%** | Best for multi-step planning |
| Random | 36% | Baseline |

---

## Applicable Scenarios

| Scenario | Recommended Method |
|----------|-------------------|
| Low-dim discrete exact state | Q-Learning ✅ |
| Continuous simple tasks | Linear Q ✅ |
| **Multi-step planning + compression** | **FCRS** ✅ |

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v5.4.0 | 2026-03-09 | **Final**: RL baseline comparison & boundary analysis |
| v5.3.1 | 2026-03-09 | **Milestone**: 56% success rate (+24%) |
| v5.3.0 | 2026-03-09 | Multi-step decision breakthrough |
| v5.1 | 2026-03-08 | Baseline FCRS release |

## Key Innovation

> Intelligence = Compression → Prediction → Selective Direction

1. **Competition + Learning Synergy**: Fixed-capacity representation pool with competitive update
2. **Prediction-Oriented Compression**: Redefine compression goal from reconstruction to prediction
3. **Forward-Looking Selection**: Select based on multi-step prediction capability
4. **Clear Boundaries**: Know when to use FCRS vs traditional RL

## Core Files

- `run_experiment.py` - One-click reproduce
- `predictive/src/experiments/stability_test.py` - 10-seed stability verification
- `predictive/src/experiments/gridworld_v2.py` - Core experiment
- `predictive/src/experiments/compare_v3.py` - RL baseline comparison
- `EXPERIMENT_REPORT.md` - Experimental rigor documentation

## Documentation

- `README.md` - Overview
- `THEORY.md` - Theory foundation
- `FINAL_REPORT.md` - Complete summary
- `EXPERIMENT_REPORT.md` - Experimental rigor

---

*Version 5.4.0 - 2026-03-09*
