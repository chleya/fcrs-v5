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

This will reproduce the core milestone results: prediction-oriented selection achieves +22% success rate improvement over reconstruction-oriented selection in multi-step decision tasks.

## Core Results

### v5.1 Baseline (Static Single-Step Task)

| System | Error |
|--------|-------|
| **FCRS** | **3.16** |
| Online Learning | 6.23 |
| Competition Only | 7.65 |
| Fixed | 7.38 |
| Random | 8.46 |

**FCRS beats all baselines (p<0.001)**

### v5.3.0 Milestone (Multi-step Sequential Decision Task)

| Selection Mode | Success Rate | Cumulative Reward |
| -------------------- | ------------ | ----------------- |
| Prediction-Oriented | **46.0%** | **-9.13** |
| Reconstruction-Oriented | 24.0% | -11.26 |
| Random Selection | 30.0% | -12.09 |

**+22.0% success rate lift in multi-step decision tasks!**

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v5.3.0 | 2026-03-09 | **Milestone**: Multi-step decision breakthrough (+22% success) |
| v5.2.x | 2026-03-09 | Bug fixes & optimization attempts |
| v5.1 | 2026-03-08 | Baseline FCRS release |

## Key Innovation

> Intelligence = Compression → Prediction → Selective Direction

1. **Competition + Learning Synergy**: Fixed-capacity representation pool with competitive update
2. **Prediction-Oriented Compression**: Redefine compression goal from reconstruction to prediction
3. **Forward-Looking Selection**: Select based on multi-step prediction capability

## Files

- `fcrs.py` - Core module
- `core_improved.py` - Full implementation
- `predictive/` - v5.3.0 predictive framework
- `experiments/` - Validation experiments
- `THEORY.md` - Theory
- `FINAL_REPORT.md` - Full summary

## License

MIT

---

*Version 5.3.0 - 2026-03-09*
