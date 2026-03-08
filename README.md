# FCRS (Fixed Capacity Representation System)

An intelligent system combining representation competition with online learning.

## Quick Start

```python
from fcrs import FCRS

fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)

for i in range(1000):
    x = your_input_data
    fcrs.step(x)

print(fcrs.get_avg_error())
```

## Results

| System | Error |
|--------|-------|
| **FCRS** | **3.16** |
| Online Learning | 6.23 |
| Competition Only | 7.65 |
| Fixed | 7.38 |
| Random | 8.46 |

**FCRS beats all baselines (p<0.001)**

## Key Innovation

> Competition + Learning = Synergy

## Files

- `fcrs.py` - Core module
- `core_improved.py` - Full implementation
- `experiments/rigorous_improved.py` - Validation
- `THEORY.md` - Theory
- `FINAL_REPORT.md` - Summary

## License

MIT

---

*Version 5.1 - 2026-03-08*
