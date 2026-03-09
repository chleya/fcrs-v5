# FCRS-v5 (Fixed Capacity Representation System)

An intelligent system combining representation competition with online learning, built on the core framework of **Compression → Prediction → Selective Direction** for resource-constrained autonomous intelligence.

## Quick Start

```python
from fcrs import FCRS
from predictive.integrated import IntegratedSystem

# Core FCRS (v5.1 Baseline: Static Compression Task)
fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)
for i in range(1000):
    x = your_input_data
    fcrs.step(x)
print(fcrs.get_avg_error())

# Predictive FCRS (v5.3.0 Milestone: Multi-step Decision Task)
system = IntegratedSystem(
    input_dim=20,
    compress_dim=5,
    pool_capacity=10,
    lr=0.01,
    selection_mode="prediction"  # options: prediction / reconstruction / random
)
# Run end-to-end sequential decision task
system.run_env(env=GridWorldEnv(), total_steps=10000)
print(f"Task Success Rate: {system.get_success_rate():.1f}%")
print(f"Cumulative Reward: {system.get_cumulative_reward():.2f}")
```

## Core Results

### v5.1 Baseline (Static Single-Step Compression Task)

FCRS beats all baselines with statistical significance (p<0.001):

| System | Average Error |
| -------------------- | ------------- |
| FCRS (v5.1 Baseline) | 3.16 |
| Online Learning Only | 6.23 |
| Competition Only | 7.65 |
| Fixed Representation | 7.38 |
| Random Selection | 8.46 |

### v5.3.0 Milestone (Multi-step Sequential Decision Task)

**Core Breakthrough**: Prediction-oriented selection significantly outperforms reconstruction-oriented selection in forward-looking planning tasks, verified in partially observable grid world navigation:

| Selection Mode | Success Rate | Cumulative Reward | Reconstruction Error |
| -------------------- | ------------ | ----------------- | -------------------- |
| Prediction-Oriented | **46.0%** | **-9.13** | 1.17 |
| Reconstruction-Oriented | 24.0% | -11.26 | **1.09** (Optimal) |
| Random Selection | 30.0% | -12.09 | 1.21 |

#### Key Improvement

- **+22.0%** absolute lift in task success rate over reconstruction-oriented selection
- **+2.14** lift in cumulative reward over reconstruction-oriented selection
- Slightly higher reconstruction error is expected: prediction-oriented compression optimizes for future state prediction, not single-step reconstruction precision

## Key Innovation

1. **Competition + Learning Synergy**: Fixed-capacity representation pool with online competitive update, achieving superior performance in static compression tasks

2. **Prediction-Oriented Compression Framework**: Core paradigm of Intelligence = Compression → Prediction → Selective Direction, redefining compression's optimization goal from signal reconstruction to future state prediction

3. **Forward-Looking Selection Mechanism**: Representation selection based on multi-step historical prediction capability, delivering significant advantages in sequential decision-making tasks that require long-term planning

## Project Structure

```
predictive/
├── docs/
│   ├── README.md          # This file
│   ├── PAPER.md           # Research paper
│   ├── THEORY.md          # Theory framework
│   └── CHANGELOG.md       # Version history
├── src/
│   ├── core/
│   │   ├── core_predictive.py    # Core modules
│   │   ├── grid_world.py         # GridWorld environment
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── optimized_predictive.py # Optimized version
│   ├── experiments/
│   │   ├── gridworld_experiment.py   # v1 experiment
│   │   └── gridworld_v2.py           # v2 experiment (milestone)
│   └── system/
│       ├── fixed_system.py
│       └── simple_fix.py
├── tests/
│   ├── test_v3.py
│   ├── strict_test.py
│   └── diagnose.py
└── archive/
    ├── experiments_v2.py
    └── experiments_fixed.py
```

## Theoretical Framework

### Intelligence = Compression → Prediction → Selective Direction

The core hypothesis of this project:

1. **Compression**: Summarize current situation into compact representation
2. **Prediction**: Predict future states based on representation
3. **Selective Direction**: Choose actions that lead to preferred outcomes

### Validation Levels

| Level | Capability | Status |
|-------|------------|--------|
| L1 | Pattern Recognition | ✅ Verified |
| L2 | Predictive Extrapolation | ✅ Verified |
| L3 | Causal Reasoning | 🔄 In Progress |

## Limitations & Future Work

- **L1-L2 capabilities validated**, causal reasoning (L3) remains exploratory
- Simple synthetic environments require more complex validation
- Current system demonstrates intelligent "capability" but lacks "understanding"

## Version History

| Version | Date | Description |
|---------|------|-------------|
| v5.3.0 | 2026-03-09 | **Milestone**: Multi-step decision breakthrough |
| v5.2.3 | 2026-03-09 | GridWorld baseline |
| v5.2.1 | 2026-03-09 | Core bug fixes |
| v5.1 | 2026-02 | Baseline FCRS |

---

*Last Updated: 2026-03-09*
