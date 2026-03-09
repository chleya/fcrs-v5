# FCRS-v5 Complete Iteration History

## Full Iteration & Bug Fix History

### Phase 1: Core Bug Fix (v5.2.1-baseline)

Fixed core implementation defects that prevented the framework from achieving expected performance:

1. **Fixed encoder weight initialization**: Replaced random initialization with Xavier initialization to solve gradient instability and vanishing gradient issues
2. **Corrected state transition model logic**: Fixed the execution order of ReLU activation and normalization to avoid zero-vector division errors
3. **Implemented unified Config management**: Centralized parameter management to solve scattered hard-coded parameters and experiment irreproducibility issues
4. **Optimized prediction selection evaluation**: Changed from single-step instantaneous error to historical multi-step prediction error evaluation

### Phase 2: Strategy Optimization & Task Exploration (v5.2.2 / v5.2.3)

1. Completed full parameter scan of multi-step prediction horizon, verified optimal horizon=5
2. Completed grid search of dual-objective loss function, verified optimal weight (recon=0.3 / pred=0.7)
3. Built grid world multi-step decision task baseline, identified and fixed the core decoupling issue between representation selection and action decision-making

### Phase 3: Core Breakthrough Verification (v5.3.0-milestone)

Rebuilt the end-to-end decision-making pipeline, strictly bound representation selection to action generation, completed rigorous controlled variable experiments, and verified the core advantage of prediction-oriented selection.

---

## Full Experimental Results

### 1. Static Single-Step Compression Task (v5.1 Baseline)

Controlled variable: All systems use the same representation pool capacity, input dimension, learning rate and training steps.

| System | Average Error | Significance |
| -------------------- | ------------- | ------------ |
| FCRS (Competition + Learning) | 3.16 | p<0.001 vs all baselines |
| Online Learning Only | 6.23 | - |
| Competition Only | 7.65 | - |
| Fixed Representation | 7.38 | - |
| Random Selection | 8.46 | - |

### 2. Multi-Step Sequential Decision Task (v5.3.0 Milestone)

**Task**: Partially observable grid world navigation (agent only observes 1 grid around, needs forward-looking planning to reach the target, with obstacle penalty and target reward)

**Controlled Variable**: Only the representation selection strategy is different, all other modules (compression, prediction, decision network, environment parameters) are completely consistent.

| Selection Mode | Success Rate (100 runs) | Cumulative Reward | Single-Step Reconstruction Error | Multi-Step Prediction Error |
| -------------------- | ------------------------ | ----------------- | -------------------------------- | --------------------------- |
| Prediction-Oriented | **46.0%** | **-9.13** | 1.17 | 0.67 |
| Reconstruction-Oriented | 24.0% | -11.26 | **1.09** (Optimal) | 0.67 |
| Random Selection | 30.0% | -12.09 | 1.21 | 0.76 |

#### Key Findings

- **Prediction-oriented selection achieves a 22.0% absolute increase in task success rate** and a **2.14 increase in cumulative reward** over reconstruction-oriented selection
- Even with similar multi-step prediction error, prediction-oriented selection achieves significantly better decision-making performance, proving that forward-looking representation selection can better support sequential planning
- Reconstruction-oriented selection maintains optimal single-step reconstruction precision, which is consistent with the theoretical conclusion that it is the optimal solution for single-step static tasks
