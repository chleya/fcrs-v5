# CHANGELOG

## 2026-03-09 - v5.3.0 Milestone!

### 🎉 Major Breakthrough
- **Core Discovery**: Prediction-oriented selection significantly outperforms reconstruction-oriented selection in multi-step decision tasks
- **Success Rate**: Prediction 46.0% vs Reconstruction 24.0% (+22.0% absolute improvement)
- **Cumulative Reward**: Prediction -9.13 vs Reconstruction -11.26 (+2.14 improvement)

### Key Files Added
- `src/core/grid_world.py` - GridWorld navigation environment
- `src/core/metrics.py` - Dual-dimension evaluation metrics
- `src/experiments/gridworld_v2.py` - Milestone experiment

### Technical Evolution
1. **v5.2.1-baseline**: Core bug fixes (Xavier init, ReLU order, Config)
2. **v5.2.2-pred-selection-attempt**: Prediction selection optimization attempt
3. **v5.2.3-gridworld-baseline**: GridWorld baseline, identified decision decoupling issue
4. **v5.3.0-milestone-core-breakthrough**: Core breakthrough - representation-driven decision

### Core Conclusion
- Single-step tasks: Reconstruction-oriented selection is mathematically optimal
- Multi-step sequential decision tasks: Prediction-oriented selection has significant advantage
- The value of prediction-oriented compression lies in forward-looking decision capability

---

## 2026-03-09 - Earlier Today

### Added
- `predictive/` directory uploaded
- PAPER.md - Complete paper draft
- THEORY.md - Theory framework and lessons learned

### Key Findings (Earlier)
- Prediction vs Reconstruction: No significant difference (-3.2%)
- Strict testing revealed system worse than simple baseline
- Ablation experiments showed all configs identical

### Lessons Learned
1. Simple baselines often stronger
2. Strict validation is critical
3. Negative results are also scientific results

---

## Earlier Versions (Reference)

### v5.0 - Original Version
- λ-based dimension control
- Edge consensus research

### v5.1 - Emergence-Driven
- Attempted "emergence-driven" mechanism
- Later questioned as "false emergence"

### v5.2 - Prediction-Oriented (Current)
- "Compression → Prediction → Selection" framework
- Includes L1-L3 capabilities
- Initial validation results were negative (before milestone)
