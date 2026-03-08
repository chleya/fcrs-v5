# FCRS-v5 完整技术文档

## 系统概述

FCRS (Fixed Capacity Representation System) v5.1 是一个基于"截断-竞争-回流"机制的自适应表征维度系统。

### 核心理念

> 智能 = 能量截断 + 信息截断 + 回流维持

---

## 架构

```
┌─────────────────────────────────────────┐
│           Environment Loop               │
│            (环境输入生成)                 │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Representation Pool              │
│  ┌─────────────────────────────────┐   │
│  │ Representation 1: dim=10        │   │
│  │ Representation 2: dim=11        │   │
│  │ ...                             │   │
│  └─────────────────────────────────┘   │
│         (表征池 + 预算约束)              │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│         Evolution Engine                │
│  - 竞争(Competition)                     │
│  - 复用(Reuse)                          │
│  - 新维度诞生(Spawn)                    │
└─────────────────────────────────────────┘
```

---

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| pool_capacity | 表征池容量 | 5 |
| vector_dim | 初始向量维度 | 10 |
| spawn_reuse_threshold | 复用阈值 | 2 |
| min_compression_gain | 最小压缩增益 | 0.001 |

---

## API

### 创建系统

```python
from core import FCRSystem

system = FCRSystem(pool_capacity=5, vector_dim=10)
```

### 运行

```python
for i in range(1000):
    system.step()
```

### 获取统计

```python
stats = system.get_statistics()
print(stats['total_dims'])  # 总维度
print(stats['new_dims_born'])  # 新维度诞生数
```

---

## 实验结果

### 长期稳定性

| 步数 | 维度 | 状态 |
|------|------|------|
| 500 | 100 | 扩张 |
| 2000 | 100 | 稳定 |
| 5000 | 102 | 准收敛 |

### 最佳配置

- pool_capacity = 10
- spawn_reuse_threshold = 2
- min_compression_gain = 0.001

---

## 文件结构

```
FCRS-v5/
├── core.py              # 核心系统
├── demo.py              # 演示脚本
├── EXPERIMENT_SUMMARY.md # 实验总结
├── PAPER_OUTLINE.md     # 论文大纲
├── PAPER.md             # 完整论文
├── neural_extension/    # 神经网络扩展
├── multi_agent/         # 多智能体
├── experiments/         # 实验代码
│   ├── benchmark.py
│   ├── ablation_test.py
│   ├── test_sensitivity.py
│   └── ...
└── paper/               # 图表
    ├── fig_dimension_evolution.png
    └── ...
```

---

## 引用

```bibtex
@article{fcrs-v5,
  title={截断与反馈：面向自适应表征维度的智能系统},
  author={Chen, Leiyang},
  year={2026}
}
```
