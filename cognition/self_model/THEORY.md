# 自我模型研究: 深入探索

**目标**: 构建真正的自我模型

---

## 四个层次

| 层次 | 能力 | 测试 |
|------|------|------|
| L1 | 自我存在 | 知道自己活着 |
| L2 | 自我模拟 | 能预测自己 |
| L3 | 自我决策 | 能选择目标 |
| L4 | 元认知 | 知道"我知道什么" |

---

## 设计

### L1: 自我存在

```
- 追踪自己的状态 (energy, age, alive)
- 能回答"我还活着吗？"
```

### L2: 自我模拟

```
- 建立"虚拟自我"
- 能在行动前模拟结果
- 比较现实 vs 模拟
```

### L3: 自我决策

```
- 有多个目标选项
- 能评估每个目标的价值
- 能选择最优
- 能学习偏好
```

### L4: 元认知

```
- 知道自己的能力边界
- 能说"我不知道"
- 能主动学习未知
```

---

## 代码结构

```python
class SelfModel:
    """
    完整自我模型
    """
    
    # L1: 存在
    self_state: {energy, age, alive}
    
    # L2: 模拟
    self_model: {position, velocity}
    simulate(action) -> predicted_state
    
    # L3: 决策
    goals: [explore, rest, learn]
    decide() -> chosen_goal
    
    # L4: 元认知
    knowledge: {known: [], unknown: []}
    interrogate(topic) -> "know/don't know"
```

---

*Start: 2026-03-08*
