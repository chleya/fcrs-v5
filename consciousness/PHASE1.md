# 意识研究 Phase 1: 自适应瓶颈实验

**日期**: 2026-03-08

---

## 核心理念

废除手动λ，让系统根据环境复杂度自己博弈：
> "我是该省力（高λ）而保持模糊，还是该观察入微（低λ）而精准？"

---

## 理论背景

| 理论 | 核心概念 | 我们要验证 |
|------|----------|-----------|
| GWT | 全局工作空间 | 表征如何"胜出"进入广播？ |
| IIT | 整合信息Φ | "黄金维度"之间的依赖性？ |
| FEP | 最小化惊奇 | 系统能否预测下一状态？ |

---

## 实验设计

### 自适应λ系统

```python
class AdaptiveFCRS:
    """自适应容量的FCRS"""
    
    def __init__(self):
        self.lambda = 0.5  # 初始值
        
        # 状态
        self.uncertainty = 0  # 环境不确定性
        self.surprise = 0    # 当前惊奇度
        self.attention = []  # 广播历史
    
    def step(self, x):
        # 1. 观察
        surprise = self.calculate_surprise(x)
        
        # 2. 适应λ
        if surprise > self.threshold:
            self.lambda *= 0.9  # 降低阈值，更细致
        else:
            self.lambda *= 1.1  # 提高阈值，更模糊
        
        # 3. 限制λ范围
        self.lambda = clamp(self.lambda, 0.01, 1.0)
        
        # 4. 选择"广播"
        if surprise > self.lambda:
            self.broadcast(x)  # 进入全局工作空间
```

---

## 关键问题

1. **λ会收敛吗？** → 是否有稳定策略
2. **广播什么？** → 什么样的表征被选中
3. **广播有什么用？** → 是否改善后续决策

---

## 预期结果

| 环境 | 预期λ | 预期广播率 |
|------|--------|-----------|
| 简单固定 | 高 | 低 |
| 复杂变化 | 低 | 高 |
| 突发威胁 | 极低 | 极高 |

---

## 与现有工作的连接

- FCRS-v5: 容量控制机制 ✅
- 意识Phase1: **自主选择机制** 🔄

---

*Start: 2026-03-08*
