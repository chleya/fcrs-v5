# FCRS-v5 项目元信息

## 基本信息

| 属性 | 值 |
|------|-----|
| 项目名称 | FCRS (Fixed Capacity Representation System) |
| 版本 | 5.1 |
| 状态 | 活跃 |
| 语言 | Python |
| 许可证 | MIT |

## 核心文件

| 文件 | 用途 |
|------|------|
| core_improved.py | 主系统(改进版) |
| experiments/rigorous_improved.py | 严格验证 |

## 依赖

```
numpy>=1.20
matplotlib>=3.5
scipy>=1.7
```

## 快速开始

```python
from core_improved import ImprovedFCRS

system = ImprovedFCRS(pool_capacity=5, input_dim=10, lr=0.01)

for i in range(1000):
    x = generate_input()
    system.step(x)

print(system.get_avg_error())
```

## 贡献者

- Chen Leiyang (项目所有者)
- OpenClaw AI Assistant

---

*元信息 v1.0 - 2026-03-08*
