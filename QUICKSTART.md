# FCRS-v5 快速开始

## 安装

```bash
pip install -r requirements.txt
```

## 基本使用

```python
from fcrs import FCRS

# 创建系统
fcrs = FCRS(pool_capacity=5, input_dim=10, lr=0.01)

# 运行
for i in range(1000):
    x = generate_input()  # 你的输入数据
    fcrs.step(x)

# 获取结果
print(fcrs.get_avg_error())
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pool_capacity | 5 | 表征池容量 |
| input_dim | 10 | 输入维度 |
| lr | 0.01 | 学习率 |

---

*快速开始 v1.0 - 2026-03-08*
