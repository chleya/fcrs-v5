# FCRS-v5 API参考

## 快速开始

```python
from core import FCRSystem

# 创建系统
system = FCRSystem(pool_capacity=5, vector_dim=10)

# 运行
for i in range(1000):
    system.step()

# 获取统计
stats = system.get_statistics()
print(stats['total_dims'])
```

---

## 核心API

### FCRSystem

| 方法 | 说明 |
|------|------|
| `step()` | 执行一步 |
| `get_statistics()` | 获取统计信息 |

### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pool_capacity | 5 | 表征池容量 |
| vector_dim | 10 | 初始维度 |
| spawn_reuse_threshold | 2 | 复用阈值 |
| min_compression_gain | 0.001 | 压缩增益阈值 |

---

## 扩展API

### 神经网络

```python
from neural_extension import FCRSNet

net = FCRSNet(input_dim=10)
output = net.forward(input_data)
```

### 多智能体

```python
from multi_agent import FCRSMultiAgent

system = FCRSMultiAgent(n_agents=3)
system.run(steps=100)
```

---

## 配置

```json
{
  "default": {
    "pool_capacity": 5,
    "vector_dim": 10,
    "spawn_reuse_threshold": 2,
    "min_compression_gain": 0.001
  }
}
```
