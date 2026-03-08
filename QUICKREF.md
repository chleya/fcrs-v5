# FCRS-v5 快速参考卡

## 命令行

```bash
# 运行演示
python demo.py

# 运行测试
python test_core.py

# 运行基准
python experiments/benchmark.py
```

---

## Python

```python
# 基础
from core import FCR FCRSystem(pool_capacity=5,System
system = vector_dim=10)

# 运行
for i in range(1000):
    system.step()

# 统计
print(system.get_statistics())
```

---

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| pool_capacity | 5 | 池大小 |
| vector_dim | 10 | 初始维度 |
| threshold | 2 | 复用阈值 |
| gain | 0.001 | 压缩增益 |
