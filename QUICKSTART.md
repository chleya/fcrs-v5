# FCRS-v5 快速开始

## 5分钟入门

### 1. 运行示例

```bash
python core.py
```

### 2. 运行测试

```bash
python -m pytest tests/
```

### 3. 运行实验

```bash
# 对比实验
python experiments/comparison.py

# MNIST测试
python experiments/real_mnist.py
```

---

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lambda_val | 0.5 | 容量成本 |
| dimension | 10 | 初始维度 |
| learning_rate | 0.1 | 学习率 |

---

## 目录结构

```
fcrs-v5/
├── core.py          # 核心代码
├── experiments/     # 实验代码
├── docs/           # 文档
└── datasets/      # 数据集
```

---

## 下一步

- 阅读 CONCEPT.md 了解理论
- 阅读 TECHNICAL_REPORT.md 了解实验
