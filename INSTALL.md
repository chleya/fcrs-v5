# FCRS-v5 安装指南

## 环境要求

- Python 3.8+
- NumPy
- scikit-learn (用于数据集)

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/chleya/fcrs-v5.git
cd fcrs-v5
```

### 2. 安装依赖

```bash
pip install numpy scikit-learn
```

### 3. 快速测试

```bash
python core.py
```

---

## 数据集下载 (可选)

数据集已下载到 `F:/datasets/`

- digits.npz (0.9 MB)
- mnist.npz (439 MB)

---

## 运行示例

### 基本运行

```python
from fcrs import FCRS

fcrs = FCRS(lambda_val=0.5)
fcrs.run(steps=1000)
```

### 自定义

```python
fcrs = FCRS(
    lambda_val=0.5,    # 容量成本
    dimension=10,       # 初始维度
    learning_rate=0.1   # 学习率
)
```

---

## 常见问题

### Q: FCRS是什么？

A: Finite Capacity Representation System - 有限竞争表征系统

### Q: λ是什么？

A: 容量成本参数，控制维度增长

### Q: 如何调参？

A: 建议从 λ=0.5 开始，根据任务调整
