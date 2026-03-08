# FCRS-v5 技术文档

## 目录

1. [概述](#1-概述)
2. [架构](#2-架构)
3. [API](#3-api)
4. [使用示例](#4-使用示例)
5. [理论](#5-理论)
6. [实验](#6-实验)

---

## 1. 概述

FCRS (Fixed Capacity Representation System) 是一个结合表征竞争与在线学习的智能系统。

**核心创新**: 表征竞争 + 在线学习 = 协同效应

### 核心特点

- 三层架构: 环境层 / 表征层 / 进化层
- 在线学习机制
- 表征竞争机制
- 预算约束

---

## 2. 架构

```
┌─────────────────────────────────────┐
│         环境层 (Environment)          │
│   RandomEnvironment / StructuredEnv  │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         表征层 (Representation)      │
│  VectorRepresentation / SimplePool   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│         进化层 (Evolution)           │
│       CompetitionEvolution           │
└─────────────────────────────────────┘
```

---

## 3. API

### 3.1 环境类

```python
class Environment:
    def generate_input(self):
        """生成输入"""
        pass
    
    def reset(self):
        """重置"""
        pass
```

### 3.2 表征类

```python
class Representation:
    def get_vector(self):
        """获取向量"""
        pass
    
    def update(self, x, lr):
        """更新"""
        pass
    
    def get_fitness(self):
        """获取适应度"""
        pass
```

### 3.3 表征池

```python
class RepresentationPool:
    def add(self, representation):
        """添加表征"""
        pass
    
    def select(self, x):
        """选择表征"""
        pass
    
    def get_total_dims(self):
        """总维度"""
        pass
```

### 3.4 主系统

```python
class FCRS:
    def __init__(self, env, pool, evolution, lr=0.01):
        pass
    
    def step(self):
        """执行一步"""
        pass
    
    def get_avg_error(self):
        """平均误差"""
        pass
    
    def get_stats(self):
        """统计信息"""
        pass
```

---

## 4. 使用示例

### 4.1 基础使用

```python
from fcrs_architecture import create_fcrs

# 创建系统
fcrs = create_fcrs(
    env_type='structured',
    pool_capacity=5,
    input_dim=10,
    n_classes=5,
    lr=0.01
)

# 运行
for _ in range(1000):
    fcrs.step()

# 结果
print(fcrs.get_avg_error())
```

### 4.2 自定义

```python
from fcrs_architecture import (
    StructuredEnvironment,
    SimplePool,
    CompetitionEvolution,
    FCRS,
    VectorRepresentation
)

# 自定义环境
env = StructuredEnvironment(input_dim=10, n_classes=5)

# 自定义表征池
pool = SimplePool(capacity=5, input_dim=10)

# 添加表征
for _ in range(3):
    rep = VectorRepresentation(env.generate_input())
    pool.add(rep)

# 创建系统
evolution = CompetitionEvolution()
fcrs = FCRS(env, pool, evolution, lr=0.01)

# 运行
for _ in range(1000):
    fcrs.step()
```

---

## 5. 理论

### 5.1 核心假设

**H1**: 资源约束下系统形成稳定表征

**H2**: 竞争-选择机制提高表征质量

**H3**: 表征复杂度与环境复杂度正相关

### 5.2 收敛性

在在线学习机制下，系统收敛到输入分布的均值。

---

## 6. 实验

### 6.1 基线对比

| 系统 | 误差 |
|------|------|
| **FCRS** | **2.11** |
| Online Learning | 6.23 |
| Competition | 7.65 |
| Fixed | 7.38 |
| Random | 8.46 |

**结论**: FCRS显著优于所有基线 (p<0.001)

### 6.2 消融实验

| 移除组件 | 效果 |
|----------|------|
| 竞争 | 误差显著增加 |
| 学习 | 误差显著增加 |

**结论**: 两个组件都重要

---

*技术文档 v1.0 - 2026-03-08*
