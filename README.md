# FCRS-v5.1 有限竞争表征系统

## 项目代号

**FCRS-v5.1** (Finite Competitive Representation System)

## 核心命题

> 有限物质系统是否能够在资源约束和内部竞争的条件下，自发形成稳定且可复用的结构表征？

## 核心哲学

**约束不是智能发展的障碍，而是智能形成的必要条件。**

正如生物进化在资源受限的环境中创造了复杂的生命形态，有限的认知资源也将迫使智能系统发展出高效、可复用的抽象表征。

---

## 架构

```
┌─────────────────────────────────────┐
│           环境环 (Environment Loop)   │
│  产生输入信号 xₜ，提供残差信号       │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      表征池 (Representation Pool)    │
│  存储候选表征，容量N有限             │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│      进化引擎 (Evolution Engine)     │
│  变异-选择-保留 + 新维度诞生          │
└─────────────────────────────────────┘
```

---

## 关键数学量

| 量 | 公式 | 含义 |
|----|------|------|
| 适应度 F | F = -‖x - ŵ‖² | 当前预测能力 |
| 复用频率 R | R = A/t | 跨情境使用率 |
| 持久度 P | P = α·F̄ + β·R - γ·Cost | 综合存活能力 |
| 压缩增益 G | G = (E_old - E_new) / E_old | 新维度价值 |

---

## v5.1 新维度机制 (2026.3)

### 核心机制

v5.1引入了三个互相咬合的机制，实现了表征空间的**自适应扩展**：

#### 1. 新维度诞生

```python
def try_spawn_new_dim(rep, recent_residuals):
    if rep.reuse < threshold:  # 复用阈值
        return False
    
    old_error = mean(abs(residuals))
    new_dim = residual_direction + noise
    new_vector = append(old_vector, new_dim)
    new_error = mean(abs(residuals - new_vector))
    
    gain = (old_error - new_error) / old_error
    
    if gain > min_compression_gain:  # 压缩增益阈值
        accept new dimension
        return True
    return False
```

**参数配置**：
- `spawn_reuse_threshold = 5`（复用次数阈值）
- `min_compression_gain = 0.001`（压缩增益阈值）
- `dim_cost = 1.0`（每加1维扣1单位预算）

#### 2. 残差信号

环境环输出"当前表征联合解释不了的部分"：

```python
def get_input_and_residual(self, pool_prediction):
    x = self.generate_input()
    residual = x - pool_prediction  # 未解释的部分
    return x, residual
```

#### 3. 维度竞争（清理）

定期清理低贡献维度：

```python
def prune_low_contrib_dims():
    for r in pool:
        useless_mask = r.dim_contrib < max_contrib * 0.05
        r.vector[useless_mask] = 0.0  # 置零
```

---

### 测试结果

**300步运行日志**：

```
运行300步...
v 新维度诞生! 压缩增益=0.312, 总维=11
v 新维度诞生! 压缩增益=0.033, 总维=11

结果:
  总维度: 30
  新维度诞生: 2
```

**结论**：300步成功诞生2个新维度，压缩增益分别为0.312和0.033，证明了自发涌现的潜力。

---

## 验证目标

| # | 目标 | 状态 |
|---|------|------|
| 1 | 原型表征的涌现 | ⚠️ 部分通过 |
| 2 | 噪声表征的灾难性遗忘 | ✅ 通过 |
| 3 | 动态平衡 | ⚠️ 部分通过 |
| 4 | 新维度涌现 (v5.1) | ✅ 通过 |

---

## 目录结构

```
fcrs-v5/
├── core.py              # 核心代码 (v5.1)
├── CONCEPT.md          # 概念文档
├── README.md           # 本文件
├── paper/
│   └── PAPER_FULL.md   # 完整论文
└── experiments/
    ├── exp1_emergence.py    # 实验1
    ├── exp2_forgetting.py    # 实验2
    ├── exp3_balance.py       # 实验3
    ├── test_new_dim3.py      # v5.1测试
    └── REPORT_FULL.md        # 实验报告
```

---

## 运行

```bash
# 测试v5.1
py -3.14 core.py

# 运行实验
py -3.14 experiments/test_new_dim3.py
```

---

## 下一步计划

1. [ ] 可视化维度演化
2. [ ] 长时运行测试（2000步）
3. [ ] 噪声环境测试
4. [ ] Ablation对比实验
5. [ ] 原型提取与复用

---

*更新时间: 2026-03-07*
