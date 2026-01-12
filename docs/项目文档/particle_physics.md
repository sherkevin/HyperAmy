# 粒子物理系统设计文档

**版本**: v3.1 - Soft Label 强度模型
**日期**: 2026-01-12
**状态**: 已实现

---

## 目录

1. [概述](#1-概述)
2. [核心设计理念](#2-核心设计理念)
3. [情绪向量处理架构](#3-情绪向量处理架构)
4. [物质状态理论](#4-物质状态理论)
5. [时间动力学](#5-时间动力学)
6. [自由能原理](#6-自由能原理)
7. [数学推导](#7-数学推导)
8. [实现细节](#8-实现细节)
9. [参数配置](#9-参数配置)

---

## 1. 概述

HyperAmy 的粒子物理系统模拟记忆在双曲空间中的演化过程。核心思想是：

> **记忆像物质一样存在三种状态——气态、液态、固态。强情绪形成的记忆像固态晶体，衰减极慢；弱情绪形成的记忆像气体，快速消散。**

### 1.1 关键特性

| 特性 | 说明 |
|-----|------|
| **解耦架构** | 情绪向量获取与后续处理完全分离 |
| **Soft Label 强度** | 模长由情绪强度决定，而非 embedding 模长 |
| **引力时间膨胀** | 质量越大，时间流逝越慢，记忆衰减越慢 |
| **三态模型** | 气态、液态、固态对应不同衰减特性 |

### 1.2 物理隐喻

```
强情绪 → 高质量 → 引力时间膨胀 → 衰减慢 → 固态记忆
弱情绪 → 低质量 → 正常时间流逝 → 衰减快 → 气态记忆
```

---

## 2. 核心设计理念

### 2.1 解耦架构

传统架构的问题：情绪向量的获取（LLM/模型）与物理计算耦合在一起。

**新架构**：完全解耦

```
┌─────────────────────────────────────────────────────────────┐
│  情绪向量获取层（可替换）                                    │
│  - LLM + Embedding API（当前）                               │
│  - 专门的情绪向量模型（未来，64/128维）                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ 输出：e_raw
┌─────────────────────────────────────────────────────────────┐
│  强度分类层（EmotionClassifier）                              │
│  文本描述 → Soft Label → I_raw                                │
└─────────────────────────────────────────────────────────────┘
                            ↓ 输出：I_raw
┌─────────────────────────────────────────────────────────────┐
│  向量处理层（EmotionVectorProcessor）                         │
│  方向：μ = e_raw / ||e_raw||                                 │
│  模长：κ = 1.0 + α × I_raw                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓ 输出：μ, κ
┌─────────────────────────────────────────────────────────────┐
│  物理计算层（PhysicsEngine）                                   │
│  时间膨胀、Poincaré 投影、遗忘判断                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 模长公式的物理意义

$$
\kappa = 1.0 + \alpha \times I_{raw}
$$

| I_raw | κ | 状态 | 物理意义 |
|-------|---|------|----------|
| 0.0 | 1.0 | - | 基准（无情绪）|
| 0.1 | 6.0 | 气态 | 微弱情绪，快速遗忘 |
| 0.5 | 26.0 | 液态 | 中等情绪，正常遗忘 |
| 0.9 | 46.0 | 固态 | 强烈情绪，类似"创伤" |

---

## 3. 情绪向量处理架构

### 3.1 数据流

```
原始文本
    ↓
[Entity Extraction] → 实体列表
    ↓
[Sentence Generation] → 情感描述 "enthusiasm, joy, hope"
    ↓
[EmotionClassifier] → Soft Label {joy: 0.6, neutral: 0.3, ...}
    ↓
[I_raw = max_{c≠neutral}(y_c)] → 0.6
    ↓
[Emotion Vector Model] → e_raw (128维向量)
    ↓
[EmotionVectorProcessor]
    → μ = e_raw / ||e_raw|| (方向)
    → κ = 1.0 + 50 × 0.6 = 31.0 (模长)
    ↓
[ParticleEntity] → μ + κ + 其他热力学属性
```

### 3.2 情绪分类器

**输入**：情感描述文本（如 "enthusiasm, contentment, camaraderie"）

**输出**：Soft Label 分布

```python
MAIN_CATEGORIES = [
    "neutral", "joy", "sadness", "anger", "fear",
    "surprise", "disgust", "love", "hate", "interest"
]
```

**强度计算**：
$$
I_{raw} = \max_{c \neq \text{neutral}} (y_c)
$$

### 3.3 向量处理器

```python
class EmotionVectorProcessor:
    def process(self, raw_vector, intensity):
        # 方向：归一化
        direction = raw_vector / np.linalg.norm(raw_vector)

        # 模长：基于强度
        modulus = 1.0 + self.alpha * intensity

        return ProcessedEmotionVector(
            direction=direction,
            modulus=modulus,
            intensity=intensity,
            state=self.compute_state(modulus)
        )
```

---

## 4. 物质状态理论

### 4.1 状态定义

| 状态 | κ 范围 | 初始半径 R₀ | 衰减特性 |
|-----|--------|-------------|----------|
| **气态** | 1.0 - 10.0 | 2 - 20 | 快速衰减 |
| **液态** | 10.0 - 40.0 | 20 - 80 | 中等衰减 |
| **固态** | 40.0 - 51.0 | 80 - 102 | 缓慢衰减 |

### 4.2 状态转换

```
高强度事件 (I_raw ↑)
    ↓
κ ↑ → R₀ ↑ → 初始位置离球心更远
    ↓
衰减时间 ∝ m → 高质量粒子衰减慢
    ↓
长期记忆形成（固态）
```

### 4.3 遗忘时间

$$
t_{forget} = \frac{m}{\gamma} \cdot \ln\left(\frac{R_0}{\epsilon}\right)
$$

其中 $m = \kappa$ 是质量/模长，$\epsilon = 10^{-3}$ 是遗忘阈值。

| 状态 | κ | t_forget (约) |
|-----|---|----------------|
| 气态 (I=0.1) | 6 | 数小时 |
| 液态 (I=0.5) | 26 | 数天 |
| 固态 (I=0.9) | 46 | 数周 |

---

## 5. 时间动力学

### 5.1 引力时间膨胀衰变

**核心公式**：

$$
R(t) = R_0 \cdot \exp\left(-\frac{\gamma}{m} \cdot \Delta t\right)
$$

其中：
- $R(t)$：时刻 $t$ 的双曲半径
- $R_0$：初始双曲半径 = $2.0 \times \kappa$
- $\gamma$：宇宙衰变常数 = 1.0
- $m$：粒子质量 = $\kappa$（模长）
- $\Delta t = t - t_0$

### 5.2 记忆强度

$$
\text{strength}(t) = \frac{R(t)}{R_0} = \exp\left(-\frac{\gamma}{m} \cdot \Delta t\right)
$$

### 5.3 物理图像

```
        R (双曲半径)
        ↑
   R₀ ─┼────────────────→ ● 高质量粒子
        │ \
        │  \            衰减慢（时间膨胀）
        │   \
        │    \
        │     \
        │      \
        │       \
        │        \
        │         \
        │          \
        │           \
        │            \
        │             ● 低质量粒子
        │              ↓ 衰减快
        └────────────────→ t (时间)
```

---

## 6. 自由能原理

### 6.1 Helmholtz 自由能

$$
\mathcal{F}(\mathbf{e}) = E(\mathbf{e}) - T \cdot S(\mathbf{e})
$$

其中：
- $E$：能量（情绪强度）
- $T$：温度（由纯度决定）
- $S$：熵（不确定性）

### 6.2 温度-纯度关系

$$
T(\mathbf{e}) = T_{\min} + (T_{\max} - T_{\min}) \cdot (1 - \mathcal{P}(\mathbf{e}))
$$

其中纯度：
$$
\mathcal{P}(\mathbf{e}) = \frac{\|\mathbf{e}|_2^2}{|\mathbf{e}|_1^2}
$$

- 高纯度 → 低温（有序态）
- 低纯度 → 高温（无序态）

### 6.3 速度-纯度关系

$$
v = |\mathbf{e}| \cdot (1 + \alpha_{\text{speed}} \cdot \mathcal{P}_{\text{norm}})
$$

现在速度直接等于模长：
$$
v = \kappa = 1.0 + \alpha \times I_{raw}
$$

---

## 7. 数学推导

### 7.1 模长公式的推导

**目标**：建立情绪强度与模长的映射关系

**约束**：
1. 最小模长为 1.0（基准）
2. 最大模长约为 50.0（强情绪）
3. 映射应该是线性的（简单可控）

**推导**：

设 $I_{raw} \in [0, 1]$ 为情绪强度，则：

$$
\kappa = \kappa_{\min} + \alpha \cdot I_{raw}
$$

取 $\kappa_{\min} = 1.0$，$\alpha = 50.0$：

$$
\boxed{\kappa = 1.0 + 50.0 \times I_{raw}}
$$

验证：
- $I_{raw} = 0$ → $\kappa = 1.0$（无情绪）
- $I_{raw} = 0.1$ → $\kappa = 6.0$（气态）
- $I_{raw} = 0.5$ → $\kappa = 26.0$（液态）
- $I_{raw} = 1.0$ → $\kappa = 51.0$（固态）

### 7.2 Poincaré 坐标投影

$$
\mathbf{z}(t) = \tanh\left(\sqrt{\frac{c}{2}} \cdot R(t)\right) \cdot \boldsymbol{\mu}
$$

其中：
- $\mathbf{z}(t)$：Poincaré 坐标（用于存储）
- $c$：空间曲率（默认 1.0）
- $R(t)$：当前双曲半径
- $\boldsymbol{\mu}$：归一化方向向量

### 7.3 衰减公式的推导

**假设**：粒子沿径向向球心坠落（测地线）

**牛顿冷却定律类比**：
$$
\frac{dR}{dt} = -k \cdot R$$

解为指数衰减：
$$
R(t) = R_0 \cdot e^{-kt}
$$

**引入引力时间膨胀**（广义相对论）：
- 质量在引力场中改变时间流逝速度
- 衰减常数与质量成反比：$k = \gamma / m$

**最终公式**：
$$
\boxed{R(t) = R_0 \cdot \exp\left(-\frac{\gamma}{m} \cdot t\right)}
$$

---

## 8. 实现细节

### 8.1 核心类结构

```python
# 情绪分类器
class EmotionClassifier:
    def get_intensity(text: str) -> float:
        """返回 I_raw ∈ [0, 1]"""

# 向量处理器
class EmotionVectorProcessor:
    def process(raw_vector, intensity):
        """返回 direction μ 和 modulus κ"""

# 情绪节点
@dataclass
class EmotionNode:
    entity_id: str
    entity: str
    emotion_vector: np.ndarray  # 原始向量（未归一化）
    intensity: float  # I_raw
    raw_description: str

# 粒子实体
@dataclass
class ParticleEntity:
    entity_id: str
    entity: str
    emotion_vector: np.ndarray  # 归一化方向 μ
    weight: float  # 模长 κ（作为质量）
    speed: float  # = κ
    temperature: float
    purity: float
    tau_v: float
    tau_T: float
    born: float
    intensity: float  # I_raw
```

### 8.2 文件结构

```
particle/
├── emotion_classifier.py       # Soft Label 分类器
├── emotion_vector_processor.py  # 向量处理器（解耦核心）
├── emotion_v2.py                 # 情绪向量获取
├── particle.py                   # 粒子聚合类
├── purity.py                     # 纯度计算
├── speed.py                      # 速度计算（已弃用）
├── temperature.py                # 温度计算
└── thermodynamics.py             # 热力学模块（未来）

poincare/
├── physics.py                    # 物理引擎（时间膨胀）
├── projector.py                  # Poincaré 投影
├── math.py                       # 数学工具
└── storage.py                    # 向量存储
```

### 8.3 关键代码片段

**初始化粒子**：
```python
# 获取原始向量和强度
emotion_nodes = self.emotion_v2.process(text, text_id, entities)

# 处理向量（解耦）
processed = self.vector_processor.process_batch(
    raw_vectors=[node.emotion_vector for node in emotion_nodes],
    intensities=[node.intensity for node in emotion_nodes]
)

# 创建粒子
particle = ParticleEntity(
    emotion_vector=processed.direction,  # μ
    weight=processed.modulus,          # κ
    speed=processed.modulus,           # v = κ
    intensity=processed.intensity      # I_raw
)
```

---

## 9. 参数配置

### 9.1 物理参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `intensity_alpha` | 50.0 | 模长系数 κ = 1.0 + α × I_raw |
| `min_modulus` | 1.0 | 最小模长 |
| `gamma` | 1.0 | 宇宙衰变常数 |
| `curvature` | 1.0 | Poincaré 球曲率 c |
| `scaling_factor` | 2.0 | 初始半径缩放因子 |
| `forgetting_threshold` | 1e-3 | 遗忘阈值 |

### 9.2 状态阈值

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `STATE_GAS_MAX` | 10.0 | 气态最大模长 |
| `STATE_SOLID_MIN` | 40.0 | 固态最小模长 |

### 9.3 热力学参数

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `T_min` | 0.1 | 最小温度（有序态）|
| `T_max` | 1.0 | 最大温度（无序态）|
| `alpha` | 0.5 | 纯度影响系数（旧）|
| `tau_base` | 86400.0 | 基准时间常数（1天）|
| `beta` | 1.0 | 温度冷却系数 |
| `gamma` | 2.0 | 速度衰减系数 |

---

## 附录A：公式速查

### A.1 核心公式

$$
\begin{align}
\text{强度:}\quad & I_{raw} = \max_{c \neq \text{neutral}} (y_c) \\
\text{模长:}\quad & \kappa = 1.0 + \alpha \times I_{raw} \\
\text{方向:}\quad & \boldsymbol{\mu} = \frac{\mathbf{e}_{raw}}{|\mathbf{e}_{raw}|} \\
\text{初始半径:}\quad & R_0 = \text{scaling\_factor} \times \kappa \\
\text{时间演化:}\quad & R(t) = R_0 \cdot \exp\left(-\frac{\gamma}{m} \cdot t\right) \\
\text{记忆强度:}\quad & S(t) = \frac{R(t)}{R_0} = \exp\left(-\frac{\gamma}{m} \cdot t\right) \\
\text{Poincaré坐标:}\quad & \mathbf{z}(t) = \tanh\left(\sqrt{\frac{c}{2}} \cdot R(t)\right) \cdot \boldsymbol{\mu}
\end{align}
$$

### A.2 状态判据

$$
\text{state} =
\begin{cases}
\text{气态} & \kappa < 10 \\
\text{液态} & 10 \leq \kappa < 40 \\
\text{固态} & \kappa \geq 40
\end{cases}
$$

---

## 附录B：版本历史

| 版本 | 日期 | 变更 |
|-----|------|------|
| v3.1 | 2026-01-12 | 引入 Soft Label 强度模型，解耦向量处理 |
| v3.0 | 2026-01-05 | 自由能原理，统计力学基础 |
| v2.0 | 2025-12-xx | 引力时间膨胀，双曲几何 |
| v1.0 | 2025-xx-xx | 初始版本 |

---

## 附录C：参考文献

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
2. Nickel & Kiela (2017). Poincaré Embeddings for Learning Hierarchical Representations. *NeurIPS*.
3. Chamberlain et al. (2020). GRaph Embeddings with Molecular Hierarchy Preservation. *ICML*.
