# H-Mem：Gravitational Time Dilation and Thermodynamic Forgetting in Hyperbolic Memory Spaces
**(H-Mem：双曲记忆空间中的引力时间膨胀与热力学遗忘机制)**

---

## 1. The Geometric Stage: Poincaré Ball (几何舞台)

我们的宇宙是一个 $d$ 维庞加莱球 $(\mathbb{D}_c^d, g^{\mathbb{D}})$，曲率为 $-c$。对于任意记忆粒子 $\mathbf{z} \in \mathbb{D}_c^d$，其黎曼度量张量为保角因子（Conformal Factor）修正后的欧氏度量：

$$
g_{\mathbf{z}}^{\mathbb{D}} = \lambda_{\mathbf{z}}^2 g^E, \quad \text{where } \lambda_{\mathbf{z}} = \frac{2}{1 - c\|\mathbf{z}\|^2}
$$

物理隐喻说明：
- 边界 ($\|\mathbf{z}\| \to 1/\sqrt{c}$)：**"Event Horizon of Clarity"**，代表极致的具体、强烈的 Traumatic Memory。
- 球心 ($\mathbf{z} = \mathbf{0}$)：**"Singularity of Oblivion"**（遗忘奇点/黑洞），所有记忆最终的归宿，信息在此湮灭（熵最大化）。

---

## 2. Initial State: Mass, Temperature, and Position (初始状态)

当一个记忆事件 $e$ 在时刻 $t_0$ 发生时，它被映射为粒子状态 $\mathcal{S}_0 = (\mathbf{z}_0, m, T)$。

### 2.1 Disentangled Representation (解耦表征)
通过编码器获得三个属性：
- **Semantic Direction ($\boldsymbol{\mu} \in S^{d-1}$)：**语义方向
- **Gravitational Mass ($m \in \mathbb{R}^+$)：**由情绪强度 $I$ 和分布纯度 $\kappa$ 决定
  $$
  m = \alpha \cdot I + \beta \cdot \log(1 + \kappa)
  $$
  物理意义：质量越大，抗拒遗忘的能力越强（惯性大）

- **Thermodynamic Temperature ($T \in \mathbb{R}^+$)：**与纯度成反比
  $$
  T = \frac{T_0}{\kappa}
  $$
  物理意义：温度越高，粒子的热运动范围（检索半径）越大

### 2.2 Initial Coordinate (初始坐标)
在庞加莱球中，距离球心的双曲距离 $d_{\mathbb{D}}(\mathbf{0}, \mathbf{z})$ 代表“具体程度”。
- 设定初始双曲半径 $R_0 \propto m$ （质量越大，初始位置离黑洞越远，越清晰）
- 根据庞加莱球的距离公式，其欧氏坐标为：
  $$
  \mathbf{z}_0 = \tanh\left(\frac{\sqrt{c}}{2} R_0\right) \cdot \boldsymbol{\mu}
  $$

---

## 3. Dynamics: Gravitational Time Dilation (动力学：引力时间膨胀)

这是本模型的核心物理定律：粒子受到球心黑洞的引力，沿着测地线（Geodesic）向球心坠落。

### 3.1 The Decay Law (符合 Ebbinghaus 曲线)
规则：“距离越远移动越快”，对应指数衰减动力学，并引入引力时间膨胀效应（大质量物体周围时间流逝变慢，衰减速率变慢）。

$t$ 时刻的双曲半径 $R(t)$ 计算为：
$$
R(t) = R_0 \cdot \exp\left( - \frac{\gamma}{m} \Delta t \right)
$$
其中：
- $\Delta t = t_{now} - t_{created}$：记忆的年龄
- $\gamma$：宇宙衰变常数
- $m$ 的作用 (Time Dilation)：$m$ 位于分母
  - Trauma ($m \to \infty$)：指数项 $\to 0$，$R(t) \approx R_0$，记忆几乎不移动（永不磨灭）
  - Trivia ($m \to 0$)：指数项 $\to -\infty$，$R(t) \to 0$，记忆瞬间落入黑洞

### 3.2 Fast Position Update (快速位置解算)
为满足快速检索需求，需要 $O(1)$ 复杂度的位置更新。
- 在庞加莱球模型，连接原点和任意点的测地线在欧氏空间中是直线，意味着粒子径向收缩，方向 $\boldsymbol{\mu}$ 不变。
- $t$ 时刻的欧氏坐标直接计算，无需微分方程积分：
  $$
  \mathbf{z}(t) = \tanh\left(\frac{\sqrt{c}}{2} \left( R_0 e^{-\frac{\gamma}{m}\Delta t} \right)\right) \cdot \boldsymbol{\mu}
  $$

---

## 4. Retrieval: The Three-Stage Process (检索三步走)

检索不再是简单的 $k$-NN，而是时空-热力学采样过程。查询（Query）视为探测波，探测当前时刻 $t_{now}$ 的粒子分布。

### Stage 1: The Hyperbolic Light Cone (锥形语义过滤)
- 先过滤语义方向。因庞加莱球保角性，可在切空间（Tangent Space）使用 Cosine Similarity：
  $$
  \mathcal{C}_{cone} = \{ \mathbf{z}_i \mid \cos(\mathbf{z}_i, \mathbf{q}) > \eta_{semantic} \}
  $$
- 仅考虑角度，忽略模长

### Stage 2: Temporal Spherical Shells (球壳时序采样)
- 检索“近期”或“远期”记忆，对应不同半径球壳。$R(t)$ 随时间衰减，半径即时间标记：
  $$
  \mathcal{C}_{shell} = \{ \mathbf{z}_i \in \mathcal{C}_{cone} \mid R_{min} < d_{\mathbb{D}}(\mathbf{0}, \mathbf{z}_i(t_{now})) < R_{max} \}
  $$

### Stage 3: Thermodynamic Neighborhood (热力学邻域)
- 粒子温度（辐射范围）：高纯度粒子（低温）如激光，需精确命中才返回；低纯度粒子（高温）如灯泡，轻微误差也能检索到
- 有效检索概率 $P(\text{retrieve} | \mathbf{q}, \mathbf{z}_i)$：
  $$
  P \propto \exp\left( - \frac{d_{\mathbb{D}}(\mathbf{q}, \mathbf{z}_i(t_{now}))}{T_i} \right)
  $$
- $T_i$ 大（模糊记忆）：分母大，对距离不敏感，容易检索；
- $T_i$ 小（清晰记忆）：分母小，距离稍微增加概率就迅速下降

---

## 5. Summary of the Mathematical Algorithm (检索算法总结)

**Algorithm 1: Fast Hyperbolic Gravitational Retrieval**

**Input:** Query $q$, Current Time $t_{now}$, Memory Index $\mathcal{M} = \{(\mathbf{z}_{0,i}, m_i, T_i, t_{0,i})\}$  
**Output:** Ranked Top-$k$ Memories

1. **Semantic Pruning (Conical):**
   - 计算切空间余弦相似度（即归一化向量点积）  
     $$
     S_1 = \{ i \in \mathcal{M} \mid \boldsymbol{\mu}_i^\top \boldsymbol{\mu}_q > \theta \}
     $$

2. **Gravitational Projection ($O(1)$ Update):**
   - 对 $i \in S_1$，计算粒子年龄 $\Delta t = t_{now} - t_{0,i}$
   - 应用 Time Dilation Decay:
     $$
     R_i(t) = R_{0,i} \cdot \exp\left(-\frac{\gamma}{m_i} \Delta t\right)
     $$
   - 更新模长：
     $$
     \|\mathbf{z}_i(t)\| = \tanh\left(\frac{\sqrt{c}}{2} R_i(t)\right)
     $$

3. **Thermal Sampling:**
   - 用 Möbius 加法计算双曲距离 $d_{hyp}(\mathbf{q}, \mathbf{z}_i(t))$：
     $$
     d_{hyp}(\mathbf{u}, \mathbf{v}) = \frac{2}{\sqrt{c}} \text{arctanh}(\sqrt{c} \|-\mathbf{u} \oplus_c \mathbf{v}\|)
     $$
   - 计算热力学得分：
     $$
     \mathrm{Score}_i = \frac{1}{d_{hyp}(\mathbf{q}, \mathbf{z}_i(t)) \cdot (1 + \beta/T_i)}
     $$
     （注：高温减弱“距离”惩罚）

4. **输出：** 按照 $\mathrm{Score}_i$ 排序，取 Top-$k$。

---

## 6. Why this is elegant (为什么这很优雅?)

- **Unified Dynamics：** Gravity 同时解决了 “Time Decay” 与 “Attention Mechanism”，无需额外机制。  
- **Geometry implies Semantics：** Hyperbolic Radius 直接对应 Abstract-Concrete 层级，因双曲空间体积随半径指数增长，边缘能容纳更多具体信息。
- **Efficiency：** 实际计算公式（Step 2）只是简单的指数运算和 Tanh，非常适合 GPU 并行和向量数据库索引。

---
