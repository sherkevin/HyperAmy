# System V3 重构计划

## 目标
按照 `system_v3.md` 设计文档，彻底重构 HyperAmy 项目，实现基于庞加莱球的引力时间膨胀与热力学遗忘机制。

---

## 设计核心回顾

### 1. 解耦表征
- **Semantic Direction (μ)**: 语义方向，单位向量 $S^{d-1}$
- **Gravitational Mass (m)**: $m = \alpha \cdot I + \beta \cdot \log(1 + \kappa)$
- **Thermodynamic Temperature (T)**: $T = T_0 / \kappa$

### 2. 初始坐标
- 双曲半径: $R_0 \propto m$
- 庞加莱坐标: $\mathbf{z}_0 = \tanh(\frac{\sqrt{c}}{2} R_0) \cdot \boldsymbol{\mu}$

### 3. 动力学 (引力时间膨胀)
- 衰变公式: $R(t) = R_0 \cdot \exp(-\frac{\gamma}{m} \Delta t)$
- 质量在分母: 质量越大，衰减越慢

### 4. 检索三步走
1. 锥体过滤: $\cos(\mathbf{z}_i, \mathbf{q}) > \eta$
2. 球壳采样: $R_{min} < d_{\mathbb{D}}(\mathbf{0}, \mathbf{z}_i) < R_{max}$
3. 热力学采样: $P \propto \exp(-d_{\mathbb{D}}(\mathbf{q}, \mathbf{z}_i) / T_i)$

---

## 实现步骤

### Phase 1: 基础数学层 (`poincare/math.py`)
**目标**: 实现庞加莱球的基础几何运算

- [ ] 实现保角因子 $\lambda_z = \frac{2}{1 - c\|z\|^2}$
- [ ] 实现双曲距离计算 $d_{\mathbb{D}}(\mathbf{u}, \mathbf{v})$
- [ ] 实现 Möbius 加法 $\mathbf{u} \oplus_c \mathbf{v}$
- [ ] 实现测地线投影
- [ ] 编写测试 `test/test_poincare_math.py`

### Phase 2: 粒子属性计算 (`particle/properties.py`)
**目标**: 实现解耦表征计算

- [ ] 实现情绪强度 $I$ 计算 (基于 emotion vector 的模长)
- [ ] 实现分布纯度 $\kappa$ 计算 (基于 L2/L1 norm ratio)
- [ ] 实现引力质量 $m = \alpha \cdot I + \beta \cdot \log(1 + \kappa)$
- [ ] 实现热力学温度 $T = T_0 / \kappa$
- [ ] 实现初始双曲半径 $R_0 \propto m$
- [ ] 编写测试 `test/test_particle_properties.py`

### Phase 3: 粒子类 (`particle/memory_particle.py`)
**目标**: 重构粒子类，符合 system_v3 设计

- [ ] 定义粒子状态: $(\boldsymbol{\mu}, m, T, R_0, t_0)$
- [ ] 实现初始坐标计算 $\mathbf{z}_0 = \tanh(\frac{\sqrt{c}}{2} R_0) \cdot \boldsymbol{\mu}$
- [ ] 实现位置更新函数 $\mathbf{z}(t) = \tanh(\frac{\sqrt{c}}{2} R_0 e^{-\frac{\gamma}{m}\Delta t}) \cdot \boldsymbol{\mu}$
- [ ] 实现双曲半径查询 $R(t)$
- [ ] 编写测试 `test/test_memory_particle_v3.py`

### Phase 4: 物理引擎 (`poincare/physics.py`)
**目标**: 重构物理层，实现引力时间膨胀

- [ ] 实现 TimeDilation 衰变函数 $R(t) = R_0 \cdot \exp(-\frac{\gamma}{m} \Delta t)$
- [ ] 移除旧的运动学积分逻辑
- [ ] 确保 $O(1)$ 复杂度的位置更新
- [ ] 编写测试 `test/test_physics_v3.py`

### Phase 5: 检索系统 (`poincare/retrieval.py`)
**目标**: 实现三步检索流程

- [ ] Step 1: 锥体过滤 (余弦相似度)
- [ ] Step 2: 球壳采样 (双曲半径范围)
- [ ] Step 3: 热力学邻域 (温度调制)
- [ ] 实现评分公式: $\mathrm{Score} = \frac{1}{d_{hyp} \cdot (1 + \beta/T)}$
- [ ] 编写测试 `test/test_retrieval_v3.py`

### Phase 6: 存储层 (`poincare/storage.py`)
**目标**: 适配新粒子结构

- [ ] 更新存储格式以支持新粒子属性
- [ ] 保持与现有 Chroma 集成
- [ ] 编写测试 `test/test_storage_v3.py`

### Phase 7: 集成与端到端测试
**目标**: 整合所有模块

- [ ] 更新主入口 `particle/__init__.py`
- [ ] 编写端到端测试 `test/test_e2e_v3.py`
- [ ] 性能基准测试

---

## 文件修改清单

### 新建文件
- `poincare/math.py` - 基础数学层
- `particle/properties.py` - 粒子属性计算
- `test/test_poincare_math.py`
- `test/test_particle_properties.py`
- `test/test_memory_particle_v3.py`
- `test/test_physics_v3.py`
- `test/test_retrieval_v3.py`

### 修改文件
- `particle/memory_particle.py` - 完全重写
- `poincare/physics.py` - 重写物理层
- `poincare/retrieval.py` - 重写检索层
- `poincare/storage.py` - 适配新结构

### 废弃文件 (备份后移除)
- `particle/speed.py` - 功能合并到 properties.py
- `particle/thermodynamics.py` - 功能合并到 properties.py
- `particle/gravitational_space.py` - 用新架构替代

---

## 参数配置

```python
# 默认参数
CURVATURE_C = 1.0          # 庞加莱球曲率
GAMMA = 1.0                # 宇宙衰变常数
ALPHA_MASS = 1.0           # 质量公式系数 α
BETA_MASS = 1.0            # 质量公式系数 β
T0 = 1.0                   # 基准温度
RETRIEVAL_BETA = 1.0       # 检索评分系数
```

---

## 测试策略

每个 Phase 完成后立即编写对应测试:
1. **单元测试**: 验证单个函数正确性
2. **可视化测试**: 绘制衰减曲线、空间分布
3. **边界测试**: 测试极限情况 (m→0, m→∞, T→0)
