# HyperAmy & Fusion Framework: Time Evolution Integration

## 架构集成图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Amygdala Workflow Layer                      │
│  (workflow/amygdala.py)                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  1. add(text) → Particle.process()                               │
│     ↓                                                           │
│  2. 生成粒子 (含 purity, tau_v, tau_T)  ✅ 新字段                │
│     ↓                                                           │
│  3. storage.upsert_entities()  ✅ 存储新字段到数据库            │
│     ↓                                                           │
│  4. auto_link_entities()  ✅ 使用新时间演化构建链接           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Storage Layer (poincare/storage.py)           │
├─────────────────────────────────────────────────────────────────┤
│  ✓ upsert_entity() - 存储 purity, tau_v, tau_T                │
│  ✓ upsert_entities() - 批量存储新字段                         │
│  ✓ ChromaDB metadata: {v, T, weight, born, purity, tau_v, tau_T}│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Retrieval Layer (poincare/retrieval.py)         │
├─────────────────────────────────────────────────────────────────┤
│  search()                                                       │
│    ↓                                                             │
│  projector.compute_state(                                      │
│    vec, v, T, born, t_now, weight,                             │
│    tau_v, tau_T, T_min  ✅ 新参数                              │
│  )                                                              │
│    ↓                                                             │
│  TimePhysics.f(v, born, t, tau_v)  ✅ 指数衰减                │
│  TimePhysics.g(T, born, t, tau_T)  ✅ 模拟退火                 │
│    ↓                                                             │
│  distance = ∫v(t)dt  ✅ 精确积分                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 Linking Layer (poincare/linking.py)              │
├─────────────────────────────────────────────────────────────────┤
│  build_hyperbolic_links()                                      │
│    ↓                                                             │
│  projector.compute_state(..., tau_v, tau_T, T_min)  ✅       │
│    ↓                                                             │
│  计算双曲距离（考虑时间演化）                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 关键集成点

### 1. Amygdala 工作流 ✅
```python
# workflow/amygdala.py:232
self.particle_projector = ParticleProjector(
    curvature=1.0,
    scaling_factor=2.0,
    max_radius=10000.0  # 允许粒子存在约5.5小时
)
```

### 2. 粒子创建 ✅
```python
# particle/particle.py:241-242
tau_v = self.tau_base * (1.0 + self.gamma * purity)
tau_T = self.tau_base * (1.0 + self.beta * purity)

particle = ParticleEntity(
    ...,
    purity=purity,
    tau_v=tau_v,
    tau_T=tau_T
)
```

### 3. 存储层 ✅
```python
# poincare/storage.py:65-67
metadata = {
    "v": float(entity.speed),
    "T": float(entity.temperature),
    ...
    "purity": float(entity.purity),
    "tau_v": float(entity.tau_v),
    "tau_T": float(entity.tau_T)
}
```

### 4. 检索层 ✅
```python
# poincare/retrieval.py:90-103
tau_v = cand_meta.get('tau_v', 86400.0)  # 向后兼容
tau_T = cand_meta.get('tau_T', 86400.0)
T_min = 0.1

dynamic_cand = self.projector.compute_state(
    ...,
    tau_v=tau_v,      # ✅ 新参数
    tau_T=tau_T,      # ✅ 新参数
    T_min=T_min       # ✅ 新参数
)
```

## Fusion 框架使用情况

### Fusion 检索文件
- `workflow/fusion_retrieval.py` - Amygdala + HippoRAG 级联
- `workflow/graph_fusion_retrieval.py` - 实体级融合

### 使用方式
```
Fusion → Amygdala → Particle Module → Storage/Retrieval/Physics
                                     ↓
                              Time Evolution ✅
```

## 向后兼容性

### 旧数据处理 ✅
```python
# 使用 .get() 提供默认值
tau_v = cand_meta.get('tau_v', 86400.0)  # 旧数据用默认值
tau_T = cand_meta.get('tau_T', 86400.0)
```

### 新数据特性 ✅
- 高纯度粒子 → tau_v 大 → 衰减慢 → 记忆久
- 低纯度粒子 → tau_v 小 → 衰减快 → 遗忘快

## 实际数据流示例

### 创建阶段
```python
text = "量子力学是描述微观世界的物理理论"

↓ Particle.process()

粒子 = {
    'entity': '量子力学',
    'speed': 35.31,
    'temperature': 0.55,
    'purity': 0.5,        # ✅ 新增
    'tau_v': 172800.0,    # ✅ 新增 (2天)
    'tau_T': 129600.0,    # ✅ 新增 (1.5天)
    'born': 1735688800
}

↓ storage.upsert_entity()

ChromaDB metadata 包含所有字段 ✅
```

### 检索阶段
```python
查询: "什么是量子力学?"

↓ 生成查询粒子

↓ projector.compute_state(查询粒子, tau_v, tau_T)

  计算 v(t) = v₀ · exp(-t/τv)
  计算 T(t) = T_min + (T₀ - T_min) · exp(-t/τT)
  计算距离 = ∫v(t)dt (精确积分)

↓ 返回动态调整后的结果
```

## 验证状态

| 组件 | 状态 | 说明 |
|------|------|------|
| `particle/particle.py` | ✅ 已更新 | 生成新字段 |
| `poincare/storage.py` | ✅ 已更新 | 存储新字段 |
| `poincare/retrieval.py` | ✅ 已更新 | 使用新参数 |
| `poincare/linking.py` | ✅ 已更新 | 使用新参数 |
| `poincare/physics.py` | ✅ 已更新 | 精确积分 |
| `test/test_particle_poincare_flow.py` | ✅ 已更新 | 测试通过 |
| `workflow/amygdala.py` | ✅ 使用中 | Fusion框架 |
| `workflow/fusion_retrieval.py` | ✅ 使用中 | Fusion检索 |

## 总结

✅ **新方法已完全集成到 HyperAmy 和 Fusion 框架**

**证据链**：
1. Particle 生成 → 创建 purity, tau_v, tau_T
2. Storage 存储 → 保存到 ChromaDB
3. Retrieval 检索 → 使用 compute_state(tau_v, tau_T)
4. Linking 链接 → 使用 compute_state(tau_v, tau_T)
5. Physics 计算 → 精确积分距离

**无遗漏**：
- ✅ 所有粒子创建路径
- ✅ 所有检索路径
- ✅ 所有链接构建路径
- ✅ 所有测试文件

**向后兼容**：
- ✅ 旧数据使用默认值 (tau=86400)
- ✅ 新数据使用计算值 (tau=base×(1+γ·purity))
