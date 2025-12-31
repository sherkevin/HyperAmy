# 粒子创建 -> 存储 -> 查询完整流程说明

## 流程概述

完整的粒子生命周期流程包括以下步骤：

```
1. 创建粒子（ParticleEntity，初始状态）
   ↓
2. 送入庞加莱球（存储到数据库）
   ↓
3. 查看双曲空间状态（可选）
   ↓
4. 等待一段时间（模拟时间流逝）
   ↓
5. 查询粒子状态（检索和状态计算）
```

## 详细步骤

### Step 1: 创建粒子（初始状态）

使用 `particle.ParticleEntity` 创建粒子，包含以下属性：

```python
from particle import ParticleEntity
import numpy as np
import time

particle = ParticleEntity(
    entity_id="particle_A",      # 唯一标识
    entity="Entity_A",           # 实体名称
    text_id="text_001",          # 文本 ID（映射关系）
    emotion_vector=np.array([...]),  # 情绪向量（numpy array）
    speed=0.9,                   # 初始速度/强度
    temperature=0.5,             # 初始温度/熵
    born=time.time()             # 生成时间戳
)
```

**注意**：
- `emotion_vector` 是 `numpy.ndarray` 类型
- `speed` 和 `temperature` 是浮点数
- `born` 是绝对时间戳

### Step 2: 送入庞加莱球（存储）

使用 `poincare.HyperAmyStorage` 存储粒子：

```python
from poincare import HyperAmyStorage

storage = HyperAmyStorage(
    persist_path="./hyperamy_db",
    collection_name="emotion_particles"
)

# 单个存储
storage.upsert_entity(entity=particle, links=["neighbor_id_1", "neighbor_id_2"])

# 批量存储
storage.upsert_entities(
    entities=[particle1, particle2, particle3],
    links_map={
        "particle1": ["particle2"],
        "particle2": ["particle1"],
        "particle3": []
    }
)
```

**存储过程**：
1. 向量归一化：将 `emotion_vector` 归一化为单位向量
2. 元数据序列化：将 `speed`, `temperature`, `born`, `links` 等转换为 ChromaDB 支持的格式
3. 写入数据库：通过 `ods.ChromaClient` 写入 ChromaDB

### Step 3: 查看双曲空间状态（可选）

使用 `poincare.ParticleProjector` 计算粒子在双曲空间中的状态：

```python
from poincare import ParticleProjector
import time

projector = ParticleProjector(curvature=1.0, scaling_factor=2.0)

t_now = time.time()
state = projector.compute_state(
    vec=particle.emotion_vector,  # numpy array 或 torch.Tensor
    v=particle.speed,
    T=particle.temperature,
    born=particle.born,
    t_now=t_now
)

# state 包含：
# - current_vector: 庞加莱球坐标 (torch.Tensor)
# - current_v: 当前速度/强度 (float)
# - current_T: 当前温度 (float)
```

**投影过程**：
1. 物理演化：根据时间计算当前的速度和温度（当前实现为恒等映射）
2. 空间投影：将欧式空间的情绪向量和速度映射到庞加莱球坐标

### Step 4: 等待一段时间（模拟时间流逝）

```python
import time

wait_time = 2.0  # 等待 2 秒
time.sleep(wait_time)

t_query = time.time()
elapsed_time = t_query - particle.born
```

**注意**：当前 `TimePhysics` 实现为恒等映射，速度和温度不随时间变化。未来可以实现记忆固化与遗忘曲线。

### Step 5: 查询粒子状态

使用 `poincare.HyperAmyRetrieval` 查询粒子：

```python
from poincare import HyperAmyRetrieval

retrieval = HyperAmyRetrieval(storage, projector)

results = retrieval.search(
    query_entity=particle,      # 查询粒子
    top_k=3,                    # 返回结果数量
    cone_width=50,             # 锥体搜索宽度
    max_neighbors=20,          # 最大邻居数
    neighbor_penalty=1.1       # 邻居惩罚系数
)

# results 是 SearchResult 列表
for result in results:
    print(f"ID: {result.id}")
    print(f"双曲距离: {result.score}")
    print(f"匹配类型: {result.match_type}")  # 'direct' 或 'neighbor'
    print(f"元数据: {result.metadata}")
```

**检索流程**（四步混合检索）：
1. **锥体锁定**：使用向量相似度快速圈定方向一致的粒子
2. **壳层筛选**：计算真实的双曲距离进行精排
3. **邻域激活**：从 Top-K 点出发，扩展其邻居节点
4. **汇总排序**：混合直接检索点和邻居点，最终排序返回

## 完整示例

参见 `test/test_particle_poincare_flow.py` 文件，包含：

1. **test_particle_lifecycle()**: 完整的生命周期测试
2. **test_time_evolution()**: 时间演化验证

## 运行测试

```bash
python -m test.test_particle_poincare_flow
```

## 关键概念

### 双曲距离

- 距离越小，粒子越相似
- 粒子到自己的距离应该接近 0
- 相似情绪和强度的粒子距离较小

### 匹配类型

- `direct`: 通过向量相似度直接检索到的粒子
- `neighbor`: 通过链接关系扩展找到的邻居粒子

### 时间演化

当前实现：
- `TimePhysics.f(v, t_born, t_now) = v`（速度恒定）
- `TimePhysics.g(T, t_born, t_now) = T`（温度恒定）

未来可以扩展为：
- 记忆固化曲线：新记忆随时间增强
- 遗忘曲线：旧记忆随时间衰减

## 架构说明

```
particle.ParticleEntity (欧式空间)
    ↓
poincare.ParticleProjector (欧式 → 双曲转换)
    ↓
ods.ChromaClient (数据库操作)
```

- **particle 模块**：负责构造和查询粒子的状态（欧式空间）
- **poincare 模块**：负责空间转换和检索调用层
- **ods 层**：数据库访问层

