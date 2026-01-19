# HyperAmy 与 Fusion 架构检索机制详解

## 目录
- [一、HyperAmy 架构检索逻辑](#一hyperamy-架构检索逻辑)
- [二、Fusion 架构检索逻辑](#二fusion-架构检索逻辑)
- [三、架构对比分析](#三架构对比分析)
- [四、使用建议](#四使用建议)

---

## 一、HyperAmy 架构检索逻辑

HyperAmy 是基于**双曲空间物理引擎**的检索系统，通过四步混合检索流水线实现精准召回。

### 1.1 整体流程

```
Query Text
    ↓
转换为 Query Particle
    ↓
┌─────────────────────────────────────────────────────────┐
│ HyperAmy 四步混合检索                                    │
├─────────────────────────────────────────────────────────┤
│ Step 1: 锥体锁定 - 欧式空间向量相似度快速筛选            │
│ Step 2: 壳层筛选 - 双曲空间距离精排                      │
│ Step 3: 邻域激活 - 粒子链接扩展                          │
│ Step 4: 汇总排序 - 混合排序返回 Top-K                    │
└─────────────────────────────────────────────────────────┘
    ↓
返回检索结果（按双曲距离排序）
```

### 1.2 核心代码实现

**入口**：`poincare/retrieval.py:116`

```python
def search(self,
           query_entity: 'ParticleEntity',
           top_k: int = 3,
           cone_width: int = 50,
           max_neighbors: int = 20,
           neighbor_penalty: float = 1.1) -> List[SearchResult]:
```

### 1.3 详细步骤解析

#### Step 1: 锥体锁定（Cone Locking）

**位置**：`poincare/retrieval.py:165-181`

```python
# 归一化查询向量
query_vec_list = ChromaClient.normalize_vector(query_entity.emotion_vector)

# 欧式空间向量相似度查询
results = self.storage.ods_client.query(
    query_embeddings=[query_vec_list],
    n_results=cone_width,  # 默认 50
    include=["metadatas", "embeddings"]
)
```

**关键点**：
- 使用欧式空间的余弦相似度快速筛选
- `cone_width` 控制筛选范围（建议 50-100）
- 快速圈定方向一致的粒子，缩小搜索空间

#### Step 2: 壳层筛选（Shell Filtering）

**位置**：`poincare/retrieval.py:183-210`

```python
# 计算每个候选的真实双曲距离
for pid, meta, vec in zip(ids, metas, vecs):
    score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
    scored_candidates.append(SearchResult(
        id=pid,
        score=score,  # 双曲距离，越小越相似
        metadata=meta,
        vector=vec,
        match_type='direct'
    ))
```

**关键点**：
- 将粒子投影到庞加莱球（双曲空间）
- 计算**真实双曲距离**：

$$d(u,v) = \text{acosh}\left(1 + \frac{2\|u-v\|^2}{(1-\|u\|^2)(1-\|v\|^2)}\right)$$

- 考虑粒子的**时间演化**：
  - **速度衰减**：$v(t) = v_0 \cdot e^{-\Delta t/\tau_v}$
  - **温度冷却**：$T(t) = T_{\min} + (T_0 - T_{\min}) \cdot e^{-\Delta t/\tau_T}$
  - **距离积分**：$\text{distance} = \text{initial} + \int_0^{\Delta t} v(s) ds$

**精确积分公式**（`poincare/physics.py:153`）：
```python
# 速度的精确积分（解析解）
integrated_distance = v * tau_v * (1.0 - math.exp(-dt / tau_v))

# 当前总距离
current_distance = initial_distance + integrated_distance
```

#### Step 3: 邻域激活（Neighborhood Activation）

**位置**：`poincare/retrieval.py:212-264`

```python
# 从 Top-K 粒子扩展邻居
for cand in top_candidates:
    links = json.loads(cand.metadata.get('links', '[]'))
    neighbor_ids.extend(links)

# 获取邻居节点并计算距离
for nid in neighbor_ids:
    score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
    score *= neighbor_penalty  # 应用惩罚系数（默认 1.1）
```

**关键点**：
- 从粒子的 `links` 字段获取邻居节点
- 应用**邻居惩罚系数**（默认 1.1，距离放大 10%）
- **性能保护**：
  - 单节点扩展限制：`max_neighbors=20`
  - 全局扩展限制：`max_neighbors * top_k`

#### Step 4: 汇总排序（Final Aggregation）

**位置**：`poincare/retrieval.py:266-268`

```python
# 混合直接检索粒子和邻居粒子
top_candidates.sort(key=lambda x: x.score)
return top_candidates[:top_k]
```

**关键点**：
- 将直接检索（`match_type='direct'`）和邻居扩展（`match_type='neighbor'`）混合排序
- 返回 Top-K 结果

### 1.4 核心优势

| 特性 | 说明 |
|------|------|
| **时间感知** | 考虑粒子随时间的演化（速度衰减、温度冷却） |
| **层次结构** | 双曲空间天然具有层次性，适合表示语义层级 |
| **邻域扩展** | 利用粒子间的链接关系，提升召回率 |
| **物理引擎** | 基于自由能原理，模拟真实物理过程 |

---

## 二、Fusion 架构检索逻辑

Fusion 架构提供两种融合方式，分别适用于不同场景。

### 2.1 方式 1：级联检索（FusionRetriever）

#### 2.1.1 整体流程

```
Query
    ↓
┌────────────────────────────────────────┐
│ HippoRAG 快速筛选（Top-K 候选）        │
│ - 图谱结构检索                         │
│ - 语义相似度匹配                       │
└────────────────────────────────────────┘
    ↓
提取候选 chunks 文本
    ↓
┌────────────────────────────────────────┐
│ Amygdala 深度精排（Top-M 结果）        │
│ - 四步混合检索                         │
│ - 双曲空间距离计算                     │
└────────────────────────────────────────┘
    ↓
融合结果（按 Amygdala 分数排序）
```

#### 2.1.2 核心代码实现

**入口**：`workflow/fusion_retrieval.py:176`

```python
def _retrieve_cascade(
    self,
    query: str,
    hipporag_top_k: int,  # HippoRAG 返回候选数（默认 20）
    amygdala_top_k: int   # 最终返回数（默认 5）
) -> List[Dict[str, Any]]:
```

#### 2.1.3 详细步骤

**Step 1: HippoRAG 快速筛选**（`workflow/fusion_retrieval.py:203-209`）

```python
hipporag_results = self.hipporag.retrieve(
    query=query,
    top_k=hipporag_top_k  # 默认 20
)
```

**Step 2: 提取候选文本**（`workflow/fusion_retrieval.py:215-217`）

```python
candidate_texts = [result['text'] for result in hipporag_results]
```

**Step 3: Amygdala 在候选中精排**（`workflow/fusion_retrieval.py:220-226`）

```python
amygdala_results = self.amygda.retrieval(
    query_text=query,
    retrieval_mode="chunk",
    top_k=len(candidate_texts)  # 获取所有候选的排名
)
```

**Step 4: 融合结果**（`workflow/fusion_retrieval.py:229-257`）

```python
# 创建文本到 Amygdala 得分的映射
amygdala_scores = {}
for result in amygdala_results:
    amygdala_scores[result['text']] = result['score']

# 筛选并重新排序
fusion_results = []
for hipporag_result in hipporag_results:
    text = hipporag_result['text']
    if text in amygdala_scores:
        fusion_results.append({
            'text': text,
            'hipporag_score': hipporag_result['score'],
            'amygdala_score': amygdala_scores[text],
            'fusion_score': amygdala_scores[text],  # 使用 Amygdala 分数排序
            'rank': len(fusion_results) + 1
        })

# 按 Amygdala 分数排序
fusion_results.sort(key=lambda x: x['amygdala_score'], reverse=True)
return fusion_results[:amygdala_top_k]
```

#### 2.1.4 核心优势

| 特性 | 说明 |
|------|------|
| **速度快** | HippoRAG 快速缩小范围，减少 Amygdala 计算量 |
| **质量高** | Amygdala 在小范围内深度精排，保证结果质量 |
| **平衡性好** | 兼顾速度和质量，适合大多数场景 |
| **默认推荐** | README 标注为"推荐"方式 |

#### 2.1.5 其他模式

**并行检索**（`workflow/fusion_retrieval.py:267`）：
```python
results = fusion.retrieve(
    query="your query",
    mode="parallel"  # 并行检索 + 分数融合
)
```

- 同时使用两个系统检索
- 归一化分数后加权融合
- 速度更快，但质量略低于级联

**单独使用**：
```python
# 仅 HippoRAG
results = fusion.retrieve(query="...", mode="hipporag_only")

# 仅 Amygdala
results = fusion.retrieve(query="...", mode="amygdala_only")
```

---

### 2.2 方式 2：图谱融合检索（GraphFusionRetriever）

#### 2.2.1 整体流程

```
Query
    ↓
Step 1: 抽取实体（使用 Amygdala）
    ↓
Step 2: HippoRAG 语义扩展 → semantic_entities
    ↓
Step 3: Amygdala 情绪扩展 → emotion_entities
    ↓
Step 4: HippoRAG fact 扩展 → fact_entities
    ↓
Step 5: 映射到 HippoRAG 实体空间 + 融合权重
    ↓
Step 6: PPR 传播（Personalized PageRank）
    ↓
Step 7: 返回排序后的 chunks
```

#### 2.2.2 核心代码实现

**入口**：`workflow/graph_fusion_retrieval.py:160`

```python
def retrieve(
    self,
    query: str,
    top_k: int = 5,
    emotion_weight: float = 0.3,    # Amygdala 情绪权重
    semantic_weight: float = 0.5,   # HippoRAG 语义权重
    fact_weight: float = 0.2,       # HippoRAG fact 权重
    linking_top_k: int = 20,
    passage_node_weight: float = 0.05
) -> List[Dict[str, Any]]:
```

#### 2.2.3 详细步骤

**Step 1: 抽取实体**（`workflow/graph_fusion_retrieval.py:196-208`）

```python
# 使用 Amygdala 的 particle 模块处理 query
query_particles = self.amygda.particle.process(
    text=query,
    text_id=f"query_{int(time.time())}"
)

query_entities = [p.entity for p in query_particles]
```

**Step 2: HippoRAG 语义扩展**（`workflow/graph_fusion_retrieval.py:274-300`）

```python
def _semantic_expansion(self, query_entities: List[str], top_k: int = 20):
    """
    在 HippoRAG entity_embedding_store 中找相似实体
    """
    semantic_entities = defaultdict(float)

    for entity in query_entities:
        entity_id = compute_mdhash_id(content=entity.lower(), prefix="entity-")
        # 使用 embedding 余弦相似度找相似实体
        similarities = retrieve_knn(
            query_ids=[entity_id],
            key_ids=entity_node_keys,
            query_vecs=query_embeddings,
            key_vecs=entity_embeddings,
            k=top_k
        )
        # 累积相似度分数
        for key_id, similarity in zip(..., ...):
            semantic_entities[key_id] += similarity

    return semantic_entities
```

**Step 3: Amygdala 情绪扩展**（`workflow/graph_fusion_retrieval.py:226-230`）

```python
def _emotion_expansion(self, query_particle, top_k: int = 20):
    """
    使用 query 粒子在 Amygdala 中检索
    """
    # 使用 HyperAmyRetrieval 搜索
    search_results = retriever.search(
        query_entity=query_particle,
        top_k=top_k,
        cone_width=top_k
    )

    # 提取实体及其双曲距离（转换为相似度）
    emotion_entities = {}
    for result in search_results:
        entity = result.metadata.get('entity', '')
        # 双曲距离越小越相似，转换为权重
        similarity = 1.0 / (1.0 + result.score)
        emotion_entities[entity] = similarity

    return emotion_entities
```

**Step 4: HippoRAG fact 扩展**（`workflow/graph_fusion_retrieval.py:234-238`）

```python
def _extract_fact_entities(self, query: str, top_k: int = 20):
    """
    提取 HippoRAG 的 fact 相关实体
    """
    # 使用 HippoRAG 的 dense passage retrieval
    fact_scores = self._hipporag_core.get_fact_scores(query)

    # 返回 Top-K fact 相关的实体
    top_indices = np.argsort(fact_scores)[::-1][:top_k]
    fact_entities = {fact_entity_keys[i]: fact_scores[i] for i in top_indices}

    return fact_entities
```

**Step 5: 融合权重**（`workflow/graph_fusion_retrieval.py:240-251`）

```python
def _merge_entity_weights(self, query_entities, semantic_entities,
                         emotion_entities, fact_entities,
                         emotion_weight, semantic_weight, fact_weight):
    """
    融合三种扩展的实体权重
    """
    entity_weights = defaultdict(float)

    # 原始 query 实体（最高权重）
    for entity in query_entities:
        entity_weights[entity] += 1.0

    # 语义扩展实体
    for entity, score in semantic_entities.items():
        entity_weights[entity] += semantic_weight * score

    # 情绪扩展实体
    for entity, score in emotion_entities.items():
        entity_weights[entity] += emotion_weight * score

    # Fact 扩展实体
    for entity, score in fact_entities.items():
        entity_weights[entity] += fact_weight * score

    return entity_weights
```

**Step 6: PPR 传播**（`workflow/graph_fusion_retrieval.py:254-259`）

```python
def _run_ppr_with_entity_weights(self, entity_weights, passage_node_weight):
    """
    在 HippoRAG 图谱上运行 Personalized PageRank
    """
    # 创建重启向量（restart vector）
    restart_vector = np.zeros(num_entities)
    for entity, weight in entity_weights.items():
        entity_idx = entity_to_idx[entity]
        restart_vector[entity_idx] = weight

    # 运行 PPR
    ppr_scores = self._hipporag_core.run_ppr(
        restart_vector=restart_vector,
        passage_node_weight=passage_node_weight
    )

    # 返回排序后的文档 IDs
    sorted_doc_ids = np.argsort(ppr_scores)[::-1]
    return sorted_doc_ids, ppr_scores
```

**Step 7: 格式化结果**（`workflow/graph_fusion_retrieval.py:262-266`）

```python
results = [
    {
        'rank': i + 1,
        'doc_id': doc_id,
        'text': doc_texts[doc_id],
        'score': ppr_scores[doc_id]
    }
    for i, doc_id in enumerate(sorted_doc_ids[:top_k])
]
```

#### 2.2.4 核心优势

| 特性 | 说明 |
|------|------|
| **实体级融合** | 在统一的 HippoRAG 实体空间中融合多种信号 |
| **多信号融合** | 语义（0.5）+ 情绪（0.3）+ Fact（0.2） |
| **图谱传播** | PPR 考虑全局图谱结构，不仅是局部相似度 |
| **质量最高** | 三种信号互补，检索质量最优 |
| **可调权重** | 根据场景调整三种信号权重 |

#### 2.2.5 权重调优建议

```python
# 默认权重（平衡）
emotion_weight=0.3, semantic_weight=0.5, fact_weight=0.2

# 侧重语义理解
emotion_weight=0.2, semantic_weight=0.7, fact_weight=0.1

# 侧重情绪感知
emotion_weight=0.6, semantic_weight=0.2, fact_weight=0.2

# 侧重事实问答
emotion_weight=0.1, semantic_weight=0.3, fact_weight=0.6
```

---

## 三、架构对比分析

### 3.1 功能对比

| 维度 | HyperAmy | Fusion 级联 | Fusion 图谱 |
|------|----------|-------------|-------------|
| **检索空间** | 双曲空间 | 欧式 + 双曲 | 图谱 |
| **相似度度量** | 双曲距离 | DPR + 双曲距离 | PPR 分数 |
| **时间演化** | ✅ 考虑 | ✅ 考虑 | ❌ 不考虑 |
| **邻域扩展** | ✅ 显式链接 | ❌ 不扩展 | ✅ 图谱结构 |
| **实体扩展** | ❌ 不支持 | ❌ 不支持 | ✅ 三种信号扩展 |
| **检索速度** | 中等 | 快 | 慢 |
| **检索质量** | 高 | 中高 | 最高 |
| **情绪感知** | ✅ 强 | ✅ 强 | ✅ 强 |
| **语义理解** | 弱 | 强（HippoRAG） | 最强（融合） |
| **实现复杂度** | 中等 | 低 | 高 |

### 3.2 性能对比

| 架构 | 速度 | 质量 | 内存 | 适用场景 |
|------|------|------|------|----------|
| **HyperAmy** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 情绪检索、对话场景 |
| **Fusion 级联** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 通用检索、推荐默认 |
| **Fusion 图谱** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高质量要求、复杂问答 |

### 3.3 使用场景对比

| 场景 | 推荐架构 | 理由 |
|------|----------|------|
| **对话检索** | HyperAmy | 情绪感知强，时间演化符合对话特性 |
| **事实问答** | Fusion 级联 | HippoRAG 的语义理解提升事实准确性 |
| **复杂推理** | Fusion 图谱 | 多信号融合 + PPR 传播，推理能力强 |
| **实时检索** | Fusion 级联 | 速度快，质量有保障 |
| **离线分析** | Fusion 图谱 | 质量最高，时间不敏感 |
| **资源受限** | HyperAmy | 单系统，内存占用小 |

---

## 四、使用建议

### 4.1 默认推荐

根据 README 和代码分析：

```python
# 推荐使用：Fusion 级联检索
from workflow import FusionRetriever

fusion = FusionRetriever(
    amygdala_save_dir="./fusion_amygdala_db",
    hipporag_save_dir="./fusion_hipporag_db"
)

results = fusion.retrieve(
    query="your query",
    hipporag_top_k=20,  # HippoRAG 返回候选数
    amygdala_top_k=5,   # 最终返回数
    mode="cascade"      # 级联检索（默认）
)
```

**理由**：
- ✅ 速度快（HippoRAG 快速筛选）
- ✅ 质量高（Amygdala 深度精排）
- ✅ 平衡性好（兼顾速度和质量）
- ✅ README 标注为"推荐"

### 4.2 特殊场景

#### 场景 1：对话检索 / 情绪感知

```python
from workflow import Amygdala

amygdala = Amygdala(save_dir="./amygdala_db")
results = amygdala.retrieval(
    query_text="我很开心",
    retrieval_mode="particle",
    top_k=10
)
```

**理由**：
- 对话具有时间演化特性
- 情绪感知是核心需求
- 无需复杂的语义理解

#### 场景 2：复杂问答 / 高质量要求

```python
from workflow import GraphFusionRetriever

fusion = GraphFusionRetriever(
    amygdala_save_dir="./graph_fusion_amygdala_db",
    hipporag_save_dir="./graph_fusion_hipporag_db"
)

results = fusion.retrieve(
    query="Why did Monte Cristo refuse the grapes?",
    top_k=5,
    emotion_weight=0.3,   # 情绪权重
    semantic_weight=0.5,  # 语义权重
    fact_weight=0.2       # Fact 权重
)
```

**理由**：
- 需要多信号融合
- PPR 传播提升推理能力
- 质量要求高于速度要求

#### 场景 3：实时检索 / 资源受限

```python
from workflow import FusionRetriever

fusion = FusionRetriever(...)

results = fusion.retrieve(
    query="your query",
    hipporag_top_k=10,  # 减少候选数，提升速度
    amygdala_top_k=3,
    mode="cascade"
)
```

**理由**：
- 级联检索速度快
- 减少候选数降低计算量
- 质量仍有保障

### 4.3 参数调优建议

#### HyperAmy 参数

```python
# 快速检索（牺牲质量）
retriever.search(
    query_entity=query_particle,
    top_k=5,
    cone_width=30,         # 减少筛选范围
    max_neighbors=10,      # 减少邻域扩展
    neighbor_penalty=1.2   # 增大惩罚系数
)

# 精准检索（牺牲速度）
retriever.search(
    query_entity=query_particle,
    top_k=10,
    cone_width=100,        # 增大筛选范围
    max_neighbors=30,      # 增加邻域扩展
    neighbor_penalty=1.05  # 减小惩罚系数
)
```

#### Fusion 级联参数

```python
# 快速检索
fusion.retrieve(
    query="...",
    hipporag_top_k=10,   # HippoRAG 返回少一些
    amygdala_top_k=3     # 最终结果少一些
)

# 精准检索
fusion.retrieve(
    query="...",
    hipporag_top_k=50,   # HippoRAG 返回更多
    amygdala_top_k=10    # 最终结果更多
)
```

#### Fusion 图谱参数

```python
# 侧重语义理解
fusion.retrieve(
    query="...",
    emotion_weight=0.2,   # 降低情绪权重
    semantic_weight=0.7,  # 提高语义权重
    fact_weight=0.1       # 降低 fact 权重
)

# 侧重情绪感知
fusion.retrieve(
    query="...",
    emotion_weight=0.6,   # 提高情绪权重
    semantic_weight=0.2,  # 降低语义权重
    fact_weight=0.2       # 保持 fact 权重
)

# 侧重事实问答
fusion.retrieve(
    query="...",
    emotion_weight=0.1,   # 降低情绪权重
    semantic_weight=0.3,  # 降低语义权重
    fact_weight=0.6       # 提高 fact 权重
)
```

---

## 五、总结

### 5.1 架构选择决策树

```
需要检索？
├─ 是否需要情绪感知？
│  ├─ 是 → HyperAmy（对话场景）
│  └─ 否 → 继续
├─ 是否需要高质量？
│  ├─ 是 → Fusion 图谱（复杂问答）
│  └─ 否 → 继续
├─ 是否需要速度？
│  ├─ 是 → Fusion 级联（推荐默认）
│  └─ 否 → Fusion 图谱
└─ 其他 → Fusion 级联（平衡选择）
```

### 5.2 核心要点

1. **HyperAmy**：双曲空间 + 时间演化 + 邻域扩展，适合情绪检索
2. **Fusion 级联**：HippoRAG 快速筛选 + Amygdala 深度精排，推荐默认
3. **Fusion 图谱**：多信号融合 + PPR 传播，质量最高

### 5.3 代码位置索引

| 架构 | 核心文件 | 关键方法 |
|------|----------|----------|
| **HyperAmy** | `poincare/retrieval.py` | `HyperAmyRetrieval.search()` |
| **Fusion 级联** | `workflow/fusion_retrieval.py` | `FusionRetriever._retrieve_cascade()` |
| **Fusion 图谱** | `workflow/graph_fusion_retrieval.py` | `GraphFusionRetriever.retrieve()` |

---

**文档版本**：v1.0
**最后更新**：2025-01-07
**作者**：Claude Code
