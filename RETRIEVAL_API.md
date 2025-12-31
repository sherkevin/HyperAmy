# Amygdala.retrieval() 方法使用文档

## 方法签名

```python
def retrieval(
    self,
    query_text: str,
    retrieval_mode: Literal["particle", "chunk"] = "particle",
    top_k: int = 10,
    cone_width: int = 50,
    max_neighbors: int = 20,
    neighbor_penalty: float = 1.1
) -> List[Dict[str, Any]]
```

## 参数说明

### 必需参数
- **query_text** (str): 查询文本，用于检索相关内容

### 可选参数
- **retrieval_mode** (Literal["particle", "chunk"]): 检索模式
  - `"particle"`: 返回最相关的粒子（默认）
  - `"chunk"`: 返回最相关的对话片段

- **top_k** (int): 返回结果数量，默认 10

- **cone_width** (int): 锥体搜索宽度（建议值 50-100）
  - 值越大，召回率越高，但计算开销越大
  - 应根据总数据量动态调整

- **max_neighbors** (int): 邻域扩展时的最大节点数，默认 20
  - 防止超级节点导致性能雪崩

- **neighbor_penalty** (float): 邻居惩罚系数，默认 1.1
  - 用于降低间接连接的优先级

---

## 返回值

### retrieval_mode="particle"

返回粒子列表，按双曲距离（相似度）排序：

```python
[
    {
        "particle_id": str,           # 粒子 ID
        "entity": str,                # 实体名称
        "score": float,               # 双曲距离（越小越相似）
        "conversation_id": str,       # 所属对话 ID
        "match_type": "direct" | "neighbor",  # 匹配类型
        "metadata": {
            "speed": float,           # 速度
            "temperature": float,     # 温度
            "born": float,            # 生成时间
            "weight": float           # 权重
        }
    },
    ...
]
```

### retrieval_mode="chunk"

返回对话片段列表，按 chunk 得分降序排序：

```python
[
    {
        "conversation_id": str,       # 对话 ID
        "text": str,                  # 对话文本
        "score": float,               # chunk 得分（越高越相关）
        "particle_count": int,        # 包含的粒子数量
        "particle_ids": List[str],    # 包含的粒子 ID 列表
        "rank": int                   # 排名
    },
    ...
]
```

---

## Chunk 排序规则

### 计算公式

```python
chunk_score = sum((total_particles - position) for each particle in chunk)
```

其中：
- `total_particles`: 检索到的粒子总数
- `position`: 粒子在搜索结果中的位置（0-based，越靠前值越大）

### 示例

假设检索到 5 个粒子，位置 0-4：

```python
粒子0: 位置0, 属于Chunk A -> Chunk A得分 += (5-0) = 5
粒子1: 位置1, 属于Chunk A -> Chunk A得分 += (5-1) = 4
粒子2: 位置2, 属于Chunk B -> Chunk B得分 += (5-2) = 3
粒子3: 位置3, 属于Chunk A -> Chunk A得分 += (5-3) = 2
粒子4: 位置4, 属于Chunk B -> Chunk B得分 += (5-4) = 1

结果：
- Chunk A: 5+4+2 = 11分（包含3个粒子，且靠前）
- Chunk B: 3+1 = 4分（包含2个粒子，且靠后）
```

**结论**: 包含越靠前粒子越多的 chunk，得分越高，排名越靠前。

---

## 使用示例

### 示例1：检索相关粒子

```python
from workflow.amygdala import Amygdala

# 初始化
amygdala = Amygdala(
    save_dir="./my_db",
    particle_collection_name="particles",
    conversation_namespace="conversations"
)

# 添加对话（如果数据库中没有数据）
amygdala.add("I love Python programming!")
amygdala.add("Machine learning is fascinating.")

# 检索相关粒子
results = amygdala.retrieval(
    query_text="I enjoy coding with Python",
    retrieval_mode="particle",
    top_k=5
)

# 查看结果
for particle in results:
    print(f"实体: {particle['entity']}")
    print(f"相似度: {particle['score']:.4f}")
    print(f"所属对话: {particle['conversation_id']}")
    print()
```

输出：
```
实体: programming
相似度: 15.9112
所属对话: test_conversation-xxx

实体: Python
相似度: 16.0234
所属对话: test_conversation-xxx
```

---

### 示例2：检索相关对话片段

```python
# 检索相关对话片段
results = amygdala.retrieval(
    query_text="web development frameworks",
    retrieval_mode="chunk",
    top_k=3
)

# 查看结果
for chunk in results:
    print(f"排名: {chunk['rank']}")
    print(f"得分: {chunk['score']:.1f}")
    print(f"包含粒子数: {chunk['particle_count']}")
    print(f"对话文本: {chunk['text']}")
    print()
```

输出：
```
排名: 1
得分: 62.0
包含粒子数: 5
对话文本: JavaScript is great for web development. React and Vue are popular frameworks.

排名: 2
得分: 36.0
包含粒子数: 3
对话文本: I love Python programming! It's amazing for data science.
```

---

### 示例3：调整检索参数

```python
# 更宽泛的检索（召回率更高）
results = amygdala.retrieval(
    query_text="programming languages",
    retrieval_mode="particle",
    top_k=20,
    cone_width=100  # 扩大搜索范围
)

# 更精确的检索（精确度更高）
results = amygdala.retrieval(
    query_text="Python programming",
    retrieval_mode="particle",
    top_k=5,
    cone_width=30,  # 缩小搜索范围
    max_neighbors=10  # 减少邻域扩展
)
```

---

## 工作原理

### 检索流程

1. **文本转粒子**: 将查询文本转换为查询粒子（使用 `Particle.process()`）
2. **粒子检索**: 使用 `HyperAmyRetrieval.search()` 检索相似粒子
   - 锥体锁定：使用向量相似度快速圈定方向一致的粒子
   - 距离排序：计算双曲距离进行精排
   - 邻域扩展：从 Top-K 点出发，扩展邻居节点
3. **结果格式化**: 根据 `retrieval_mode` 格式化结果
   - particle 模式：直接返回粒子列表
   - chunk 模式：将粒子映射回对话，计算 chunk 得分

### 双曲距离

- 使用庞加莱球模型计算粒子之间的距离
- 距离越小，表示越相似
- 考虑粒子的速度、温度、权重等动态特性

---

## 性能优化建议

### 1. 调整 cone_width

```python
# 数据量小 (< 1000 粒子)
cone_width=30

# 数据量中等 (1000-10000 粒子)
cone_width=50  # 默认值

# 数据量大 (> 10000 粒子)
cone_width=100
```

### 2. 调整 max_neighbors

```python
# 不需要邻域扩展（更快）
max_neighbors=0

# 需要一定扩展（平衡）
max_neighbors=20  # 默认值

# 需要大量扩展（更全面）
max_neighbors=50
```

### 3. 选择合适的检索模式

```python
# 需要细粒度结果（查看具体实体）
retrieval_mode="particle"

# 需要整体上下文（查看完整对话）
retrieval_mode="chunk"
```

---

## 常见问题

### Q1: 为什么有些查询没有结果？

**A**: 可能原因：
1. 查询文本未生成任何粒子（实体提取失败）
2. 数据库中确实没有相关粒子
3. cone_width 太小，搜索范围不够

**解决方案**：
- 检查查询文本是否能提取到实体
- 增大 cone_width 参数
- 确保数据库中有相关数据

### Q2: Chunk 模式和 Particle 模式如何选择？

**A**:
- **Particle 模式**：适合需要细粒度检索的场景
  - 示例：查找提到"Python"的具体句子
  - 优点：精确，能看到具体实体
  - 缺点：缺少上下文

- **Chunk 模式**：适合需要整体上下文的场景
  - 示例：查找关于"编程"的对话
  - 优点：有完整上下文，结果更聚合
  - 缺点：不够细粒度

### Q3: 如何提高检索准确度？

**A**:
1. **优化查询文本**：使用更具体的描述
   ```python
   # ❌ 不够具体
   "programming"

   # ✅ 更具体
   "Python programming for data science"
   ```

2. **调整参数**：
   ```python
   # 提高精确度
   top_k=5,           # 减少返回数量
   cone_width=30,     # 缩小搜索范围
   max_neighbors=10   # 减少邻域扩展
   ```

3. **使用更多数据**：添加更多相关对话到数据库

---

## 更新日志

### v1.0 (2025-12-29)
- ✅ 实现基础检索功能
- ✅ 支持粒子检索模式（particle）
- ✅ 支持对话片段检索模式（chunk）
- ✅ 实现 chunk 得分排序算法
- ✅ 完整的参数配置支持
- ✅ 详细的日志记录

---

## 相关文档

- [Amygdala类文档](../README.md)
- [HyperAmyRetrieval文档](../poincare/retrieval.py)
- [Particle处理流程](../particle/README.md)
