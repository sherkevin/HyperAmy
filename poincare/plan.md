# HyperAmy 开发文档：存储与混合检索模块 (v1.2)

## 0. 模块结构与依赖

### 0.1 模块组织

```
poincare/
├── __init__.py          # 模块导出
├── types.py             # Point, SearchResult 数据类型定义
├── physics.py           # TimePhysics, ParticleProjector 物理计算层
├── storage.py           # HyperAmyStorage 存储层
└── retrieval.py        # HyperAmyRetrieval 检索层
```

### 0.2 依赖要求

```txt
torch >= 1.9.0
chromadb >= 0.4.0
numpy >= 1.21.0
```

## 1. 技术选型与架构

本模块旨在实现一个轻量级、神经-符号结合的情绪记忆系统。

### 向量与属性存储: ChromaDB (持久化到本地磁盘)

**作用**: 存储 `emotion_vector` (作为静态索引，用于 Step 1 锥体搜索) 和 `v`, `T`, `born` 等物理元数据。

**优势**: Python 原生，Serverless，轻量级，支持高效的 Metadata 过滤。

### 物理计算层: NumPy / PyTorch

**作用**: 运行时实时计算时间演化函数 $f(v,t)$、$g(T,t)$ 以及双曲空间距离。

**优势**: 利用向量化计算能力，低延迟处理物理模拟。

### 图关系存储: Chroma Metadata

**作用**: 直接在元数据中存储 `links` 字段 (Adjacency List)。

**优势**: 避免引入重量级图数据库 (Neo4j)，适合 Agent 记忆规模 (10w-100w 级)。

## 2. 核心数据结构与存储 Schema

定义了内存中的业务对象与数据库持久化格式的映射关系。

### 2.1 Python 对象定义 (业务层)

文件: `poincare/types.py`

```python
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
import torch

@dataclass
class Point:
    """
    HyperAmy 系统中的基本粒子单位 (业务层对象)
    """
    id: str
    emotion_vector: torch.Tensor  # [Dim], 原始欧式空间方向向量
    v: float                      # 初始强度/速度 (v_init)
    T: float                      # 初始温度/熵 (T_init)
    born: float                   # 粒子生成时的绝对时间戳
    current_time: Optional[float] = None  # 数据记录更新的时间戳，None 表示使用当前时间
    links: List[str] = field(default_factory=list)  # 邻居粒子的ID列表

    def to_metadata(self) -> Dict[str, Any]:
        """
        序列化为 ChromaDB 支持的 Metadata 格式
        
        ChromaDB 的 metadata 只支持 int, float, str, bool 类型，
        列表和字典必须序列化为 JSON 字符串。
        """
        update_time = self.current_time if self.current_time is not None else time.time()
        return {
            "v": float(self.v),
            "T": float(self.T),
            "born": float(self.born),
            "last_updated": float(update_time),
            "links": json.dumps(self.links)  # 序列化为 JSON 字符串
        }

@dataclass
class SearchResult:
    """
    检索结果数据结构，提供明确的返回格式
    """
    id: str
    score: float          # 双曲距离 (越小越好)
    metadata: Dict[str, Any]
    vector: List[float]
    match_type: Literal['direct', 'neighbor']  # 匹配类型：直接检索或邻域扩展
```

### 2.2 数据库 Schema (ChromaDB 层)

在 ChromaDB 中，一条记录包含以下字段：

- **ID**: `point.id` (String, 独立字段，不存于 Metadata)
- **Embedding**: `point.emotion_vector` (归一化后的 `List[float]`)
- **Metadata**: (Dict) 存储物理属性和图关系序列化字符串

**注意**: ChromaDB 的 metadata 只支持 `int`, `float`, `str`, `bool` 类型，列表和字典必须序列化为 JSON 字符串。

```json
{
    "v": 0.9,
    "T": 0.5,
    "born": 1715000000.0,
    "last_updated": 1715000000.0,
    "links": "[\"id_A\", \"id_B\", \"id_C\"]"
}
```

## 3. 物理演化函数接口 (Physics Evolution Interfaces)

文件: `poincare/physics.py`

本层定义了粒子属性随时间变化的物理法则。

```python
import math
import torch
import torch.nn.functional as F

class TimePhysics:
    """
    物理演化层：计算粒子属性随时间的演化
    
    目前 f 和 g 均为恒等映射，为未来记忆固化与遗忘曲线预留接口。
    """
    @staticmethod
    def f(v: float, t_born: float, t_now: float) -> float:
        """
        速度/强度演化函数 f(v, t)
        
        Args:
            v: 初始速度/强度
            t_born: 粒子产生时间
            t_now: 当前计算时间
            
        Returns:
            当前时刻的速度/强度
        """
        # 默认实现: f(x, t) = x (速度恒定，无阻力)
        # 边界处理: 速度不能为负
        return max(0.0, v)

    @staticmethod
    def g(T: float, t_born: float, t_now: float) -> float:
        """
        温度演化函数 g(T, t)
        
        Args:
            T: 初始温度
            t_born: 粒子产生时间
            t_now: 当前计算时间
            
        Returns:
            当前时刻的温度
        """
        # 默认实现: g(x, t) = x (温度恒定，无冷却)
        # 边界处理: 温度不能为负
        return max(0.0, T)

class ParticleProjector:
    """
    空间投影层：欧式空间 -> 庞加莱球
    
    将物理属性映射为双曲几何坐标，支持任意曲率的双曲空间。
    """
    def __init__(self, curvature: float = 1.0, scaling_factor: float = 2.0):
        """
        Args:
            curvature (c): 双曲空间的曲率，默认 1.0。曲率越大，空间弯曲程度越高。
            scaling_factor: 强度映射放大系数，默认 2.0。系数越大，同等强度下离圆心越远。
        """
        self.c = curvature
        self.scaling_factor = scaling_factor
        # 预计算 sqrt(c) 避免重复计算
        self.sqrt_c = math.sqrt(curvature)

    def compute_state(self, 
                      vec: torch.Tensor, 
                      v: float, 
                      T: float, 
                      born: float, 
                      t_now: float) -> dict:
        """
        计算粒子在当前时刻的动态状态（双曲坐标、当前速度、当前温度）
        
        性能优化：直接接收原始数值，避免构建 Point 对象的开销。
        
        Args:
            vec: 情感向量 (torch.Tensor)
            v: 初始速度/强度
            T: 初始温度
            born: 生成时间戳
            t_now: 当前时间戳
            
        Returns:
            包含以下键的字典:
            - current_vector: 庞加莱球坐标 (torch.Tensor)
            - current_v: 当前速度/强度 (float)
            - current_T: 当前温度 (float)
        """
        # 1. 物理演化：计算当前时刻的 v 和 T
        v_current = TimePhysics.f(v, born, t_now)
        T_current = TimePhysics.g(T, born, t_now)
        
        # 2. 空间投影：欧式 -> 双曲
        # 零向量保护：避免归一化零向量导致的 NaN
        vec_norm = torch.norm(vec)
        if vec_norm < 1e-9:
            direction = torch.zeros_like(vec)
        else:
            direction = F.normalize(vec, p=2, dim=-1)
        
        # 广义双曲投影公式
        # r = tanh(sqrt(c) * dist / 2) / sqrt(c)
        # 这里 dist = v_current * scaling_factor
        arg = self.sqrt_c * v_current * self.scaling_factor / 2.0
        r = torch.tanh(torch.tensor(arg, dtype=vec.dtype)) / self.sqrt_c
        
        poincare_coord = r * direction
        
        return {
            "current_vector": poincare_coord,
            "current_v": v_current,
            "current_T": T_current
        }
```

## 4. 核心功能实现

### 4.1 存储与更新函数 (Storage & Update)

文件: `poincare/storage.py`

负责数据的持久化存储。

```python
import logging
import torch
import chromadb
from poincare.types import Point

logger = logging.getLogger(__name__)

class HyperAmyStorage:
    """
    存储层：负责 Point 对象的持久化
    
    使用 ChromaDB 作为底层存储，支持向量相似度搜索和元数据过滤。
    """
    def __init__(self, persist_path="./hyperamy_db"):
        """
        Args:
            persist_path: ChromaDB 数据库持久化路径
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name="emotion_particles",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度进行向量搜索
        )

    def upsert_point(self, point: Point):
        """
        存储或更新粒子
        
        Args:
            point: 要存储的 Point 对象
            
        Raises:
            Exception: 存储失败时抛出异常
        """
        try:
            # 1. 向量归一化：确保存储的是归一化向量，纯粹表示"方向"
            norm_vec = torch.nn.functional.normalize(point.emotion_vector, p=2, dim=-1)
            embedding_list = norm_vec.tolist()

            # 2. 准备 Metadata (调用 Point 的封装方法)
            metadata = point.to_metadata()

            # 3. 写入 ChromaDB
            self.collection.upsert(
                ids=[point.id],
                embeddings=[embedding_list],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to upsert point {point.id}: {str(e)}")
            raise e
```

### 4.2 混合检索流水线 (The Hybrid Retrieval Pipeline)

文件: `poincare/retrieval.py`

实现了 锥体 -> 壳层 -> 邻域 -> 汇总 的四步检索法。

```python
import time
import json
import logging
import torch
from typing import List, Dict, Any

from poincare.types import Point, SearchResult
from poincare.storage import HyperAmyStorage
from poincare.physics import ParticleProjector

logger = logging.getLogger(__name__)

class HyperAmyRetrieval:
    """
    检索层：实现混合检索流水线
    
    四步检索流程：
    1. 锥体锁定：使用向量相似度快速圈定方向一致的粒子
    2. 壳层筛选：计算真实的双曲距离进行精排
    3. 邻域激活：从 Top-K 点出发，扩展其邻居节点
    4. 汇总排序：混合直接检索点和邻居点，最终排序返回
    """
    def __init__(self, storage: HyperAmyStorage, projector: ParticleProjector):
        """
        Args:
            storage: 存储层实例
            projector: 投影器实例
        """
        self.collection = storage.collection
        self.projector = projector

    def _poincare_dist(self, u: torch.Tensor, v: torch.Tensor) -> float:
        """
        计算两个双曲坐标之间的庞加莱距离
        
        Args:
            u, v: 庞加莱球内的坐标向量
            
        Returns:
            双曲距离（非负浮点数）
        """
        sq_dist = torch.sum((u - v) ** 2)
        u_sq = torch.sum(u ** 2)
        v_sq = torch.sum(v ** 2)
        denom = (1 - u_sq) * (1 - v_sq) + 1e-7
        arg = 1 + 2 * sq_dist / denom
        # 边界保护：确保 arg >= 1，避免 acosh 域错误
        # 当坐标接近球面边界时，可能出现数值不稳定
        arg = torch.clamp(arg, min=1.0 + 1e-7)
        return torch.acosh(arg).item()

    def _calculate_score_raw(self, 
                           dynamic_query: dict, 
                           cand_vec: List[float], 
                           cand_meta: Dict[str, Any], 
                           t_now: float) -> float:
        """
        计算查询点与候选点的双曲距离（性能优化版本）
        
        直接使用原始数值，避免构建 Point 对象的开销。
        适用于批量计算场景。
        
        Args:
            dynamic_query: 查询点的动态状态（已预计算）
            cand_vec: 候选点的向量
            cand_meta: 候选点的元数据
            t_now: 当前时间戳
            
        Returns:
            双曲距离
        """
        # 计算候选点的动态状态
        dynamic_cand = self.projector.compute_state(
            vec=torch.tensor(cand_vec),
            v=cand_meta['v'],
            T=cand_meta['T'],
            born=cand_meta['born'],
            t_now=t_now
        )
        
        # 计算双曲距离
        return self._poincare_dist(
            dynamic_query['current_vector'],
            dynamic_cand['current_vector']
        )

    def search(self, 
               query_point: Point, 
               top_k: int = 3, 
               cone_width: int = 50, 
               max_neighbors: int = 20,
               neighbor_penalty: float = 1.1) -> List[SearchResult]:
        """
        执行混合检索
        
        Args:
            query_point: 查询点
            top_k: 返回结果数量
            cone_width: 锥体搜索宽度。建议值 50-100。
                       值越大，召回率越高，但后续物理计算开销越大。
                       应根据总数据量 N 动态调整 (如 logN * 10)。
            max_neighbors: 邻域扩展时的最大节点数，防止超级节点导致性能雪崩。
            neighbor_penalty: 邻居惩罚系数，默认 1.1 (邻居距离会被放大 10%)。
                             用于降低间接连接的优先级。
        
        Returns:
            检索结果列表，按双曲距离从小到大排序
        """
        t_now = time.time()
        
        # 预计算 Query 状态（避免在循环中重复计算）
        dynamic_query = self.projector.compute_state(
            vec=query_point.emotion_vector,
            v=query_point.v,
            T=query_point.T,
            born=query_point.born,
            t_now=t_now
        )

        # --- Step 1: Cone Search (锥体锁定) ---
        query_vec_list = torch.nn.functional.normalize(
            query_point.emotion_vector, p=2, dim=-1
        ).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_vec_list],
                n_results=cone_width,
                include=["metadatas", "embeddings"]
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        if not results['ids'] or not results['ids'][0]:
            return []

        # --- Step 2: Distance Ranking (壳层/距离筛选) ---
        ids = results['ids'][0]
        metas = results['metadatas'][0]
        vecs = results['embeddings'][0]

        scored_candidates = []
        
        for pid, meta, vec in zip(ids, metas, vecs):
            try:
                score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
                scored_candidates.append(SearchResult(
                    id=pid, 
                    score=score, 
                    metadata=meta, 
                    vector=vec, 
                    match_type='direct'
                ))
            except Exception as e:
                logger.warning(f"Error calculating score for {pid}: {e}")
                continue

        # 初步排序，取前 K 个进入邻域扩展
        scored_candidates.sort(key=lambda x: x.score)
        top_candidates = scored_candidates[:top_k]

        # --- Step 3: Neighborhood Expansion (邻域激活) ---
        neighbor_ids = []
        for cand in top_candidates:
            try:
                links_str = cand.metadata.get('links', '[]')
                links = json.loads(links_str)
                # 性能保护：限制单个节点的扩展数量
                if len(links) > max_neighbors:
                    links = links[:max_neighbors]
                neighbor_ids.extend(links)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse links for {cand.id}: {e}")
        
        # 去重并过滤已存在的点
        existing_ids = {c.id for c in top_candidates}
        neighbor_ids = list(set(neighbor_ids) - existing_ids)
        
        # 全局性能保护：限制总扩展数量
        if len(neighbor_ids) > max_neighbors * top_k:
            neighbor_ids = neighbor_ids[:max_neighbors * top_k]

        if neighbor_ids:
            try:
                nb_results = self.collection.get(
                    ids=neighbor_ids, 
                    include=["metadatas", "embeddings"]
                )
                if nb_results['ids']:
                    for i, nid in enumerate(nb_results['ids']):
                        meta = nb_results['metadatas'][i]
                        vec = nb_results['embeddings'][i]
                        try:
                            score = self._calculate_score_raw(dynamic_query, vec, meta, t_now)
                            # 应用邻居惩罚系数
                            score *= neighbor_penalty 
                            
                            top_candidates.append(SearchResult(
                                id=nid, 
                                score=score, 
                                metadata=meta, 
                                vector=vec, 
                                match_type='neighbor'
                            ))
                        except Exception as e:
                            logger.debug(f"Error calculating neighbor score for {nid}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Neighbor fetch failed: {e}")

        # --- Step 4: Final Aggregation (汇总与最终排序) ---
        top_candidates.sort(key=lambda x: x.score)
        return top_candidates[:top_k]
```

## 5. 使用示例

```python
import time
import torch
import logging
from poincare.types import Point
from poincare.physics import ParticleProjector
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval

# 配置日志
logging.basicConfig(level=logging.INFO)

def main():
    # 1. 初始化组件
    projector = ParticleProjector(curvature=1.0, scaling_factor=2.0)
    storage = HyperAmyStorage(persist_path="./demo_db")
    retrieval = HyperAmyRetrieval(storage, projector)

    # 2. 插入测试数据
    # 模拟场景：三个点，A是中心，B和C是邻居
    # A (愤怒, 强) -> B (愤怒, 弱) -> C (开心)
    vec_anger = torch.tensor([0.9, 0.1, 0.0])
    vec_happy = torch.tensor([0.0, 0.1, 0.9])
    
    p_a = Point(
        id="A", 
        emotion_vector=vec_anger, 
        v=0.9, 
        T=0.5, 
        born=time.time(), 
        links=["B", "C"]
    )
    p_b = Point(
        id="B", 
        emotion_vector=vec_anger, 
        v=0.2, 
        T=0.5, 
        born=time.time()
    )
    p_c = Point(
        id="C", 
        emotion_vector=vec_happy, 
        v=0.5, 
        T=0.5, 
        born=time.time()
    )

    print("Upserting points...")
    storage.upsert_point(p_a)
    storage.upsert_point(p_b)
    storage.upsert_point(p_c)

    # 3. 构造查询点 (愤怒, 强)
    query = Point(
        id="Q", 
        emotion_vector=vec_anger, 
        v=0.9, 
        T=0.5, 
        born=time.time()
    )

    # 4. 执行检索
    print("Searching...")
    results = retrieval.search(
        query, 
        top_k=3, 
        cone_width=50,
        neighbor_penalty=1.2
    )

    # 5. 输出结果
    for res in results:
        print(
            f"ID: {res.id}, "
            f"Score: {res.score:.4f}, "
            f"Type: {res.match_type}, "
            f"v: {res.metadata['v']}"
        )

if __name__ == "__main__":
    main()
```

## 6. 性能优化建议

1. **批量操作**: 对于大量点的插入，考虑实现 `upsert_points()` 批量接口
2. **缓存机制**: 对于频繁查询的点，可以考虑缓存其动态状态
3. **异步处理**: 对于大规模数据，可以考虑异步处理邻域扩展
4. **参数调优**: `cone_width` 和 `max_neighbors` 应根据实际数据规模动态调整

## 7. 注意事项

1. **数值稳定性**: 当坐标接近庞加莱球边界时，可能出现数值不稳定，已添加边界保护
2. **内存管理**: 大规模数据时注意内存占用，考虑分批处理
3. **并发安全**: ChromaDB 的并发写入需要额外考虑，建议使用锁机制
4. **数据一致性**: `links` 字段需要手动维护，确保引用的 ID 存在
