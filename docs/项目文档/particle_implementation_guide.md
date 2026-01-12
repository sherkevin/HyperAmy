# 粒子物理系统实现指南

**版本**: v3.1 - Soft Label 强度模型
**日期**: 2026-01-12
**状态**: 已实现

---

## 目录

1. [架构概览](#1-架构概览)
2. [核心模块](#2-核心模块)
3. [使用指南](#3-使用指南)
4. [API 参考](#4-api-参考)
5. [示例代码](#5-示例代码)

---

## 1. 架构概览

### 1.1 模块结构

```
particle/
├── emotion_classifier.py       # 情绪分类器（文本 → I_raw）
├── emotion_vector_processor.py  # 向量处理器（e_raw → μ, κ）
├── emotion_v2.py                 # 情绪向量获取（与解耦）
├── particle.py                   # 粒子聚合类
├── purity.py                     # 纯度计算
├── temperature.py                # 温度计算
└── emotion_cache.py              # 缓存管理

poincare/
├── physics.py                    # 物理引擎（时间膨胀）
├── projector.py                  # Poincaré 投影
├── math.py                       # 数学工具
├── storage.py                    # 向量存储
└── retrieval.py                  # 检索引擎
```

### 1.2 数据流（解耦设计）

```
┌─────────────────────────────────────────────────────────────┐
│  步骤1: 情绪向量获取（可替换）                                │
│  - LLM + Embedding API（当前实现）                            │
│  - 专门的情绪向量模型（未来）                                 │
│  输出: e_raw (原始向量，64/128维)                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤2: 情绪强度分类                                          │
│  - 输入: 情感描述文本                                          │
│  - 输出: Soft Label → I_raw ∈ [0, 1]                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤3: 向量处理（解耦核心）                                  │
│  - 方向: μ = e_raw / ||e_raw||                                │
│  - 模长: κ = 1.0 + α × I_raw                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤4: 粒子初始化                                           │
│  - ParticleEntity(μ, κ, T₀, purity, ...)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  步骤5: 时间演化（可选）                                       │
│  - R(t) = R₀ × exp(-γ/m × t)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块

### 2.1 EmotionClassifier（情绪分类器）

**文件**: `particle/emotion_classifier.py`

**功能**: 将情感描述文本转换为情绪强度 I_raw

```python
from particle.emotion_classifier import get_global_classifier

classifier = get_global_classifier()

# 从文本获取强度
intensity = classifier.get_intensity("enthusiasm, joy, hope")
# intensity ≈ 0.6-0.8
```

**规则分类**（默认）:
- 识别预定义的情绪词
- 映射到主要类别
- 返回最大非中性概率

**LLM 分类**（可选）:
- 更准确但较慢
- 输出完整的 Soft Label 分布

### 2.2 EmotionVectorProcessor（向量处理器）

**文件**: `particle/emotion_vector_processor.py`

**功能**: 将原始向量和强度转换为方向和模长

```python
from particle.emotion_vector_processor import get_global_processor

processor = get_global_processor(alpha=50.0)

# 处理向量
processed = processor.process(
    raw_vector=emotion_vector,  # 来自模型的原始向量
    intensity=0.7                # 来自分类器
)

# 结果
print(processed.direction)   # 归一化方向 μ
print(processed.modulus)     # 模长 κ = 1.0 + 50 × 0.7 = 36.0
print(processed.state)       # "液态"
```

**状态判据**:
| 模长 κ | 状态 |
|--------|------|
| < 10 | 气态（快速衰减）|
| 10-40 | 液态（中等衰减）|
| ≥ 40 | 固态（缓慢衰减）|

### 2.3 EmotionNode（情绪节点）

**文件**: `particle/emotion_v2.py`

```python
@dataclass
class EmotionNode:
    entity_id: str              # 实体 ID（MD5 hash）
    entity: str                 # 实体名称
    emotion_vector: np.ndarray  # 原始向量（未归一化）
    text_id: str                # 来源文本 ID
    intensity: float            # 情绪强度 I_raw
    raw_description: str        # 原始情感描述
```

### 2.4 ParticleEntity（粒子实体）

**文件**: `particle/particle.py`

```python
@dataclass
class ParticleEntity:
    entity_id: str              # 粒子唯一 ID
    entity: str                 # 实体名称
    text_id: str                # 来源文本 ID
    emotion_vector: np.ndarray  # 归一化方向 μ
    weight: float               # 模长 κ（作为质量）
    speed: float                # 初始速度 = κ
    temperature: float          # 初始温度
    purity: float               # 纯度
    tau_v: float                # 速度衰减常数
    tau_T: float                # 温度冷却常数
    born: float                 # 生成时间戳
    intensity: float            # 情绪强度 I_raw
```

### 2.5 PhysicsEngine（物理引擎）

**文件**: `poincare/physics.py`

**核心公式**:
```python
# 时间膨胀衰变
R_t = TimeDynamics.hyperbolic_radius(
    initial_radius=R_0,
    mass=m,           # = κ
    delta_t=t,
    gamma=1.0
)

# 记忆强度
strength = TimeDynamics.memory_strength(
    initial_radius=R_0,
    mass=m,
    delta_t=t,
    gamma=1.0
)

# 遗忘判断
forgotten = TimeDynamics.is_forgotten(
    initial_radius=R_0,
    mass=m,
    delta_t=t,
    threshold=1e-3
)
```

---

## 3. 使用指南

### 3.1 基本用法

```python
from particle.particle import Particle

# 初始化
particle = Particle(
    model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-3",
    intensity_alpha=50.0  # κ = 1.0 + α × I_raw
)

# 处理文本
particles = particle.process(
    text="The Count's eyes filled with terror as he slowly reached for the emerald pill.",
    text_id="doc_001"
)

# 查看结果
for p in particles:
    print(f"Entity: {p.entity}")
    print(f"  Intensity: {p.intensity:.3f}")
    print(f"  Modulus: {p.weight:.2f}")
    print(f"  State: {['气态', '液态', '固态'][
        ['gas' if p.weight < 10 else
         'liquid' if p.weight < 40 else 'solid']]}")
```

### 3.2 自定义情绪向量（解耦）

```python
# 假设你有一个专门的情绪向量模型
raw_vector = model.get_emotion_vector(text)  # 64维向量

# 获取强度
from particle.emotion_classifier import get_global_classifier
intensity = get_global_classifier().get_intensity(description_text)

# 处理向量
from particle.emotion_vector_processor import get_global_processor
processed = get_global_processor().process(raw_vector, intensity)

# 创建粒子
from particle.particle import ParticleEntity
particle = ParticleEntity(
    entity_id="unique_id",
    entity="Count Monte Cristo",
    text_id="doc_001",
    emotion_vector=processed.direction,  # μ
    weight=processed.modulus,           # κ
    speed=processed.modulus,
    temperature=0.5,
    purity=0.8,
    tau_v=100000,
    tau_T=100000,
    born=time.time(),
    intensity=processed.intensity
)
```

### 3.3 时间演化查询

```python
from poincare.physics import PhysicsEngine, TimeDynamics

# 初始化物理引擎
physics = PhysicsEngine(curvature=1.0, gamma=1.0)

# 计算当前状态
state = physics.compute_state(
    direction=particle.emotion_vector,
    mass=particle.weight,
    temperature=particle.temperature,
    initial_radius=particle.weight * 2.0,  # R₀
    created_at=particle.born,
    t_now=time.time()
)

print(f"Current radius: {state.hyperbolic_radius:.4f}")
print(f"Memory strength: {state.memory_strength:.4f}")
print(f"Is forgotten: {state.is_forgotten}")
```

### 3.4 完整工作流

```python
from workflow.amygdala import Amygdala

# 初始化 Amygdala（集成所有模块）
amygdala = Amygdala(
    save_dir="./amygdala_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-3"
)

# 添加对话
amygdala.add(
    conversation="The Count's eyes filled with terror...",
    conversation_id="doc_001"
)

# 检索
results = amygdala.retrieve(
    query="What is the Count afraid of?",
    top_k=5
)
```

---

## 4. API 参考

### 4.1 EmotionClassifier

```python
class EmotionClassifier:
    def classify(text: str, use_llm: bool = False) -> Dict[str, float]
    def get_intensity(text: str, use_llm: bool = False) -> float
    def get_intensity_batch(texts: List[str], use_llm: bool = False) -> List[float]
```

### 4.2 EmotionVectorProcessor

```python
class EmotionVectorProcessor:
    def __init__(alpha: float = 50.0, min_modulus: float = 1.0)
    def process(self, raw_vector: np.ndarray, intensity: float, normalize: bool = True) -> ProcessedEmotionVector
    def process_batch(self, raw_vectors: List[np.ndarray], intensities: List[float]) -> List[ProcessedEmotionVector]
    def compute_state(self, modulus: float) -> str
```

### 4.3 ProcessedEmotionVector

```python
@dataclass
class ProcessedEmotionVector:
    direction: np.ndarray  # 归一化方向 μ
    modulus: float         # 模长 κ
    intensity: float       # 情绪强度 I_raw
    state: str             # 状态（"气态"/"液态"/"固态"）
```

### 4.4 Particle

```python
class Particle:
    def __init__(
        model_name=None,
        embedding_model_name=None,
        intensity_alpha: float = 50.0,  # NEW
        **kwargs
    )
    def process(self, text: str, text_id: str, entities: Optional[List[str]] = None) -> List[ParticleEntity]
```

---

## 5. 示例代码

### 5.1 完整示例

```python
from workflow.amygdala import Amygdala
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 初始化
amygdala = Amygdala(
    save_dir="./example_db",
    llm_model_name="DeepSeek-V3.2",
    embedding_model_name="GLM-Embedding-3"
)

# 添加对话
conversations = [
    "I have an excellent appetite, said Albert.",
    "The Count took a small greenish pill and swallowed it.",
    "She looked at him with terror in her eyes.",
    "This is my food, said the Count, I never eat.",
    "The conversation drifted to Eastern philosophy."
]

for i, conv in enumerate(conversations):
    amygdala.add(
        conversation=conv,
        conversation_id=f"conv_{i:03d}"
    )

# 检索
query = "What philosophical concept does the Count adhere to?"
results = amygdala.retrieve(query=query, top_k=5)

for i, r in enumerate(results):
    print(f"Rank {i+1}: {r['conversation_id']}")
    print(f"  Score: {r['score']:.4f}")
    print(f"  Intensity: {r.get('intensity', 0):.3f}")
```

### 5.2 自定义情绪向量模型

```python
import numpy as np
from particle.emotion_classifier import EmotionClassifier
from particle.emotion_vector_processor import EmotionVectorProcessor
from particle.particle import ParticleEntity

# 假设的模型
class MyEmotionModel:
    def get_vector(self, text: str) -> np.ndarray:
        # 返回 128 维情绪向量
        return np.random.randn(128)

# 使用
model = MyEmotionModel()
raw_vector = model.get_vector("Joyful excitement fills the room")

# 分类
intensity = EmotionClassifier().get_intensity("Joyful, excitement, happy")
# intensity ≈ 0.8

# 处理
processor = EmotionVectorProcessor(alpha=50.0)
processed = processor.process(raw_vector, intensity)
# processed.modulus ≈ 41.0 (固态)

# 创建粒子
particle = ParticleEntity(
    entity_id="joy_001",
    entity="joy",
    text_id="doc_001",
    emotion_vector=processed.direction,
    weight=processed.modulus,
    speed=processed.modulus,
    intensity=processed.intensity,
    temperature=0.3,
    purity=0.9,
    tau_v=200000,
    tau_T=200000,
    born=time.time()
)
```

---

## 附录：快速参考

### A.1 关键公式

```
I_raw = max_{c≠neutral}(y_c)           # 情绪强度
κ = 1.0 + α × I_raw                    # 模长 (α=50)
μ = e_raw / ||e_raw||                   # 方向
R_0 = 2.0 × κ                          # 初始半径
R(t) = R₀ × exp(-γ/m × t)               # 时间演化
```

### A.2 状态阈值

```
气态: κ < 10
液态: 10 ≤ κ < 40
固态: κ ≥ 40
```

### A.3 文件映射

| 功能 | 文件 |
|-----|------|
| 情绪分类 | `emotion_classifier.py` |
| 向量处理 | `emotion_vector_processor.py` |
| 向量获取 | `emotion_v2.py` |
| 粒子聚合 | `particle.py` |
| 物理引擎 | `poincare/physics.py` |
| 坐标投影 | `poincare/projector.py` |
| 向量存储 | `poincare/storage.py` |
| 检索引擎 | `poincare/retrieval.py` |
| 高层接口 | `workflow/amygdala.py` |
