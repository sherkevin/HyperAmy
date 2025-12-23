# HippoRAG + 情感分析整合说明

## ✅ 整合完成

已成功将 HyperAmy 的情感分析功能整合到 HippoRAG 框架中。

## 📁 新增文件

- `sentiment/`: 情感分析模块
  - `emotion_vector.py`: 情感向量提取
  - `emotion_store.py`: 情感向量存储
  - `hipporag_enhanced.py`: 增强版 HippoRAG
- `scripts/test_integration.py`: 整合测试脚本
- `scripts/test_dataset_integration.py`: 数据集测试脚本

## 🚀 使用方法

```python
from sentiment.hipporag_enhanced import HippoRAGEnhanced

hipporag = HippoRAGEnhanced(
    global_config=config,
    enable_emotion=True,
    emotion_weight=0.3
)

hipporag.index(docs)
results = hipporag.retrieve(queries)
```

详细说明请查看 `整合完成说明.md`

