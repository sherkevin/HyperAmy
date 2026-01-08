# HyperAmy 测试文档

本文档描述 HyperAmy 项目中所有测试的用途和使用方法。

## 目录结构

```
test/
├── README.md                              # 本文档
├── test_amygdala.py                       # Amygdala 工作流测试
├── test_particle.py                       # Particle 模块测试
├── test_emotion_v2.py                     # EmotionV2 情感提取测试
├── test_poincare.py                       # 双曲空间存储测试
├── test_poincare_retrieval.py             # 双曲空间检索测试
├── test_particle_poincare_flow.py         # 粒子-双曲空间完整流程测试
├── test_retrieval.py                      # Amygdala 检索功能测试
├── test_monte_cristo_retrieval.py         # Monte Cristo 场景检索测试
├── test_monte_cristo_retrieval_detailed.py # Monte Cristo 详细日志测试
├── test_linking.py                        # 双曲空间链接构建测试
├── test_speed.py                          # 速度计算测试
├── test_labels.py                         # Labels (Particle) 测试
├── test_entity.py                         # 实体抽取测试
├── test_sentence.py                       # 句子生成测试
├── test_emotion.py                        # Emotion (旧版) 测试
├── test_infer.py                          # 推理和 token 概率测试
├── test_completion_client.py              # LLM 客户端测试
├── test_bge.py                            # BGE 嵌入测试
├── test_dataset_integration.py            # 数据集整合测试
├── test_integration.py                    # HippoRAG 整合测试
├── test_improved_ner.py                   # 改进的 NER 测试
└── test_failed_sentences.py               # 失败句子分析测试
```

---

## 核心功能测试

### test_amygdala.py
**用途**: 测试 Amygdala 工作流类的完整功能

**测试内容**:
- 基本功能：初始化、添加对话、生成粒子
- 关系映射：粒子与对话的对应关系
- 持久化：保存和加载关系映射
- 辅助方法：查询对话、查询粒子
- 边界情况：空对话、重复对话
- 邻域链接：自动构建粒子链接

**运行方式**:
```bash
python -m test.test_amygdala
```

---

### test_particle.py
**用途**: 测试 Particle 模块的基本功能

**测试内容**:
- Particle 类初始化
- 文本处理和粒子生成
- ParticleEntity 结构验证

**运行方式**:
```bash
python -m test.test_particle
```

---

### test_emotion_v2.py
**用途**: 测试 EmotionV2 情感提取模块

**测试内容**:
- 实体抽取
- 情感描述生成
- 情感向量嵌入
- EmotionNode 创建

**运行方式**:
```bash
python -m test.test_emotion_v2
```

---

## 双曲空间测试

### test_poincare.py
**用途**: 测试双曲空间存储功能

**测试内容**:
- HyperAmyStorage 初始化
- 粒子存储和批量操作
- 向量投影和归一化
- 数据库持久化

**运行方式**:
```bash
python -m test.test_poincare
```

**输出**: 数据库保存在 `./test_poincare_db/`

---

### test_poincare_retrieval.py
**用途**: 测试双曲空间检索功能

**测试内容**:
- 创建并存储 10 个粒子
- 时间演化检索
- 双曲距离计算
- 检索结果排序验证

**运行方式**:
```bash
python -m test.test_poincare_retrieval
```

**输出**:
- 数据库: `./test_retrieval_db/`
- 日志: `log/test_poincare_retrieval.log`

---

### test_particle_poincare_flow.py
**用途**: 测试粒子-双曲空间完整流程

**测试内容**:
- 粒子生命周期：创建 → 存储 → 查询
- 时间演化验证
- ParticleProjector 状态计算
- 双曲距离验证

**运行方式**:
```bash
python -m test.test_particle_poincare_flow
```

**输出**: 数据库保存在 `./test_particle_poincare_flow_db/`

**详细文档**: 参见项目根目录的 `README.md` 中的 "粒子创建 -> 存储 -> 查询完整流程" 章节

---

## 检索功能测试

### test_retrieval.py
**用途**: 测试 Amygdala 检索功能

**测试内容**:
- Particle 模式检索（返回粒子）
- Chunk 模式检索（返回对话片段）
- Chunk 得分排序算法
- 参数配置测试

**运行方式**:
```bash
python -m test.test_retrieval
```

**输出**:
- 数据库: `./test_retrieval_functionality_db/`
- 日志: `log/test_retrieval.log`

---

### test_monte_cristo_retrieval.py
**用途**: Monte Cristo 场景检索测试（真实案例）

**测试场景**:
- Query: "Why did the Count strictly refuse the muscatel grapes..."
- 存储 5 个来自《基督山伯爵》的场景
- 检索相关 chunk 并验证排序

**运行方式**:
```bash
python -m test.test_monte_cristo_retrieval
```

**输出**:
- 数据库: `./test_monte_cristo_db/`
- 验证是否检索到关键 chunk（东方哲学、情感对峙）

---

### test_monte_cristo_retrieval_detailed.py
**用途**: Monte Cristo 详细日志测试（用于展示完整检索流程）

**测试内容**:
- Step 1: 查询文本转粒子
- Step 2: 粒子检索（显示 Top-10）
- Step 3: 粒子到 Chunk 映射和得分计算
- Step 4: 最终检索结果

**运行方式**:
```bash
python -m test.test_monte_cristo_retrieval_detailed 2>&1 | tee log/test_monte_cristo_retrieval.log
```

**输出**:
- 数据库: `./test_monte_cristo_db/`
- 日志: `log/test_monte_cristo_retrieval.log`（完整检索流程）

---

## 辅助模块测试

### test_linking.py
**用途**: 测试双曲空间链接构建

**测试内容**:
- 自动链接发现
- 链接关系验证

**运行方式**:
```bash
python -m test.test_linking
```

---

### test_speed.py
**用途**: 测试速度计算

**测试内容**:
- 基于概率的惊讶值计算
- Speed 属性验证

**运行方式**:
```bash
python -m test.test_speed
```

---

### test_labels.py
**用途**: 测试 Labels（已重命名为 Particle）

**测试内容**:
- Particle 类功能
- 标签和属性

**运行方式**:
```bash
python -m test.test_labels
```

---

### test_entity.py
**用途**: 测试实体抽取

**测试内容**:
- OpenIE 实体抽取
- 实体格式验证

**运行方式**:
```bash
python -m test.test_entity
```

---

### test_sentence.py
**用途**: 测试句子生成

**测试内容**:
- 情感描述生成
- Affective description

**运行方式**:
```bash
python -m test.test_sentence
```

---

## 旧版/辅助测试

### test_emotion.py
**用途**: 测试旧版 Emotion 模块

**运行方式**:
```bash
python -m test.test_emotion
```

---

### test_infer.py
**用途**: 测试推理和 token 概率分析

**运行方式**:
```bash
python -m test.test_infer
```

---

### test_completion_client.py
**用途**: 测试 LLM 客户端

**运行方式**:
```bash
python -m test.test_completion_client
```

---

### test_bge.py
**用途**: 测试 BGE 嵌入模型

**运行方式**:
```bash
python -m test.test_bge
```

---

### test_dataset_integration.py
**用途**: 测试数据集整合

**运行方式**:
```bash
python -m test.test_dataset_integration
```

---

### test_integration.py
**用途**: 测试 HippoRAG 整合

**运行方式**:
```bash
python -m test.test_integration
```

---

### test_improved_ner.py
**用途**: 测试改进的 NER

**运行方式**:
```bash
python -m test.test_improved_ner
```

---

### test_failed_sentences.py
**用途**: 分析失败句子

**运行方式**:
```bash
python -m test.test_failed_sentences
```

---

## 测试数据库位置

测试运行后会在项目根目录创建以下测试数据库：

```
./test_amygdala_db/
./test_poincare_db/
./test_retrieval_db/
./test_particle_poincare_flow_db/
./test_retrieval_functionality_db/
./test_monte_cristo_db/
./hyperamy_db_test_*
./test_amygdala_*
```

清理测试数据库：
```bash
rm -rf ./test_*_db/
rm -rf ./hyperamy_db_test_*
rm -rf ./test_amygdala_*
```

---

## 测试日志

测试日志保存在 `log/` 目录：

- `log/test_amygdala.log`
- `log/test_poincare_retrieval.log`
- `log/test_particle.log`
- `log/test_particle_poincare_flow.log`
- `log/test_retrieval.log`
- `log/test_monte_cristo_retrieval.log`
- `log/test_emotion_v2.log`
- `log/test_failed_sentences.log`

---

## 快速开始

运行所有核心测试：
```bash
python -m test.test_amygdala
python -m test.test_particle
python -m test.test_emotion_v2
python -m test.test_poincare
python -m test.test_poincare_retrieval
python -m test.test_retrieval
```

运行 Monte Cristo 完整测试：
```bash
python -m test.test_monte_cristo_retrieval_detailed 2>&1 | tee log/test_monte_cristo_retrieval.log
```

---

## 注意事项

1. **环境配置**: 所有测试需要配置 `.env` 文件（在 `llm/` 目录下）
2. **数据库清理**: 测试会创建数据库，定期清理以释放空间
3. **日志级别**: 大部分测试使用 INFO 级别日志，可在代码中调整
4. **API 调用**: 测试会调用 LLM API，注意 API 配额
