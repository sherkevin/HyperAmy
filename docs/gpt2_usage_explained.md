# GPT-2 模型调用详解

## 1. 模型加载

### 代码位置
`src/data_prep.py` 第 195-204 行

### 加载过程

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 1. 加载Tokenizer
self.surprisal_tokenizer = AutoTokenizer.from_pretrained('gpt2')

# 2. 加载模型
self.surprisal_model = AutoModelForCausalLM.from_pretrained('gpt2')

# 3. 如果有GPU，移到GPU上
if torch.cuda.is_available():
    self.surprisal_model = self.surprisal_model.cuda()

# 4. 设置为评估模式（不计算梯度）
self.surprisal_model.eval()

# 5. 设置pad_token（GPT-2默认没有pad_token）
if self.surprisal_tokenizer.pad_token is None:
    self.surprisal_tokenizer.pad_token = self.surprisal_tokenizer.eos_token
```

### 说明
- **模型来源**：HuggingFace Hub (`gpt2`)
- **模型类型**：`AutoModelForCausalLM` - 因果语言模型（用于生成和计算loss）
- **设备**：优先使用GPU（CUDA），否则使用CPU
- **评估模式**：`eval()` 关闭dropout和batch normalization的训练行为

---

## 2. 计算Surprisal Score的完整流程

### 代码位置
`src/data_prep.py` 第 258-292 行

### 详细步骤

```python
def compute_surprisal_score(self, text: str) -> float:
    # ========== 步骤1: Tokenize文本 ==========
    inputs = self.surprisal_tokenizer(
        text, 
        return_tensors='pt',        # 返回PyTorch张量
        truncation=True,            # 如果超过512个token，截断
        max_length=512              # 最大长度限制
    )
    # inputs 是一个字典，包含：
    # - 'input_ids': token ID序列 [batch_size, seq_len]
    # - 'attention_mask': 注意力掩码 [batch_size, seq_len]
    
    # ========== 步骤2: 移到GPU（如果可用） ==========
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # ========== 步骤3: 计算Loss ==========
    with torch.no_grad():  # 不计算梯度，节省内存和加速
        # 关键：传入 labels=inputs['input_ids']
        # 这样模型会计算每个token的预测loss
        outputs = self.surprisal_model(
            **inputs, 
            labels=inputs['input_ids']  # 标签就是输入本身（自监督学习）
        )
        
        # outputs.loss 是平均的交叉熵损失
        loss = outputs.loss  # shape: scalar tensor
        
        # ========== 步骤4: 计算PPL (Perplexity) ==========
        # PPL = exp(loss)
        # 表示模型对这段文本的"困惑程度"
        ppl = torch.exp(loss).item()  # 转换为Python float
    
    # ========== 步骤5: 归一化到[0, 1] ==========
    log_ppl = np.log(ppl + 1)
    max_log_ppl = np.log(1000 + 1)  # 假设最大PPL为1000
    surprisal_score = min(log_ppl / max_log_ppl, 1.0)
    
    return surprisal_score
```

---

## 3. 关键点解释

### 3.1 为什么使用 `labels=inputs['input_ids']`？

GPT-2 是自回归语言模型，训练目标是预测下一个token。当我们传入 `labels` 时：
- 模型会计算每个位置的预测概率
- 与真实token（即输入本身）比较
- 得到交叉熵损失

**示例**：
```
输入: "She stopped, trembling"
Token序列: [She, stopped, ,, trembling]

模型预测:
- 位置0: 预测"stopped"的概率
- 位置1: 预测","的概率
- 位置2: 预测"trembling"的概率
- ...

Loss = 平均交叉熵（预测概率 vs 真实token）
```

### 3.2 PPL (Perplexity) 的含义

**困惑度（Perplexity）** = exp(loss)

- **PPL低**（接近1）：模型对文本很"熟悉"，预测准确 → 文本符合模型预期
- **PPL高**（>100）：模型对文本很"困惑"，预测不准 → 文本意外/不符合预期

**在我们的应用中**：
- 高PPL → 高惊奇度 → 文本包含"意外"或"颠覆性"内容
- 这些内容往往对应情节转折、危机时刻等

### 3.3 归一化公式

```python
log_ppl = np.log(ppl + 1)
max_log_ppl = np.log(1000 + 1)
surprisal_score = min(log_ppl / max_log_ppl, 1.0)
```

**为什么用对数归一化？**
- PPL的范围很大（1到数千），直接归一化会导致大部分值接近0
- 使用对数可以压缩范围，让分布更均匀

**示例**：
- PPL = 10 → log(11) / log(1001) ≈ 0.23
- PPL = 100 → log(101) / log(1001) ≈ 0.46
- PPL = 1000 → log(1001) / log(1001) = 1.0

---

## 4. 实际调用示例

### 示例1：普通文本

```python
text = "The weather is nice today."
# Tokenize → 计算loss → PPL ≈ 20-50 → surprisal_score ≈ 0.1-0.2
```

### 示例2：高惊奇度文本

```python
text = "She could feel the emptiness, the vast black gulfs of air that yawned around her."
# Tokenize → 计算loss → PPL ≈ 100-200 → surprisal_score ≈ 0.4-0.6
```

### 示例3：非常意外的文本

```python
text = "Suddenly, the dragon appeared and everything changed."
# Tokenize → 计算loss → PPL ≈ 500-1000 → surprisal_score ≈ 0.8-1.0
```

---

## 5. 依赖库

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
```

**安装**：
```bash
pip install transformers torch
```

**模型下载**：
- 首次运行时会自动从HuggingFace Hub下载
- 模型大小：约500MB
- 下载位置：`~/.cache/huggingface/transformers/`

---

## 6. 性能考虑

### 内存使用
- GPT-2模型约500MB
- 每个文本的计算需要临时内存（取决于文本长度）

### 计算速度
- CPU：约50-200ms/文本（取决于长度）
- GPU：约5-20ms/文本

### 优化建议
- 使用GPU加速（如果有）
- 批量处理（当前代码是逐个处理）
- 文本截断到512 tokens（已实现）

---

## 7. 与Emotion Score的对比

| 维度 | Emotion Score | Surprisal Score |
|------|---------------|-----------------|
| 模型 | roberta-base-go_emotions | GPT-2 |
| 检测内容 | 负面情感（fear, anger等） | 文本的意外性 |
| 计算方式 | 情感分类概率 | 语言模型困惑度 |
| 反映 | 情感强度 | 情节意外性 |
| 权重 | 70% | 30% |

两者结合（Mass = 0.7 × Emotion + 0.3 × Surprisal）能够同时捕捉：
1. **情感信号**：恐惧、愤怒等负面情感
2. **意外信号**：情节转折、颠覆性事件

