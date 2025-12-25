# 实体抽取
I love Apple products, especially the iPhone.    



  描述: 测试是否能识别 'Apple products' 而不是单独的 'Apple'

  提取的实体: ['Apple', 'iPhone']

  期望的实体: ['Apple products', 'Apple products, especially the iPhone']

- hippo不关心这个，因为他是一个图谱，他也不关心实体在句子中的含义，因为基本差不多，但是我们不行，实体的情绪就是在句子中体现出来的

# 训练情绪抽取模块
现在是实体抽取-》情绪句子构造-》llm情绪识别
之后该改为实体抽取-》token级别[+-k边缘]聚合-》mlp情绪识别

# 改进聚合（hippo和情绪的都用）
【+-k边缘】解决实体短程问题
- Global Token Injection (全局锚点注入):引入整句话的 [CLS] 向量 $c_{global}$ 作为第 $M+1$ 个节点。它是全句的摘要（包含 "Apple" 的语境）。$$\mathbf{H}^{(0)} = [e_1, e_2, \dots, e_M, c_{global}]$$
- Interaction (交互/消息传递):在这些向量上跑一层标准的 Multi-Head Self-Attention (MHSA)。$$\mathbf{H}^{(1)} = \text{LayerNorm}(\mathbf{H}^{(0)} + \text{MHSA}(\mathbf{H}^{(0)}))$$
- 发生了什么？当计算 "Phone" ($e_2$) 的新向量时，Attention 机制会计算它和 "Apple products" ($e_1$) 以及全局语境 ($c_{global}$) 的相似度。由于 "Phone" 和 "Apple" 在语义空间（BGE）本来就有关联，Attention 权重会很高。于是，$e_2$ “吸取” 了 $e_1$ 的信息。输出的 $e_2'$ 不再是 "Phone"，而是 "Phone (context: Apple)"。