#!/usr/bin/env python
"""
下载和准备 HotpotQA 数据集

从 HuggingFace 下载 HotpotQA 数据集并转换为项目格式
"""
import os
import json
from typing import List, Dict

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠️  datasets 库未安装，将使用备用方法")

def download_hotpotqa(num_examples: int = 100):
    """
    下载 HotpotQA 数据集
    
    Args:
        num_examples: 下载的样本数量（用于快速测试）
    """
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    corpus_path = os.path.join(dataset_dir, "hotpotqa_corpus.json")
    qa_path = os.path.join(dataset_dir, "hotpotqa.json")
    
    if HAS_DATASETS:
        print("从 HuggingFace 下载 HotpotQA 数据集...")
        try:
            # 下载 dev 集（较小，适合测试）
            dataset = load_dataset("hotpot_qa", "distractor", split="validation")
            print(f"   ✅ 下载成功: {len(dataset)} 个样本")
            
            # 限制样本数量（用于快速测试）
            if num_examples > 0:
                dataset = dataset.select(range(min(num_examples, len(dataset))))
                print(f"   ✅ 使用前 {len(dataset)} 个样本")
            
            # 提取语料库（所有文档）
            corpus_dict = {}
            qa_samples = []
            
            for example in dataset:
                try:
                    # 提取问题和答案
                    question = example['question']
                    answer = example['answer']
                    
                    # 提取相关文档（supporting facts）
                    supporting_facts = example.get('supporting_facts', [])
                    context = example.get('context', {})
                    
                    # HotpotQA 的 context 是 dict，包含 'title' 和 'sentences' 列表
                    if not isinstance(context, dict):
                        continue
                    
                    titles = context.get('title', [])
                    sentences_list = context.get('sentences', [])
                    
                    # 构建标题到句子的映射
                    title_to_sentences = {}
                    for i, title in enumerate(titles):
                        if i < len(sentences_list):
                            # sentences_list[i] 是一个句子列表
                            sents = sentences_list[i] if isinstance(sentences_list[i], list) else [str(sentences_list[i])]
                            title_to_sentences[title] = sents
                    
                    # 构建相关文档列表（gold_docs）- 使用 supporting_facts
                    relevant_docs = []
                    if supporting_facts:
                        for fact in supporting_facts:
                            if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                                title = str(fact[0])
                                sent_idx = fact[1]
                                if title in title_to_sentences:
                                    sents = title_to_sentences[title]
                                    if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                                        # 使用单个句子作为文档
                                        doc_text = f"{title}\n{sents[sent_idx]}"
                                        if doc_text not in relevant_docs:
                                            relevant_docs.append(doc_text)
                    
                    # 如果没有 supporting_facts，使用所有文档
                    if not relevant_docs:
                        for title, sents in title_to_sentences.items():
                            doc_text = f"{title}\n{' '.join(sents)}"
                            relevant_docs.append(doc_text)
                    
                    qa_samples.append({
                        "question": question,
                        "answer": answer,
                        "relevant_docs": relevant_docs[:5]  # 最多5个相关文档
                    })
                    
                    # 收集所有文档到语料库
                    for title, sents in title_to_sentences.items():
                        doc_text = f"{title}\n{' '.join(sents)}"
                        doc_id = f"{title}_{abs(hash(doc_text)) % 1000000}"
                        if doc_id not in corpus_dict:
                            corpus_dict[doc_id] = {
                                "title": title,
                                "text": ' '.join(sents)
                            }
                except Exception as e:
                    print(f"   ⚠️  处理样本时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # 转换为列表格式
            corpus = list(corpus_dict.values())
            
            # 保存语料库
            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(corpus, f, ensure_ascii=False, indent=2)
            print(f"   ✅ 语料库已保存: {len(corpus)} 个文档 -> {corpus_path}")
            
            # 保存 QA 数据
            with open(qa_path, 'w', encoding='utf-8') as f:
                json.dump(qa_samples, f, ensure_ascii=False, indent=2)
            print(f"   ✅ QA 数据已保存: {len(qa_samples)} 个问题 -> {qa_path}")
            
            return corpus, qa_samples
            
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            print(f"   ⚠️  将使用现有数据集或创建示例数据集")
            return None, None
    else:
        print("⚠️  datasets 库未安装，无法自动下载")
        print(f"   请手动下载 HotpotQA 数据集或安装: pip install datasets")
        return None, None

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("HotpotQA 数据集下载和准备")
    print("=" * 70)
    
    # 支持命令行参数：python download_hotpotqa.py [num_examples]
    # num_examples=0 表示下载完整数据集
    if len(sys.argv) > 1:
        try:
            num_examples = int(sys.argv[1])
            if num_examples == 0:
                print("📥 下载完整 HotpotQA 数据集...")
            else:
                print(f"📥 下载前 {num_examples} 个样本...")
        except ValueError:
            print("⚠️  参数无效，使用默认值（完整数据集）")
            num_examples = 0
    else:
        # 默认下载完整数据集
        num_examples = 0
        print("📥 下载完整 HotpotQA 数据集（默认）...")
        print("   提示: 可以传入参数指定数量，如: python download_hotpotqa.py 100")
    
    # 下载数据集
    corpus, qa_samples = download_hotpotqa(num_examples=num_examples)
    
    if corpus and qa_samples:
        print(f"\n✅ 数据集准备完成!")
        print(f"   语料库: {len(corpus)} 个文档")
        print(f"   QA 数据: {len(qa_samples)} 个问题")
        
        # 计算文件大小
        import os
        corpus_path = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset', 'hotpotqa_corpus.json')
        qa_path = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset', 'hotpotqa.json')
        if os.path.exists(corpus_path):
            size_mb = os.path.getsize(corpus_path) / (1024 * 1024)
            print(f"   语料库文件大小: {size_mb:.2f} MB")
        if os.path.exists(qa_path):
            size_mb = os.path.getsize(qa_path) / (1024 * 1024)
            print(f"   QA 文件大小: {size_mb:.2f} MB")
    else:
        print(f"\n⚠️  数据集准备失败，请检查网络连接或手动下载")

