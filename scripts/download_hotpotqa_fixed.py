#!/usr/bin/env python
"""
下载和准备 HotpotQA 数据集 - 修复版本
使用 HuggingFace Hub 直接下载，避免 datasets 库的路径模式问题
"""
import os
import json
from typing import List, Dict

try:
    from huggingface_hub import hf_hub_download
    import pandas as pd
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    print("⚠️  huggingface_hub 或 pandas 库未安装")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

def download_hotpotqa_via_hub():
    """通过 HuggingFace Hub 直接下载"""
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    corpus_path = os.path.join(dataset_dir, "hotpotqa_corpus.json")
    qa_path = os.path.join(dataset_dir, "hotpotqa.json")
    
    if not HAS_HF_HUB:
        print("❌ 需要安装: pip install huggingface_hub pandas")
        return None, None
    
    try:
        print("📥 从 HuggingFace Hub 下载 HotpotQA 数据集...")
        
        # 下载验证集文件
        validation_file = hf_hub_download(
            repo_id="hotpot_qa",
            filename="distractor_qa/dev.json",
            repo_type="dataset"
        )
        
        print(f"   ✅ 文件下载成功: {validation_file}")
        
        # 读取 JSON 文件
        with open(validation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"   ✅ 读取成功: {len(data)} 个样本")
        
        # 处理数据
        corpus_dict = {}
        qa_samples = []
        
        for example in data:
            try:
                question = example.get('question', '')
                answer = example.get('answer', '')
                supporting_facts = example.get('supporting_facts', [])
                context = example.get('context', {})
                
                if not isinstance(context, dict):
                    continue
                
                titles = context.get('title', [])
                sentences_list = context.get('sentences', [])
                
                # 构建标题到句子的映射
                title_to_sentences = {}
                for i, title in enumerate(titles):
                    if i < len(sentences_list):
                        sents = sentences_list[i] if isinstance(sentences_list[i], list) else [str(sentences_list[i])]
                        title_to_sentences[title] = sents
                
                # 构建相关文档列表
                relevant_docs = []
                if supporting_facts:
                    for fact in supporting_facts:
                        if isinstance(fact, (list, tuple)) and len(fact) >= 2:
                            title = str(fact[0])
                            sent_idx = fact[1]
                            if title in title_to_sentences:
                                sents = title_to_sentences[title]
                                if isinstance(sent_idx, int) and 0 <= sent_idx < len(sents):
                                    doc_text = f"{title}\n{sents[sent_idx]}"
                                    if doc_text not in relevant_docs:
                                        relevant_docs.append(doc_text)
                
                if not relevant_docs:
                    for title, sents in title_to_sentences.items():
                        doc_text = f"{title}\n{' '.join(sents)}"
                        relevant_docs.append(doc_text)
                
                qa_samples.append({
                    "question": question,
                    "answer": answer,
                    "relevant_docs": relevant_docs[:5]
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
        import traceback
        traceback.print_exc()
        return None, None

def download_hotpotqa_fallback():
    """回退方法：尝试修复 fsspec 版本问题"""
    try:
        print("尝试修复 fsspec 版本兼容性...")
        import subprocess
        import sys
        
        # 尝试降级 fsspec
        print("   降级 fsspec 到兼容版本...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fsspec<2024.0.0", "--quiet"])
        
        # 重新导入
        from datasets import load_dataset
        dataset = load_dataset("hotpot_qa", "distractor", split="validation")
        print(f"   ✅ 成功: {len(dataset)} 个样本")
        return dataset
    except Exception as e:
        print(f"   ❌ 回退方法失败: {e}")
        return None

if __name__ == "__main__":
    print("=" * 70)
    print("HotpotQA 数据集下载和准备 - 修复版本")
    print("=" * 70)
    
    # 首先尝试通过 Hub 直接下载
    corpus, qa_samples = download_hotpotqa_via_hub()
    
    if not corpus or not qa_samples:
        print("\n⚠️  直接下载失败，尝试回退方法...")
        dataset = download_hotpotqa_fallback()
        if dataset:
            # 处理数据集...
            print("   需要进一步处理数据集...")
    
    if corpus and qa_samples:
        print(f"\n✅ 数据集准备完成!")
        print(f"   语料库: {len(corpus)} 个文档")
        print(f"   QA 数据: {len(qa_samples)} 个问题")
        
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
        print(f"\n⚠️  数据集准备失败")

