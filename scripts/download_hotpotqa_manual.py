#!/usr/bin/env python
"""
手动下载 HotpotQA 数据集
使用 requests 直接从 HuggingFace 下载
"""
import os
import json
import requests
from typing import List, Dict

def download_file(url: str, save_path: str):
    """下载文件"""
    print(f"📥 下载: {url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        print(f"   文件大小: {total_size / (1024*1024):.2f} MB")
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024*1024) == 0:  # 每MB打印一次
                            print(f"   进度: {progress:.1f}%", end='\r')
        print(f"\n   ✅ 下载完成: {save_path}")
        return True
    else:
        print(f"   ❌ 下载失败: HTTP {response.status_code}")
        return False

def process_hotpotqa_data(json_file: str):
    """处理下载的 HotpotQA 数据"""
    dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'hipporag', 'reproduce', 'dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    corpus_path = os.path.join(dataset_dir, "hotpotqa_corpus.json")
    qa_path = os.path.join(dataset_dir, "hotpotqa.json")
    
    print(f"📖 读取数据文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"   ✅ 读取成功: {len(data)} 个样本")
    
    corpus_dict = {}
    qa_samples = []
    
    for idx, example in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"   处理进度: {idx + 1}/{len(data)}")
        
        try:
            question = example.get('question', '')
            answer = example.get('answer', '')
            supporting_facts = example.get('supporting_facts', [])
            context = example.get('context', [])
            
            # HotpotQA 原始格式：context 是列表，每个元素是 [title, [sentence1, sentence2, ...]]
            if not isinstance(context, list):
                continue
            
            # 构建标题到句子的映射
            title_to_sentences = {}
            for item in context:
                if isinstance(item, list) and len(item) >= 2:
                    title = str(item[0])
                    sentences = item[1] if isinstance(item[1], list) else [str(item[1])]
                    title_to_sentences[title] = sentences
            
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
            print(f"   ⚠️  处理样本 {idx} 时出错: {e}")
            continue
    
    # 转换为列表格式
    corpus = list(corpus_dict.values())
    
    # 保存语料库
    print(f"\n💾 保存语料库: {len(corpus)} 个文档")
    with open(corpus_path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    
    # 保存 QA 数据
    print(f"💾 保存 QA 数据: {len(qa_samples)} 个问题")
    with open(qa_path, 'w', encoding='utf-8') as f:
        json.dump(qa_samples, f, ensure_ascii=False, indent=2)
    
    # 计算文件大小
    corpus_size = os.path.getsize(corpus_path) / (1024 * 1024)
    qa_size = os.path.getsize(qa_path) / (1024 * 1024)
    
    print(f"\n✅ 数据集准备完成!")
    print(f"   语料库: {len(corpus)} 个文档 ({corpus_size:.2f} MB)")
    print(f"   QA 数据: {len(qa_samples)} 个问题 ({qa_size:.2f} MB)")
    
    return corpus, qa_samples

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("HotpotQA 数据集手动下载和处理")
    print("=" * 70)
    
    # 官方下载链接（来自 HotpotQA 官网）
    urls = [
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",  # 官方链接
        "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_dev_distractor_v1.json",
        "https://github.com/hotpotqa/hotpot/raw/master/hotpot_dev_distractor_v1.json",
    ]
    
    temp_file = "/tmp/hotpot_dev_distractor_v1.json"
    
    # 尝试下载
    downloaded = False
    for url in urls:
        print(f"\n尝试 URL: {url}")
        if download_file(url, temp_file):
            downloaded = True
            break
    
    if downloaded:
        print(f"\n处理下载的数据...")
        corpus, qa_samples = process_hotpotqa_data(temp_file)
        os.remove(temp_file)
        print(f"\n✅ 完成!")
    else:
        print(f"\n❌ 所有下载链接都失败")
        print(f"   请手动下载 HotpotQA 数据集并保存为: {temp_file}")
        print(f"   然后运行: python {__file__} {temp_file}")

