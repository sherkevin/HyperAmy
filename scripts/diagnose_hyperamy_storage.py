#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断HyperAmy存储缺失点数的原因
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 80)
    print("HyperAmy存储诊断")
    print("=" * 80)
    
    # 1. 检查原始数据
    chunks_file = project_root / "data" / "training" / "monte_cristo_train_full.jsonl"
    print(f"\n【步骤1】检查原始数据文件: {chunks_file}")
    
    if not chunks_file.exists():
        print(f"❌ 数据文件不存在: {chunks_file}")
        return
    
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    print(f"✅ 原始chunks总数: {len(chunks)}")
    
    # 2. 检查id_to_content映射文件
    id_to_content_file = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "hyperamy_id_to_content.json"
    print(f"\n【步骤2】检查id_to_content映射文件: {id_to_content_file}")
    
    if not id_to_content_file.exists():
        print(f"❌ 映射文件不存在: {id_to_content_file}")
        return
    
    with open(id_to_content_file, 'r', encoding='utf-8') as f:
        id_to_content = json.load(f)
    
    print(f"✅ 映射文件中的点数: {len(id_to_content)}")
    
    # 3. 检查哪些chunks缺失了
    print(f"\n【步骤3】分析缺失的chunks...")
    
    stored_chunk_indices = set()
    for point_id in id_to_content.keys():
        # point_id格式应该是 chunk_{index}
        if point_id.startswith('chunk_'):
            try:
                idx = int(point_id.split('_')[1])
                stored_chunk_indices.add(idx)
            except:
                pass
    
    total_chunks = len(chunks)
    missing_indices = set(range(total_chunks)) - stored_chunk_indices
    
    print(f"  总chunks数: {total_chunks}")
    print(f"  已存储chunks数: {len(stored_chunk_indices)}")
    print(f"  缺失chunks数: {len(missing_indices)}")
    
    if missing_indices:
        print(f"\n  缺失的chunk索引（前20个）: {sorted(list(missing_indices))[:20]}")
        
        # 4. 检查缺失chunks的特征
        print(f"\n【步骤4】分析缺失chunks的特征...")
        
        missing_with_short_text = 0
        missing_with_no_text = 0
        missing_with_errors = []
        
        for idx in sorted(list(missing_indices))[:100]:  # 只检查前100个
            chunk = chunks[idx]
            text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
            
            if not isinstance(text, str):
                missing_with_no_text += 1
                missing_with_errors.append((idx, "非字符串类型"))
            elif len(text.strip()) <= 20:
                missing_with_short_text += 1
                missing_with_errors.append((idx, f"文本长度不足 (len={len(text.strip())})"))
            else:
                missing_with_errors.append((idx, "未知原因"))
        
        print(f"  缺失chunks中文本为空的: {missing_with_no_text}")
        print(f"  缺失chunks中文本长度<=20的: {missing_with_short_text}")
        print(f"  其他原因: {len(missing_indices) - missing_with_short_text - missing_with_no_text}")
        
        if missing_with_errors:
            print(f"\n  缺失chunks示例（前10个）:")
            for idx, reason in missing_with_errors[:10]:
                chunk = chunks[idx]
                text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
                print(f"    索引{idx}: {reason}, text_len={len(text) if isinstance(text, str) else 'N/A'}")
    else:
        print("  ✅ 所有chunks都已存储！")
    
    # 5. 检查ChromaDB实际存储数量
    print(f"\n【步骤5】检查ChromaDB实际存储数量...")
    try:
        from poincare.storage import HyperAmyStorage
        
        storage_path = project_root / "outputs" / "three_methods_comparison_monte_cristo" / "hyperamy_db"
        if storage_path.exists():
            try:
                storage = HyperAmyStorage(persist_path=str(storage_path))
                count = storage.collection.count()
                print(f"  ✅ ChromaDB存储的点数: {count}")
                
                if count != len(id_to_content):
                    print(f"  ⚠️  警告: ChromaDB点数 ({count}) 与映射文件点数 ({len(id_to_content)}) 不一致！")
            except Exception as e:
                print(f"  ⚠️  ChromaDB读取失败: {e}")
                print(f"     可能原因：数据库正在被其他进程使用或数据库损坏")
        else:
            print(f"  ❌ 存储目录不存在: {storage_path}")
    except Exception as e:
        print(f"  ⚠️  无法检查ChromaDB: {e}")
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)

if __name__ == "__main__":
    main()
