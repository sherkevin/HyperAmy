#!/usr/bin/env python
"""检查 HotpotQA 数据格式"""
import json
import requests

url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

print("下载数据文件（前100KB用于检查格式）...")
response = requests.get(url, stream=True)

# 读取前100KB
chunk_size = 1024
data_bytes = b''
for i, chunk in enumerate(response.iter_content(chunk_size=chunk_size)):
    data_bytes += chunk
    if len(data_bytes) > 100 * 1024:  # 100KB
        break

data_str = data_bytes.decode('utf-8', errors='ignore')

# 尝试解析JSON
try:
    # 如果是数组，找到第一个完整的对象
    if data_str.strip().startswith('['):
        # 找到第一个完整的JSON对象
        start = data_str.find('{')
        if start != -1:
            brace_count = 0
            end = start
            for i in range(start, len(data_str)):
                if data_str[i] == '{':
                    brace_count += 1
                elif data_str[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            if end > start:
                first_obj_str = data_str[start:end]
                first_obj = json.loads(first_obj_str)
                print(f"✅ 成功解析第一个样本")
                print(f"数据类型: {type(first_obj)}")
                print(f"样本的键: {list(first_obj.keys())}")
                
                context = first_obj.get('context', None)
                print(f"\ncontext类型: {type(context)}")
                
                if isinstance(context, list):
                    print(f"context是列表，长度: {len(context)}")
                    if len(context) > 0:
                        print(f"context[0]类型: {type(context[0])}")
                        print(f"context[0]内容示例: {str(context[0])[:200]}")
                elif isinstance(context, dict):
                    print(f"context是字典，键: {list(context.keys())}")
                    for key in list(context.keys())[:3]:
                        val = context[key]
                        print(f"  {key}: {type(val)}, 长度: {len(val) if hasattr(val, '__len__') else 'N/A'}")
                        if isinstance(val, list) and len(val) > 0:
                            print(f"    {key}[0]示例: {str(val[0])[:100]}")
                else:
                    print(f"context内容: {str(context)[:300]}")
                
                # 检查其他字段
                print(f"\n其他字段:")
                for key in ['question', 'answer', 'supporting_facts']:
                    if key in first_obj:
                        val = first_obj[key]
                        print(f"  {key}: {type(val)}, 内容: {str(val)[:100]}")
except Exception as e:
    print(f"❌ 解析失败: {e}")
    import traceback
    traceback.print_exc()
