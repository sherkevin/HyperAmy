"""
测试：单独测试未识别到粒子的句子

从 test_amygdala.log 中提取出未识别到粒子的句子，单独测试 EmotionV2 是否能识别到实体。
"""
import logging
from particle.emotion_v2 import EmotionV2, EmotionNode
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 从日志中提取的未识别到粒子的句子
FAILED_SENTENCES = [
    "I love Python programming and I'm excited about machine learning!",
    "I love Python programming and machine learning!",
    "The weather is beautiful today, I feel happy.",
    "I'm frustrated with this bug in my code.",
    "Learning new technologies is exciting and challenging.",
    "The weather is beautiful today.",
    "Second conversation about weather.",
    "Third conversation about coding.",
    "New conversation added after reload.",
    "This is a test conversation.",
    "Machine learning is fascinating.",
    "I'm working on a new project.",
]


def test_failed_sentences():
    """
    测试未识别到粒子的句子
    
    检查 EmotionV2 是否能从这些句子中提取到实体
    """
    print("=" * 100)
    print("测试：未识别到粒子的句子")
    print("=" * 100)
    
    emotion_v2 = EmotionV2()
    
    results = {
        "total": len(FAILED_SENTENCES),
        "success": 0,
        "failed": 0,
        "details": []
    }
    
    print(f"\n总共 {len(FAILED_SENTENCES)} 个句子需要测试\n")
    
    for i, sentence in enumerate(FAILED_SENTENCES, 1):
        print("-" * 100)
        print(f"测试 {i}/{len(FAILED_SENTENCES)}: {sentence}")
        print("-" * 100)
        
        text_id = f"test_failed_{i}"
        
        try:
            # 调用 EmotionV2.process
            nodes = emotion_v2.process(text=sentence, text_id=text_id)
            
            if nodes:
                results["success"] += 1
                print(f"✓ 成功: 提取到 {len(nodes)} 个情绪节点")
                print(f"  实体列表: {[node.entity for node in nodes]}")
                for j, node in enumerate(nodes, 1):
                    print(f"  节点 {j}:")
                    print(f"    - entity_id: {node.entity_id}")
                    print(f"    - entity: {node.entity}")
                    print(f"    - vector_shape: {node.emotion_vector.shape}")
                    print(f"    - vector_norm: {np.linalg.norm(node.emotion_vector):.6f}")
                    print(f"    - vector_preview: {node.emotion_vector[:5].tolist()}")
                
                results["details"].append({
                    "sentence": sentence,
                    "status": "success",
                    "node_count": len(nodes),
                    "entities": [node.entity for node in nodes]
                })
            else:
                results["failed"] += 1
                print(f"✗ 失败: 未提取到任何情绪节点")
                print(f"  原因: 未提取到实体或实体处理失败")
                
                results["details"].append({
                    "sentence": sentence,
                    "status": "failed",
                    "node_count": 0,
                    "entities": []
                })
        
        except Exception as e:
            results["failed"] += 1
            print(f"✗ 异常: {str(e)}")
            import traceback
            print(f"  错误堆栈:\n{traceback.format_exc()}")
            
            results["details"].append({
                "sentence": sentence,
                "status": "error",
                "node_count": 0,
                "entities": [],
                "error": str(e)
            })
        
        print()
    
    # 输出总结
    print("=" * 100)
    print("测试总结")
    print("=" * 100)
    print(f"总句子数: {results['total']}")
    print(f"成功提取实体: {results['success']} ({results['success']/results['total']*100:.1f}%)")
    print(f"失败: {results['failed']} ({results['failed']/results['total']*100:.1f}%)")
    print()
    
    print("成功提取实体的句子:")
    for detail in results["details"]:
        if detail["status"] == "success":
            print(f"  ✓ {detail['sentence']}")
            print(f"    实体: {detail['entities']}")
    
    print("\n未提取到实体的句子:")
    for detail in results["details"]:
        if detail["status"] == "failed":
            print(f"  ✗ {detail['sentence']}")
    
    print("\n出现异常的句子:")
    for detail in results["details"]:
        if detail["status"] == "error":
            print(f"  ✗ {detail['sentence']}")
            print(f"    错误: {detail.get('error', 'Unknown error')}")
    
    print("=" * 100)
    
    return results


def test_with_predefined_entities():
    """
    测试：使用预定义实体
    
    对于未提取到实体的句子，尝试手动提供实体，看看是否能生成粒子
    """
    print("\n" + "=" * 100)
    print("测试：使用预定义实体")
    print("=" * 100)
    
    emotion_v2 = EmotionV2()
    
    # 为一些句子手动提供可能的实体
    test_cases = [
        {
            "sentence": "I love Python programming and I'm excited about machine learning!",
            "entities": ["Python", "machine learning"]
        },
        {
            "sentence": "The weather is beautiful today, I feel happy.",
            "entities": ["weather", "today"]
        },
        {
            "sentence": "I'm frustrated with this bug in my code.",
            "entities": ["bug", "code"]
        },
        {
            "sentence": "Learning new technologies is exciting and challenging.",
            "entities": ["technologies"]
        },
        {
            "sentence": "Machine learning is fascinating.",
            "entities": ["machine learning"]
        },
    ]
    
    print(f"\n测试 {len(test_cases)} 个句子（使用预定义实体）\n")
    
    for i, case in enumerate(test_cases, 1):
        print("-" * 100)
        print(f"测试 {i}/{len(test_cases)}")
        print(f"句子: {case['sentence']}")
        print(f"预定义实体: {case['entities']}")
        print("-" * 100)
        
        text_id = f"test_predefined_{i}"
        
        try:
            nodes = emotion_v2.process(
                text=case['sentence'],
                text_id=text_id,
                entities=case['entities']
            )
            
            if nodes:
                print(f"✓ 成功: 生成 {len(nodes)} 个情绪节点")
                for j, node in enumerate(nodes, 1):
                    print(f"  节点 {j}: entity={node.entity}, vector_shape={node.emotion_vector.shape}")
            else:
                print(f"✗ 失败: 未生成任何情绪节点")
                print(f"  可能原因: 情感描述生成失败或情绪嵌入失败")
        
        except Exception as e:
            print(f"✗ 异常: {str(e)}")
            import traceback
            print(f"  错误堆栈:\n{traceback.format_exc()}")
        
        print()


if __name__ == "__main__":
    # 测试1: 自动提取实体
    results = test_failed_sentences()
    
    # 测试2: 使用预定义实体
    test_with_predefined_entities()
    
    print("\n" + "=" * 100)
    print("所有测试完成")
    print("=" * 100)

