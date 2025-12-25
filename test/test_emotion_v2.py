"""
测试 EmotionV2 类

测试完整的情绪节点生成流程：
1. 实体抽取
2. 情感描述生成（情绪词列表）
3. 情绪嵌入向量生成
"""
from point_label.emotion_v2 import EmotionV2, EmotionNode
import numpy as np
from collections import defaultdict


def test_basic_processing():
    """测试基本处理流程"""
    print("=" * 60)
    print("测试 1: 基本处理流程")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "Barack Obama was the 44th president of the United States. He served from 2009 to 2017."
    text_id = "test_text_1"
    
    print(f"\n原始文本: {text}")
    print(f"文本 ID: {text_id}")
    
    nodes = emotion_v2.process(text, text_id)
    
    print(f"\n生成的情绪节点数量: {len(nodes)}")
    
    for node in nodes:
        print(f"\n节点信息:")
        print(f"  - 实体 ID: {node.entity_id}")
        print(f"  - 实体名称: {node.entity}")
        print(f"  - 情绪向量形状: {node.emotion_vector.shape}")
        print(f"  - 情绪向量维度: {len(node.emotion_vector)}")
        print(f"  - 原文本 ID: {node.text_id}")
        print(f"  - 情绪向量前5维: {node.emotion_vector[:5]}")
        print(f"  - 情绪向量L2范数: {np.linalg.norm(node.emotion_vector):.4f}")
    
    assert len(nodes) > 0, "应该至少生成一个节点"
    assert all(isinstance(node, EmotionNode) for node in nodes), "所有节点都应该是 EmotionNode 类型"
    assert all(node.text_id == text_id for node in nodes), "所有节点的 text_id 应该匹配"
    
    print("\n✓ 基本处理流程测试通过")


def test_with_predefined_entities():
    """测试使用预定义实体"""
    print("\n" + "=" * 60)
    print("测试 2: 使用预定义实体")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "Apple Inc. is a technology company founded by Steve Jobs."
    text_id = "test_text_2"
    entities = ["Apple Inc.", "Steve Jobs"]
    
    print(f"\n原始文本: {text}")
    print(f"预定义实体: {entities}")
    
    nodes = emotion_v2.process(text, text_id, entities=entities)
    
    print(f"\n生成的情绪节点数量: {len(nodes)}")
    
    for node in nodes:
        print(f"\n节点信息:")
        print(f"  - 实体: {node.entity}")
        print(f"  - 情绪向量形状: {node.emotion_vector.shape}")
    
    assert len(nodes) == len(entities), f"应该生成 {len(entities)} 个节点"
    
    print("\n✓ 预定义实体测试通过")


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 60)
    print("测试 3: 批量处理")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    texts = [
        "Microsoft was founded by Bill Gates.",
        "Google is a search engine company.",
        "Amazon was started by Jeff Bezos."
    ]
    
    text_ids = ["text_1", "text_2", "text_3"]
    
    print(f"\n批量处理 {len(texts)} 个文本")
    
    all_nodes = emotion_v2.batch_process(texts, text_ids)
    
    print(f"\n总共生成 {len(all_nodes)} 个情绪节点")
    
    # 按 text_id 分组统计
    nodes_by_text = defaultdict(list)
    for node in all_nodes:
        nodes_by_text[node.text_id].append(node)
    
    for text_id, nodes in nodes_by_text.items():
        print(f"  - {text_id}: {len(nodes)} 个节点")
    
    assert len(all_nodes) > 0, "应该至少生成一个节点"
    
    print("\n✓ 批量处理测试通过")


def test_emotion_node_structure():
    """测试 EmotionNode 结构"""
    print("\n" + "=" * 60)
    print("测试 4: EmotionNode 结构验证")
    print("=" * 60)
    
    # 创建测试节点
    test_vector = np.random.rand(2560)  # 模拟嵌入向量
    
    node = EmotionNode(
        entity_id="test_entity_1",
        entity="Test Entity",
        emotion_vector=test_vector,
        text_id="test_text_1"
    )
    
    print(f"\n节点结构:")
    print(f"  - entity_id: {node.entity_id} (类型: {type(node.entity_id)})")
    print(f"  - entity: {node.entity} (类型: {type(node.entity)})")
    print(f"  - emotion_vector: shape={node.emotion_vector.shape}, dtype={node.emotion_vector.dtype}")
    print(f"  - text_id: {node.text_id} (类型: {type(node.text_id)})")
    
    assert isinstance(node.entity_id, str), "entity_id 应该是字符串"
    assert isinstance(node.entity, str), "entity 应该是字符串"
    assert isinstance(node.emotion_vector, np.ndarray), "emotion_vector 应该是 numpy 数组"
    assert isinstance(node.text_id, str), "text_id 应该是字符串"
    
    print("\n✓ EmotionNode 结构验证通过")


def test_empty_text():
    """测试空文本处理"""
    print("\n" + "=" * 60)
    print("测试 5: 空文本处理")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = ""
    text_id = "empty_text"
    
    nodes = emotion_v2.process(text, text_id)
    
    print(f"\n空文本处理结果: {len(nodes)} 个节点")
    
    assert len(nodes) == 0, "空文本应该返回空列表"
    
    print("\n✓ 空文本处理测试通过")


def test_no_entities():
    """测试无实体文本"""
    print("\n" + "=" * 60)
    print("测试 6: 无实体文本处理")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "This is a simple sentence without any named entities."
    text_id = "no_entities_text"
    
    nodes = emotion_v2.process(text, text_id)
    
    print(f"\n无实体文本处理结果: {len(nodes)} 个节点")
    
    # 可能返回空列表，也可能提取到一些实体
    print(f"\n✓ 无实体文本处理测试完成（结果: {len(nodes)} 个节点）")


def test_emotion_vector_properties():
    """测试情绪向量属性"""
    print("\n" + "=" * 60)
    print("测试 7: 情绪向量属性验证")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "Elon Musk is the CEO of Tesla and SpaceX."
    text_id = "test_vector_properties"
    
    nodes = emotion_v2.process(text, text_id)
    
    if len(nodes) > 0:
        node = nodes[0]
        vector = node.emotion_vector
        
        print(f"\n情绪向量属性:")
        print(f"  - 形状: {vector.shape}")
        print(f"  - 维度: {len(vector)}")
        print(f"  - 数据类型: {vector.dtype}")
        print(f"  - 最小值: {vector.min():.6f}")
        print(f"  - 最大值: {vector.max():.6f}")
        print(f"  - 均值: {vector.mean():.6f}")
        print(f"  - L2 范数: {np.linalg.norm(vector):.6f}")
        
        assert len(vector.shape) == 1, "向量应该是一维数组"
        assert vector.dtype in [np.float32, np.float64], "向量应该是浮点类型"
        
        print("\n✓ 情绪向量属性验证通过")
    else:
        print("\n⚠ 未生成节点，跳过向量属性验证")


def test_emotion_keywords_format():
    """测试情绪词格式（验证 Sentence 返回的是情绪词列表）"""
    print("\n" + "=" * 60)
    print("测试 8: 情绪词格式验证")
    print("=" * 60)
    
    from utils.sentence import Sentence
    
    sentence_processor = Sentence()
    
    text = "Barack Obama was the 44th president of the United States."
    entities = ["Barack Obama"]
    
    descriptions = sentence_processor.generate_affective_descriptions(text, entities)
    
    print(f"\n原始文本: {text}")
    print(f"实体: {entities[0]}")
    
    if entities[0] in descriptions:
        description = descriptions[entities[0]]
        print(f"\n生成的情感描述: {description}")
        
        # 检查是否是逗号分隔的词列表格式
        words = [w.strip() for w in description.split(',')]
        print(f"\n解析后的情绪词: {words}")
        print(f"情绪词数量: {len(words)}")
        
        # 验证格式：应该是3-8个词
        assert 3 <= len(words) <= 8, f"情绪词数量应该在3-8之间，实际为{len(words)}"
        assert all(len(w) > 0 for w in words), "所有词都应该非空"
        
        print("\n✓ 情绪词格式验证通过")
    else:
        print("\n⚠ 未生成情感描述，跳过格式验证")


def test_multiple_entities_same_text():
    """测试同一文本中的多个实体"""
    print("\n" + "=" * 60)
    print("测试 9: 同一文本中的多个实体")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "Apple Inc. was founded by Steve Jobs in California."
    text_id = "test_multiple_entities"
    
    nodes = emotion_v2.process(text, text_id)
    
    print(f"\n原始文本: {text}")
    print(f"文本 ID: {text_id}")
    print(f"\n生成的情绪节点数量: {len(nodes)}")
    
    # 检查所有节点都有相同的 text_id
    unique_text_ids = set(node.text_id for node in nodes)
    assert len(unique_text_ids) == 1, f"所有节点应该有相同的 text_id，实际有 {len(unique_text_ids)} 个不同的 text_id"
    assert list(unique_text_ids)[0] == text_id, "text_id 应该匹配"
    
    # 检查 entity_id 的唯一性
    entity_ids = [node.entity_id for node in nodes]
    assert len(entity_ids) == len(set(entity_ids)), "所有 entity_id 应该是唯一的"
    
    print("\n节点详情:")
    for i, node in enumerate(nodes):
        print(f"  节点 {i+1}:")
        print(f"    - 实体: {node.entity}")
        print(f"    - 实体 ID: {node.entity_id}")
        print(f"    - 情绪向量维度: {len(node.emotion_vector)}")
    
    print("\n✓ 多实体处理测试通过")


def test_entity_id_format():
    """测试 entity_id 格式"""
    print("\n" + "=" * 60)
    print("测试 10: entity_id 格式验证")
    print("=" * 60)
    
    emotion_v2 = EmotionV2()
    
    text = "Microsoft was founded by Bill Gates."
    text_id = "test_entity_id_format"
    
    nodes = emotion_v2.process(text, text_id)
    
    if len(nodes) > 0:
        print(f"\n文本 ID: {text_id}")
        print(f"生成的节点数量: {len(nodes)}")
        
        for i, node in enumerate(nodes):
            print(f"\n节点 {i+1}:")
            print(f"  - entity_id: {node.entity_id}")
            print(f"  - 实体: {node.entity}")
            
            # 验证 entity_id 格式：应该是 text_id_entity_{idx}
            expected_prefix = f"{text_id}_entity_"
            assert node.entity_id.startswith(expected_prefix), \
                f"entity_id 应该以 '{expected_prefix}' 开头，实际为 '{node.entity_id}'"
            
            # 验证索引部分
            idx_str = node.entity_id[len(expected_prefix):]
            assert idx_str.isdigit(), f"entity_id 的索引部分应该是数字，实际为 '{idx_str}'"
            assert int(idx_str) == i, f"entity_id 的索引应该为 {i}，实际为 {idx_str}"
        
        print("\n✓ entity_id 格式验证通过")
    else:
        print("\n⚠ 未生成节点，跳过 entity_id 格式验证")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("EmotionV2 测试套件")
    print("=" * 60)
    
    try:
        test_basic_processing()
        test_with_predefined_entities()
        test_batch_processing()
        test_emotion_node_structure()
        test_empty_text()
        test_no_entities()
        test_emotion_vector_properties()
        test_emotion_keywords_format()
        test_multiple_entities_same_text()
        test_entity_id_format()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

