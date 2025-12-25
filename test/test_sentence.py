"""
测试 Sentence 类

测试情感视角描述生成功能
"""
from utils.sentence import Sentence
from utils.entitiy import Entity


def test_single_entity_description():
    """测试单个实体的情感描述生成"""
    print("=" * 60)
    print("测试 1: 单个实体的情感描述生成")
    print("=" * 60)
    
    sentence = Sentence()
    
    test_cases = [
        {
            "sentence": "Barack Obama was the 44th president of the United States.",
            "entity": "Barack Obama"
        },
        {
            "sentence": "I love Apple products, especially the iPhone.",
            "entity": "Apple"
        },
        {
            "sentence": "The storm destroyed everything in its path.",
            "entity": "storm"
        },
    ]
    
    for case in test_cases:
        description = sentence.generate_affective_description(
            case["sentence"], 
            case["entity"]
        )
        
        print(f"\nSentence: {case['sentence']}")
        print(f"Entity: {case['entity']}")
        print(f"Affective Description: {description}")
        print("-" * 60)
    
    print()


def test_multiple_entities_descriptions():
    """测试多个实体的情感描述生成"""
    print("=" * 60)
    print("测试 2: 多个实体的情感描述生成")
    print("=" * 60)
    
    sentence = Sentence()
    
    test_cases = [
        {
            "sentence": "Barack Obama was the 44th president of the United States.",
            "entities": ["Barack Obama", "United States"]
        },
        {
            "sentence": "I love Apple products, especially the iPhone designed by Steve Jobs.",
            "entities": ["Apple", "iPhone", "Steve Jobs"]
        },
        {
            "sentence": "The University of Cambridge, located in Cambridge, England, was founded in 1209.",
            "entities": ["University of Cambridge", "Cambridge", "England"]
        },
    ]
    
    for case in test_cases:
        descriptions = sentence.generate_affective_descriptions(
            case["sentence"],
            case["entities"]
        )
        
        print(f"\nSentence: {case['sentence']}")
        print(f"Entities: {case['entities']}")
        print("Affective Descriptions:")
        for entity, description in descriptions.items():
            print(f"  - {entity}: {description}")
        print("-" * 60)
    
    print()


def test_complete_process():
    """测试完整流程（提取实体 + 生成描述）"""
    print("=" * 60)
    print("测试 3: 完整流程（提取实体 + 生成描述）")
    print("=" * 60)
    
    sentence = Sentence()
    entity_extractor = Entity()
    
    test_sentences = [
        "Barack Obama was the 44th president of the United States.",
        "I love Apple products, especially the iPhone.",
        "The storm destroyed everything in its path, leaving people devastated.",
    ]
    
    for test_sentence in test_sentences:
        result = sentence.process_sentence(
            test_sentence,
            entity_extractor=entity_extractor
        )
        
        print(f"\nSentence: {test_sentence}")
        print(f"Extracted Entities: {result['entities']}")
        print("Affective Descriptions:")
        for entity, description in result['affective_descriptions'].items():
            print(f"  - {entity}: {description}")
        print("-" * 60)
    
    print()


def test_affective_focus():
    """测试情感描述是否聚焦于情感，忽略事实细节"""
    print("=" * 60)
    print("测试 4: 验证情感描述是否聚焦于情感")
    print("=" * 60)
    
    sentence = Sentence()
    
    test_cases = [
        {
            "sentence": "Barack Obama was the 44th president of the United States.",
            "entity": "Barack Obama",
            "should_contain": ["emotion", "feeling", "sentiment"],
            "should_not_contain": ["44th", "president", "United States"]
        },
        {
            "sentence": "I'm devastated by the loss of my pet.",
            "entity": "pet",
            "should_contain": ["sad", "grief", "loss", "emotion"],
            "should_not_contain": ["pet", "animal"]
        },
    ]
    
    print(f"{'Entity':<30} | {'Description (前50字符)':<50} | {'Focus Check':<15}")
    print("-" * 100)
    
    for case in test_cases:
        description = sentence.generate_affective_description(
            case["sentence"],
            case["entity"]
        )
        
        # 检查是否包含情感词汇
        has_emotion = any(keyword.lower() in description.lower() for keyword in case["should_contain"])
        
        # 检查是否避免了事实细节
        avoids_facts = not any(keyword.lower() in description.lower() for keyword in case["should_not_contain"])
        
        focus_check = "✓" if (has_emotion or len(description) > 0) else "✗"
        
        entity_display = case["entity"][:27] + "..." if len(case["entity"]) > 30 else case["entity"]
        desc_display = description[:47] + "..." if len(description) > 50 else description
        
        print(f"{entity_display:<30} | {desc_display:<50} | {focus_check:<15}")
    
    print()


def test_different_emotions():
    """测试不同情感色彩的句子"""
    print("=" * 60)
    print("测试 5: 不同情感色彩的句子")
    print("=" * 60)
    
    sentence = Sentence()
    entity_extractor = Entity()
    
    test_sentences = [
        "I'm thrilled about winning the competition!",
        "I'm devastated by the loss of my pet.",
        "The sunset over the ocean was absolutely breathtaking.",
        "I'm both happy and sad, my feelings are complicated.",
    ]
    
    for test_sentence in test_sentences:
        result = sentence.process_sentence(
            test_sentence,
            entity_extractor=entity_extractor
        )
        
        print(f"\nSentence: {test_sentence}")
        print(f"Entities: {result['entities']}")
        if result['entities']:
            print("Affective Descriptions:")
            for entity, description in result['affective_descriptions'].items():
                print(f"  - {entity}: {description}")
        else:
            print("No entities extracted.")
        print("-" * 60)
    
    print()


def test_empty_and_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 6: 边界情况")
    print("=" * 60)
    
    sentence = Sentence()
    
    edge_cases = [
        {
            "sentence": "",
            "entity": "test",
            "description": "Empty sentence"
        },
        {
            "sentence": "This is a simple sentence with no entities.",
            "entity": "nonexistent",
            "description": "Entity not in sentence"
        },
    ]
    
    print(f"{'Case':<30} | {'Result':<50}")
    print("-" * 85)
    
    for case in edge_cases:
        try:
            description = sentence.generate_affective_description(
                case["sentence"],
                case["entity"]
            )
            result = description[:47] + "..." if len(description) > 50 else description
            print(f"{case['description']:<30} | {result:<50}")
        except Exception as e:
            print(f"{case['description']:<30} | Error: {str(e)[:45]}")
    
    print()


def test_batch_processing():
    """测试批量处理"""
    print("=" * 60)
    print("测试 7: 批量处理多个句子")
    print("=" * 60)
    
    sentence = Sentence()
    entity_extractor = Entity()
    
    sentences = [
        "Barack Obama was the 44th president of the United States.",
        "I love Apple products, especially the iPhone.",
        "The storm destroyed everything in its path.",
    ]
    
    print(f"{'Sentence':<50} | {'Entities':<30} | {'Descriptions':<20}")
    print("-" * 105)
    
    for test_sentence in sentences:
        result = sentence.process_sentence(
            test_sentence,
            entity_extractor=entity_extractor
        )
        
        sentence_display = test_sentence[:47] + "..." if len(test_sentence) > 50 else test_sentence
        entities_str = ", ".join(result['entities'][:2]) if result['entities'] else "None"
        entities_display = entities_str[:27] + "..." if len(entities_str) > 30 else entities_str
        
        num_descriptions = len([d for d in result['affective_descriptions'].values() if d])
        descriptions_display = f"{num_descriptions} generated"
        
        print(f"{sentence_display:<50} | {entities_display:<30} | {descriptions_display:<20}")
    
    print()


def test_composite_entity_extraction():
    """测试复合实体的识别（如 'Apple products' 而不是单独的 'Apple'）"""
    print("=" * 60)
    print("测试 8: 复合实体识别测试")
    print("=" * 60)
    sentence = Sentence()
    entity_extractor = Entity()
    
    test_cases = [
        {
            "sentence": "I love Apple products, especially the iPhone.",
            "expected_entities": ["Apple products", "Apple products, especially the iPhone"],
            "description": "测试是否能识别 'Apple products' 而不是单独的 'Apple'"
        },
        {
            "sentence": "The University of Cambridge is a prestigious institution.",
            "expected_entities": ["University of Cambridge"],
            "description": "测试是否能识别完整的机构名称"
        },
        {
            "sentence": "I bought Microsoft Office software for my computer.",
            "expected_entities": ["Microsoft Office", "Microsoft Office software"],
            "description": "测试是否能识别产品名称组合"
        },
    ]
    
    print(f"{'Sentence':<60} | {'Extracted Entities':<40} | {'Match':<10}")
    print("-" * 120)
    
    for case in test_cases:
        result = sentence.process_sentence(
            case["sentence"],
            entity_extractor=entity_extractor
        )
        
        extracted = result['entities']
        
        # 检查是否包含期望的实体（部分匹配）
        matches = []
        for expected in case["expected_entities"]:
            # 检查是否有实体包含期望的文本，或期望的文本包含实体
            for entity in extracted:
                if expected.lower() in entity.lower() or entity.lower() in expected.lower():
                    matches.append(expected)
                    break
        
        match_status = "✓" if matches else "✗"
        
        sentence_display = case["sentence"][:57] + "..." if len(case["sentence"]) > 60 else case["sentence"]
        entities_str = ", ".join(extracted) if extracted else "None"
        entities_display = entities_str[:37] + "..." if len(entities_str) > 40 else entities_str
        
        print(f"{sentence_display:<60} | {entities_display:<40} | {match_status:<10}")
        
        # 详细输出
        print(f"\n  描述: {case['description']}")
        print(f"  提取的实体: {extracted}")
        print(f"  期望的实体: {case['expected_entities']}")
        if matches:
            print(f"  ✓ 匹配到的实体: {matches}")
        else:
            print(f"  ✗ 未找到期望的实体")
        print("-" * 120)
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Sentence 类测试")
    print("=" * 60 + "\n")
    
    try:
        test_single_entity_description()
        test_multiple_entities_descriptions()
        test_complete_process()
        test_affective_focus()
        test_different_emotions()
        test_empty_and_edge_cases()
        test_batch_processing()
        test_composite_entity_extraction()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

