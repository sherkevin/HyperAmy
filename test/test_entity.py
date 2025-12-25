"""
测试 Entity 类

测试实体抽取功能，验证抽取到的实体是否合理
"""
from utils.entitiy import Entity


def test_basic_entity_extraction():
    """测试基本实体抽取"""
    print("=" * 60)
    print("测试 1: 基本实体抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunk = "Barack Obama was the 44th president of the United States."
    entities = entity.extract_entities(chunk)
    
    print(f"Chunk: {chunk}")
    print(f"Extracted Entities: {entities}")
    print(f"Number of Entities: {len(entities)}")
    
    # 验证合理性：应该包含人名和地名
    expected_keywords = ["Obama", "United States"]
    found_keywords = [kw for kw in expected_keywords if any(kw.lower() in ent.lower() for ent in entities)]
    print(f"Expected Keywords Found: {found_keywords}")
    print()


def test_person_entities():
    """测试人名实体抽取"""
    print("=" * 60)
    print("测试 2: 人名实体抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunks = [
        "Albert Einstein developed the theory of relativity.",
        "Marie Curie was a pioneering scientist.",
        "Steve Jobs co-founded Apple Inc.",
    ]
    
    print(f"{'Chunk':<60} | {'Entities':<40}")
    print("-" * 105)
    
    for chunk in chunks:
        entities = entity.extract_entities(chunk)
        chunk_display = chunk[:57] + "..." if len(chunk) > 60 else chunk
        entities_str = ", ".join(entities) if entities else "None"
        entities_display = entities_str[:37] + "..." if len(entities_str) > 40 else entities_str
        print(f"{chunk_display:<60} | {entities_display:<40}")
        
        # 验证合理性：应该包含人名
        has_person = any(len(ent.split()) >= 2 or any(word[0].isupper() for word in ent.split()) for ent in entities)
        print(f"  -> Contains person name: {'✓' if has_person else '✗'}")
    
    print()


def test_location_entities():
    """测试地名实体抽取"""
    print("=" * 60)
    print("测试 3: 地名实体抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunks = [
        "Paris is the capital of France.",
        "The Great Wall of China is a famous landmark.",
        "New York City is located in the United States.",
    ]
    
    print(f"{'Chunk':<60} | {'Entities':<40}")
    print("-" * 105)
    
    for chunk in chunks:
        entities = entity.extract_entities(chunk)
        chunk_display = chunk[:57] + "..." if len(chunk) > 60 else chunk
        entities_str = ", ".join(entities) if entities else "None"
        entities_display = entities_str[:37] + "..." if len(entities_str) > 40 else entities_str
        print(f"{chunk_display:<60} | {entities_display:<40}")
        
        # 验证合理性：应该包含地名
        has_location = any(ent in chunk for ent in entities)
        print(f"  -> Contains location: {'✓' if has_location else '✗'}")
    
    print()


def test_organization_entities():
    """测试组织实体抽取"""
    print("=" * 60)
    print("测试 4: 组织实体抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunks = [
        "Apple Inc. was founded by Steve Jobs.",
        "Microsoft Corporation is a technology company.",
        "The United Nations was established in 1945.",
    ]
    
    print(f"{'Chunk':<60} | {'Entities':<40}")
    print("-" * 105)
    
    for chunk in chunks:
        entities = entity.extract_entities(chunk)
        chunk_display = chunk[:57] + "..." if len(chunk) > 60 else chunk
        entities_str = ", ".join(entities) if entities else "None"
        entities_display = entities_str[:37] + "..." if len(entities_str) > 40 else entities_str
        print(f"{chunk_display:<60} | {entities_display:<40}")
        
        # 验证合理性：应该包含组织名
        has_org = any(any(keyword in ent for keyword in ["Inc", "Corporation", "United"]) for ent in entities)
        print(f"  -> Contains organization: {'✓' if has_org else '✗'}")
    
    print()


def test_triple_extraction():
    """测试三元组抽取"""
    print("=" * 60)
    print("测试 5: 三元组抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunk = "Barack Obama was the 44th president of the United States."
    triples = entity.extract_triples(chunk)
    
    print(f"Chunk: {chunk}")
    print(f"Number of Triples: {len(triples)}")
    print()
    print("Extracted Triples:")
    print("-" * 60)
    for i, triple in enumerate(triples, 1):
        if len(triple) == 3:
            print(f"{i}. Subject: {triple[0]}")
            print(f"   Relation: {triple[1]}")
            print(f"   Object: {triple[2]}")
        else:
            print(f"{i}. Invalid triple format: {triple}")
        print()
    
    # 验证合理性：三元组应该包含有意义的关系
    valid_triples = [t for t in triples if len(t) == 3]
    print(f"Valid Triples: {len(valid_triples)}/{len(triples)}")
    print()


def test_extract_all():
    """测试同时抽取实体和三元组"""
    print("=" * 60)
    print("测试 6: 同时抽取实体和三元组")
    print("=" * 60)
    
    entity = Entity()
    
    chunk = "Albert Einstein developed the theory of relativity in 1905."
    result = entity.extract_all(chunk)
    
    print(f"Chunk: {chunk}")
    print()
    print(f"Entities ({len(result['entities'])}):")
    for i, ent in enumerate(result['entities'], 1):
        print(f"  {i}. {ent}")
    print()
    print(f"Triples ({len(result['triples'])}):")
    for i, triple in enumerate(result['triples'], 1):
        if len(triple) == 3:
            print(f"  {i}. [{triple[0]}, {triple[1]}, {triple[2]}]")
        else:
            print(f"  {i}. Invalid: {triple}")
    print()


def test_complex_sentences():
    """测试复杂句子"""
    print("=" * 60)
    print("测试 7: 复杂句子实体抽取")
    print("=" * 60)
    
    entity = Entity()
    
    chunks = [
        "The iPhone was designed by Apple Inc. and first released in 2007.",
        "The University of Cambridge, located in Cambridge, England, was founded in 1209.",
        "NASA's Mars rover Perseverance landed on Mars in February 2021.",
    ]
    
    print(f"{'Chunk':<70} | {'Entities':<50}")
    print("-" * 125)
    
    for chunk in chunks:
        entities = entity.extract_entities(chunk)
        chunk_display = chunk[:67] + "..." if len(chunk) > 70 else chunk
        entities_str = ", ".join(entities) if entities else "None"
        entities_display = entities_str[:47] + "..." if len(entities_str) > 50 else entities_str
        print(f"{chunk_display:<70} | {entities_display:<50}")
    
    print()


def test_empty_and_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试 8: 边界情况")
    print("=" * 60)
    
    entity = Entity()
    
    edge_chunks = [
        "",  # 空字符串
        "This is a simple sentence with no entities.",  # 无实体
        "a",  # 单个字符
    ]
    
    print(f"{'Chunk':<60} | {'Entities':<40}")
    print("-" * 105)
    
    for chunk in edge_chunks:
        try:
            entities = entity.extract_entities(chunk)
            chunk_display = chunk[:57] + "..." if len(chunk) > 60 else chunk
            if len(chunk) == 0:
                chunk_display = "(empty)"
            entities_str = ", ".join(entities) if entities else "None"
            entities_display = entities_str[:37] + "..." if len(entities_str) > 40 else entities_str
            print(f"{chunk_display:<60} | {entities_display:<40}")
        except Exception as e:
            chunk_display = chunk[:57] + "..." if len(chunk) > 60 else chunk
            if len(chunk) == 0:
                chunk_display = "(empty)"
            print(f"{chunk_display:<60} | Error: {str(e)[:30]}")
    
    print()


def test_entity_quality_assessment():
    """测试实体质量评估"""
    print("=" * 60)
    print("测试 9: 实体质量评估")
    print("=" * 60)
    
    entity = Entity()
    
    # 测试用例：包含明确的实体
    test_cases = [
        {
            "chunk": "Barack Obama was born in Hawaii.",
            "expected_entities": ["Barack Obama", "Hawaii"],
            "description": "人名和地名"
        },
        {
            "chunk": "Apple Inc. was founded in 1976.",
            "expected_entities": ["Apple"],
            "description": "公司名和年份"
        },
        {
            "chunk": "The Eiffel Tower is located in Paris, France.",
            "expected_entities": ["Eiffel Tower", "Paris", "France"],
            "description": "地标和地名"
        },
    ]
    
    print(f"{'Description':<20} | {'Chunk':<50} | {'Extracted':<30} | {'Quality':<10}")
    print("-" * 115)
    
    for case in test_cases:
        entities = entity.extract_entities(case["chunk"])
        
        # 计算质量分数：匹配的预期实体数量 / 总预期实体数量
        matched = sum(1 for exp in case["expected_entities"] 
                     if any(exp.lower() in ent.lower() or ent.lower() in exp.lower() 
                            for ent in entities))
        quality_score = matched / len(case["expected_entities"]) if case["expected_entities"] else 0
        
        chunk_display = case["chunk"][:47] + "..." if len(case["chunk"]) > 50 else case["chunk"]
        entities_str = ", ".join(entities[:3]) if len(entities) > 3 else ", ".join(entities)
        entities_display = entities_str[:27] + "..." if len(entities_str) > 30 else entities_str
        
        quality_str = f"{quality_score:.2f}" if quality_score > 0 else "0.00"
        
        print(f"{case['description']:<20} | {chunk_display:<50} | {entities_display:<30} | {quality_str:<10}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Entity 类测试")
    print("=" * 60 + "\n")
    
    try:
        test_basic_entity_extraction()
        test_person_entities()
        test_location_entities()
        test_organization_entities()
        test_triple_extraction()
        test_extract_all()
        test_complex_sentences()
        test_empty_and_edge_cases()
        test_entity_quality_assessment()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

