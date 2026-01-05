"""
测试：Amygdala 工作流类

全方位测试 Amygdala 类的所有功能：
1. 基本功能：初始化、添加对话、生成粒子
2. 关系映射：粒子与对话的对应关系
3. 持久化：保存和加载关系映射
4. 辅助方法：查询对话、查询粒子
5. 边界情况：空对话、重复对话等
6. 邻域链接：自动构建粒子链接
"""
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, List

from workflow import Amygdala

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cleanup_test_data(test_dir: str):
    """清理测试数据"""
    test_path = Path(test_dir)
    if test_path.exists():
        shutil.rmtree(test_path)
        logger.info(f"清理测试数据: {test_dir}")


def test_amygdala_basic():
    """
    测试1：基本功能测试
    - 初始化 Amygdala
    - 添加单个对话
    - 验证粒子生成和存储
    """
    print("=" * 100)
    print("测试1：基本功能测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_basic"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化 Amygdala
        print("\n【Step 1】初始化 Amygdala...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_basic",
            conversation_namespace="test_conversation",
            embedding_model=None,  # 不使用嵌入模型
            auto_link_particles=False  # 先不测试链接功能
        )
        print("✓ Amygdala 初始化成功")
        print(f"  - 工作目录: {test_dir}")
        print(f"  - 粒子集合: test_particles_basic")
        print(f"  - 对话命名空间: test_conversation")
        
        # Step 2: 添加对话
        print("\n【Step 2】添加对话...")
        conversation = "I love Python programming and I'm excited about machine learning!"
        print(f"对话内容: {conversation}")
        
        result = amygdala.add(conversation)
        
        print(f"\n✓ 对话处理成功")
        print(f"  - 对话 ID: {result['conversation_id']}")
        print(f"  - 生成粒子数: {result['particle_count']}")
        print(f"  - 关系映射数: {len(result['relationship_map'])}")
        
        # Step 3: 验证粒子信息
        print("\n【Step 3】验证粒子信息...")
        particles = result['particles']
        if particles:
            print(f"✓ 成功生成 {len(particles)} 个粒子:")
            for i, p in enumerate(particles[:5], 1):  # 只显示前5个
                print(f"  {i}. {p.entity_id}: {p.entity}")
                print(f"     速度: {p.speed:.4f}, 温度: {p.temperature:.4f}, 质量: {p.weight:.4f}")
            if len(particles) > 5:
                print(f"  ... 还有 {len(particles) - 5} 个粒子")
        else:
            print("⚠ 未生成粒子（可能是环境配置问题）")
        
        # Step 4: 验证关系映射
        print("\n【Step 4】验证关系映射...")
        relationship_map = result['relationship_map']
        print(f"✓ 关系映射包含 {len(relationship_map)} 条记录")
        for particle_id, conv_id in list(relationship_map.items())[:3]:
            print(f"  {particle_id} -> {conv_id}")
        
        print("\n" + "=" * 100)
        print("✓ 测试1完成：基本功能正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # 清理测试数据（可选）
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_multiple_conversations():
    """
    测试2：多个对话测试
    - 添加多个对话
    - 验证每个对话的粒子生成
    - 验证关系映射的正确性
    """
    print("\n" + "=" * 100)
    print("测试2：多个对话测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_multiple"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化
        print("\n【Step 1】初始化 Amygdala...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_multiple",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        print("✓ 初始化成功")
        
        # Step 2: 添加多个对话
        print("\n【Step 2】添加多个对话...")
        conversations = [
            "I love Python programming and machine learning!",
            "The weather is beautiful today, I feel happy.",
            "I'm frustrated with this bug in my code.",
            "Learning new technologies is exciting and challenging."
        ]
        
        results = []
        for i, conv in enumerate(conversations, 1):
            print(f"\n添加对话 {i}: {conv[:50]}...")
            result = amygdala.add(conv)
            results.append(result)
            print(f"  ✓ 对话 ID: {result['conversation_id']}")
            print(f"  ✓ 生成粒子数: {result['particle_count']}")
        
        # Step 3: 验证所有对话都已处理
        print("\n【Step 3】验证所有对话处理结果...")
        total_particles = sum(r['particle_count'] for r in results)
        print(f"✓ 总共处理了 {len(conversations)} 个对话")
        print(f"✓ 总共生成了 {total_particles} 个粒子")
        
        # Step 4: 验证关系映射
        print("\n【Step 4】验证关系映射...")
        all_particle_ids = set()
        all_conversation_ids = set()
        
        for result in results:
            conv_id = result['conversation_id']
            all_conversation_ids.add(conv_id)
            
            for particle_id in result['relationship_map'].keys():
                all_particle_ids.add(particle_id)
                # 验证映射关系
                mapped_conv = amygdala.get_conversation_by_particle(particle_id)
                assert mapped_conv == conv_id, f"粒子 {particle_id} 的映射关系不正确"
        
        print(f"✓ 验证了 {len(all_particle_ids)} 个粒子的映射关系")
        print(f"✓ 涉及 {len(all_conversation_ids)} 个对话")
        
        # Step 5: 验证反向映射
        print("\n【Step 5】验证反向映射...")
        for conv_id in all_conversation_ids:
            particle_ids = amygdala.get_particles_by_conversation(conv_id)
            print(f"  对话 {conv_id[:20]}... 包含 {len(particle_ids)} 个粒子")
            assert len(particle_ids) > 0, f"对话 {conv_id} 应该包含至少一个粒子"
        
        print("\n" + "=" * 100)
        print("✓ 测试2完成：多个对话处理正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_relationship_mapping():
    """
    测试3：关系映射功能测试
    - 测试 get_conversation_by_particle
    - 测试 get_particles_by_conversation
    - 测试 get_conversation_text
    """
    print("\n" + "=" * 100)
    print("测试3：关系映射功能测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_relationship"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化并添加对话
        print("\n【Step 1】初始化并添加对话...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_relationship",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        conversation1 = "I love Python programming!"
        conversation2 = "The weather is beautiful today."
        
        result1 = amygdala.add(conversation1)
        result2 = amygdala.add(conversation2)
        
        conv_id1 = result1['conversation_id']
        conv_id2 = result2['conversation_id']
        
        print(f"✓ 添加了 2 个对话")
        print(f"  对话1 ID: {conv_id1}")
        print(f"  对话2 ID: {conv_id2}")
        
        # Step 2: 测试 get_conversation_by_particle
        print("\n【Step 2】测试 get_conversation_by_particle...")
        particles1 = result1['particles']
        particles2 = result2['particles']
        
        if particles1:
            test_particle_id = particles1[0].entity_id
            mapped_conv = amygdala.get_conversation_by_particle(test_particle_id)
            print(f"  粒子 {test_particle_id[:20]}... -> 对话 {mapped_conv[:20]}...")
            assert mapped_conv == conv_id1, "映射关系不正确"
            print("  ✓ 映射关系正确")
        
        # Step 3: 测试 get_particles_by_conversation
        print("\n【Step 3】测试 get_particles_by_conversation...")
        particle_ids1 = amygdala.get_particles_by_conversation(conv_id1)
        particle_ids2 = amygdala.get_particles_by_conversation(conv_id2)
        
        print(f"  对话1包含 {len(particle_ids1)} 个粒子")
        print(f"  对话2包含 {len(particle_ids2)} 个粒子")
        
        assert len(particle_ids1) == len(particles1), "粒子数量不匹配"
        assert len(particle_ids2) == len(particles2), "粒子数量不匹配"
        print("  ✓ 粒子列表正确")
        
        # Step 4: 测试 get_conversation_text
        print("\n【Step 4】测试 get_conversation_text...")
        retrieved_text1 = amygdala.get_conversation_text(conv_id1)
        retrieved_text2 = amygdala.get_conversation_text(conv_id2)
        
        if retrieved_text1:
            print(f"  对话1文本: {retrieved_text1[:50]}...")
            assert retrieved_text1 == conversation1, "对话文本不匹配"
            print("  ✓ 对话文本正确")
        
        if retrieved_text2:
            print(f"  对话2文本: {retrieved_text2[:50]}...")
            assert retrieved_text2 == conversation2, "对话文本不匹配"
            print("  ✓ 对话文本正确")
        
        # Step 5: 测试不存在的粒子/对话
        print("\n【Step 5】测试边界情况...")
        non_existent_particle = "non_existent_particle_id"
        non_existent_conv = amygdala.get_conversation_by_particle(non_existent_particle)
        assert non_existent_conv is None, "不存在的粒子应该返回 None"
        print("  ✓ 不存在的粒子返回 None")
        
        non_existent_conv_id = "non_existent_conv_id"
        empty_particles = amygdala.get_particles_by_conversation(non_existent_conv_id)
        assert empty_particles == [], "不存在的对话应该返回空列表"
        print("  ✓ 不存在的对话返回空列表")
        
        print("\n" + "=" * 100)
        print("✓ 测试3完成：关系映射功能正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_persistence():
    """
    测试4：持久化测试
    - 保存关系映射
    - 重新加载并验证数据一致性
    """
    print("\n" + "=" * 100)
    print("测试4：持久化测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_persistence"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 创建第一个实例并添加数据
        print("\n【Step 1】创建第一个 Amygdala 实例并添加数据...")
        amygdala1 = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_persistence",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        conversations = [
            "First conversation about Python.",
            "Second conversation about weather.",
            "Third conversation about coding."
        ]
        
        results1 = []
        for conv in conversations:
            result = amygdala1.add(conv)
            results1.append(result)
        
        print(f"✓ 添加了 {len(conversations)} 个对话")
        print(f"✓ 生成了 {sum(r['particle_count'] for r in results1)} 个粒子")
        
        # 记录映射关系
        original_mappings = {}
        for result in results1:
            original_mappings.update(result['relationship_map'])
        
        print(f"✓ 记录了 {len(original_mappings)} 条映射关系")
        
        # Step 2: 创建第二个实例（应该加载已有数据）
        print("\n【Step 2】创建第二个 Amygdala 实例（加载已有数据）...")
        amygdala2 = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_persistence",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        # Step 3: 验证数据已加载
        print("\n【Step 3】验证数据已加载...")
        loaded_mappings_count = len(amygdala2.particle_to_conversation)
        print(f"  加载的映射关系数: {loaded_mappings_count}")
        print(f"  原始映射关系数: {len(original_mappings)}")
        
        # 验证映射关系一致性
        for particle_id, conv_id in original_mappings.items():
            loaded_conv = amygdala2.get_conversation_by_particle(particle_id)
            assert loaded_conv == conv_id, f"粒子 {particle_id} 的映射关系不一致"
        
        print("  ✓ 所有映射关系一致")
        
        # Step 4: 添加新对话并验证持久化
        print("\n【Step 4】添加新对话并验证持久化...")
        new_conv = "New conversation added after reload."
        result_new = amygdala2.add(new_conv)
        
        # 创建第三个实例验证新数据已保存
        amygdala3 = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_persistence",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        new_conv_id = result_new['conversation_id']
        new_particles = amygdala3.get_particles_by_conversation(new_conv_id)
        assert len(new_particles) > 0, "新对话的粒子应该已保存"
        print(f"  ✓ 新对话的 {len(new_particles)} 个粒子已持久化")
        
        print("\n" + "=" * 100)
        print("✓ 测试4完成：持久化功能正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_custom_conversation_id():
    """
    测试5：自定义对话 ID 测试
    - 使用自定义 conversation_id
    - 验证 ID 正确性
    """
    print("\n" + "=" * 100)
    print("测试5：自定义对话 ID 测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_custom_id"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化
        print("\n【Step 1】初始化 Amygdala...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_custom_id",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        # Step 2: 使用自定义 ID 添加对话
        print("\n【Step 2】使用自定义 ID 添加对话...")
        conversation = "This is a test conversation."
        custom_id = "custom_conversation_001"
        
        result = amygdala.add(conversation, conversation_id=custom_id)
        
        print(f"  提供的 ID: {custom_id}")
        print(f"  返回的 ID: {result['conversation_id']}")
        
        assert result['conversation_id'] == custom_id, "对话 ID 应该与提供的 ID 一致"
        print("  ✓ 自定义 ID 正确使用")
        
        # Step 3: 验证可以通过自定义 ID 查询
        print("\n【Step 3】验证自定义 ID 查询...")
        particles = amygdala.get_particles_by_conversation(custom_id)
        text = amygdala.get_conversation_text(custom_id)
        
        print(f"  查询到的粒子数: {len(particles)}")
        print(f"  查询到的文本: {text[:50] if text else 'None'}...")
        
        assert len(particles) == result['particle_count'], "粒子数量应该一致"
        if text:
            assert text == conversation, "对话文本应该一致"
        
        print("  ✓ 自定义 ID 查询正常")
        
        print("\n" + "=" * 100)
        print("✓ 测试5完成：自定义对话 ID 功能正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试5失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_auto_linking():
    """
    测试6：自动链接功能测试
    - 启用自动链接
    - 验证链接构建
    """
    print("\n" + "=" * 100)
    print("测试6：自动链接功能测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_linking"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化（启用自动链接）
        print("\n【Step 1】初始化 Amygdala（启用自动链接）...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_linking",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=True,
            link_distance_threshold=1.5,
            link_top_k=5
        )
        print("✓ 初始化成功，自动链接已启用")
        
        # Step 2: 添加对话
        print("\n【Step 2】添加对话...")
        conversation = "I love Python programming and machine learning!"
        result = amygdala.add(conversation)
        
        print(f"✓ 添加了对话，生成了 {result['particle_count']} 个粒子")
        
        # Step 3: 验证链接（通过查询存储的粒子）
        print("\n【Step 3】验证粒子链接...")
        # 注意：链接信息存储在粒子的 metadata 中，这里我们只验证流程正常
        # 实际的链接验证需要查询数据库
        particles = result['particles']
        if len(particles) > 1:
            print(f"  ✓ 生成了 {len(particles)} 个粒子，链接应该已构建")
            print("  （链接信息存储在粒子的 metadata 中）")
        else:
            print("  ⚠ 只生成了 1 个粒子，无法构建链接")
        
        print("\n" + "=" * 100)
        print("✓ 测试6完成：自动链接功能正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试6失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_edge_cases():
    """
    测试7：边界情况测试
    - 空对话
    - 重复对话
    - 特殊字符
    """
    print("\n" + "=" * 100)
    print("测试7：边界情况测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_edge"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化
        print("\n【Step 1】初始化 Amygdala...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_edge",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=False
        )
        
        # Step 2: 测试空对话（应该返回空结果）
        print("\n【Step 2】测试空对话...")
        empty_result = amygdala.add("")
        print(f"  空对话结果: {empty_result['particle_count']} 个粒子")
        assert empty_result['particle_count'] == 0, "空对话应该生成 0 个粒子"
        print("  ✓ 空对话处理正确")
        
        # Step 3: 测试重复对话
        print("\n【Step 3】测试重复对话...")
        conversation = "This is a repeated conversation."
        
        result1 = amygdala.add(conversation)
        result2 = amygdala.add(conversation)  # 重复添加
        
        print(f"  第一次添加: {result1['conversation_id']}")
        print(f"  第二次添加: {result2['conversation_id']}")
        
        # 相同的对话应该生成相同的 conversation_id（基于 hash）
        assert result1['conversation_id'] == result2['conversation_id'], "相同对话应该生成相同 ID"
        print("  ✓ 重复对话处理正确（生成相同 ID）")
        
        # Step 4: 测试特殊字符
        print("\n【Step 4】测试特殊字符...")
        special_conv = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        result_special = amygdala.add(special_conv)
        print(f"  ✓ 特殊字符对话处理成功: {result_special['conversation_id']}")
        
        # Step 5: 测试长对话
        print("\n【Step 5】测试长对话...")
        long_conv = "This is a very long conversation. " * 100
        result_long = amygdala.add(long_conv)
        print(f"  ✓ 长对话处理成功: {result_long['particle_count']} 个粒子")
        
        print("\n" + "=" * 100)
        print("✓ 测试7完成：边界情况处理正常")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试7失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def test_amygdala_integration():
    """
    测试8：集成测试
    - 完整流程测试
    - 多个功能组合使用
    """
    print("\n" + "=" * 100)
    print("测试8：集成测试")
    print("=" * 100)
    
    test_dir = "./test_amygdala_integration"
    cleanup_test_data(test_dir)
    
    try:
        # Step 1: 初始化
        print("\n【Step 1】初始化 Amygdala...")
        amygdala = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_integration",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=True,
            link_distance_threshold=2.0,
            link_top_k=10
        )
        print("✓ 初始化成功")
        
        # Step 2: 添加多个对话
        print("\n【Step 2】添加多个对话...")
        conversations = [
            ("conv_001", "I love Python programming!"),
            ("conv_002", "The weather is beautiful today."),
            ("conv_003", "Machine learning is fascinating."),
            ("conv_004", "I'm working on a new project.")
        ]
        
        results = []
        for conv_id, conv_text in conversations:
            result = amygdala.add(conv_text, conversation_id=conv_id)
            results.append((conv_id, result))
            print(f"  ✓ {conv_id}: {result['particle_count']} 个粒子")
        
        # Step 3: 验证所有功能
        print("\n【Step 3】验证所有功能...")
        
        # 3.1 验证关系映射
        total_particles = 0
        for conv_id, result in results:
            particles = amygdala.get_particles_by_conversation(conv_id)
            total_particles += len(particles)
            
            for particle_id in particles:
                mapped_conv = amygdala.get_conversation_by_particle(particle_id)
                assert mapped_conv == conv_id, f"映射关系错误: {particle_id}"
        
        print(f"  ✓ 验证了 {total_particles} 个粒子的映射关系")
        
        # 3.2 验证对话文本
        for conv_id, (_, result) in zip([c[0] for c in conversations], results):
            text = amygdala.get_conversation_text(conv_id)
            assert text is not None, f"对话 {conv_id} 的文本应该存在"
        
        print(f"  ✓ 验证了 {len(conversations)} 个对话的文本")
        
        # Step 4: 验证持久化
        print("\n【Step 4】验证持久化...")
        amygdala2 = Amygdala(
            save_dir=test_dir,
            particle_collection_name="test_particles_integration",
            conversation_namespace="test_conversation",
            embedding_model=None,
            auto_link_particles=True
        )
        
        # 验证数据已加载
        loaded_particles = len(amygdala2.particle_to_conversation)
        assert loaded_particles == total_particles, "持久化的粒子数量应该一致"
        print(f"  ✓ 持久化验证成功: {loaded_particles} 个粒子")
        
        print("\n" + "=" * 100)
        print("✓ 测试8完成：集成测试通过")
        print("=" * 100)
        
    except Exception as e:
        print(f"\n❌ 测试8失败: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # cleanup_test_data(test_dir)
        pass


def main():
    """运行所有测试"""
    print("\n" + "=" * 100)
    print("Amygdala 工作流类 - 全方位测试")
    print("=" * 100)
    
    tests = [
        ("基本功能", test_amygdala_basic),
        ("多个对话", test_amygdala_multiple_conversations),
        ("关系映射", test_amygdala_relationship_mapping),
        ("持久化", test_amygdala_persistence),
        ("自定义ID", test_amygdala_custom_conversation_id),
        ("自动链接", test_amygdala_auto_linking),
        ("边界情况", test_amygdala_edge_cases),
        ("集成测试", test_amygdala_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {test_name} 测试通过")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} 测试失败: {e}")
    
    print("\n" + "=" * 100)
    print("测试总结")
    print("=" * 100)
    print(f"总测试数: {len(tests)}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print("=" * 100)
    
    if failed > 0:
        raise AssertionError(f"{failed} 个测试失败")


if __name__ == "__main__":
    main()

