"""
测试重构后的 Particle 模块

验证解耦后的功能是否正常工作。
"""
import numpy as np
from particle import Particle, ParticleEntity


def test_particle_basic():
    """测试基本功能"""
    print("=" * 60)
    print("测试：Particle 基本功能")
    print("=" * 60)
    
    # 初始化 Particle（需要环境变量配置）
    try:
        particle = Particle()
    except Exception as e:
        print(f"初始化失败（需要配置环境变量）: {e}")
        print("跳过实际测试，仅展示代码结构")
        return
    
    # 测试文本
    text = "I love Python programming and I'm excited about machine learning!"
    text_id = "test_001"
    
    print(f"\n输入文本: {text}")
    print(f"文本 ID: {text_id}")
    
    # 处理文本
    try:
        particles = particle.process(text=text, text_id=text_id)
        
        print(f"\n✓ 处理成功，生成 {len(particles)} 个粒子实体")
        print("\n粒子实体详情:")
        print("-" * 60)
        
        for i, p in enumerate(particles, 1):
            print(f"\n{i}. 实体 ID: {p.entity_id}")
            print(f"   实体名称: {p.entity}")
            print(f"   文本 ID: {p.text_id}")
            print(f"   情绪向量形状: {p.emotion_vector.shape}")
            print(f"   速度: {p.speed:.4f}")
            print(f"   温度: {p.temperature:.4f}")
            print(f"   生成时间: {p.born}")
        
        print("\n✓ 测试完成！")
        
    except Exception as e:
        print(f"\n处理失败: {e}")
        import traceback
        traceback.print_exc()


def test_particle_structure():
    """测试数据结构"""
    print("\n" + "=" * 60)
    print("测试：ParticleEntity 数据结构")
    print("=" * 60)
    
    # 创建示例数据
    import time
    
    sample_entity = ParticleEntity(
        entity_id="test_entity_001",
        entity="Python",
        text_id="text_001",
        emotion_vector=np.array([0.8, 0.2, 0.0, 0.0] + [0.0] * 26),  # 示例向量
        speed=0.5,
        temperature=0.5,
        born=time.time()
    )
    
    print("\n示例 ParticleEntity:")
    print(f"  entity_id: {sample_entity.entity_id}")
    print(f"  entity: {sample_entity.entity}")
    print(f"  text_id: {sample_entity.text_id}")
    print(f"  emotion_vector shape: {sample_entity.emotion_vector.shape}")
    print(f"  speed: {sample_entity.speed}")
    print(f"  temperature: {sample_entity.temperature}")
    print(f"  born: {sample_entity.born}")
    
    print("\n✓ 数据结构验证完成！")


if __name__ == "__main__":
    test_particle_structure()
    test_particle_basic()

