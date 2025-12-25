"""
测试示例：验证 poincare 模块的基本功能

运行方式：
    python poincare/test_example.py
"""
import time
import torch
import logging
from poincare.types import Point
from poincare.physics import ParticleProjector
from poincare.storage import HyperAmyStorage
from poincare.retrieval import HyperAmyRetrieval

# 配置日志
logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 60)
    print("HyperAmy Poincare 模块测试")
    print("=" * 60)
    
    # 1. 初始化组件
    print("\n1. 初始化组件...")
    projector = ParticleProjector(curvature=1.0, scaling_factor=2.0)
    storage = HyperAmyStorage(persist_path="./test_db")
    retrieval = HyperAmyRetrieval(storage, projector)
    print("✓ 组件初始化成功")

    # 2. 插入测试数据
    print("\n2. 插入测试数据...")
    # 模拟场景：三个点，A是中心，B和C是邻居
    # A (愤怒, 强) -> B (愤怒, 弱) -> C (开心)
    vec_anger = torch.tensor([0.9, 0.1, 0.0])
    vec_happy = torch.tensor([0.0, 0.1, 0.9])
    
    p_a = Point(
        id="A", 
        emotion_vector=vec_anger, 
        v=0.9, 
        T=0.5, 
        born=time.time(), 
        links=["B", "C"]
    )
    p_b = Point(
        id="B", 
        emotion_vector=vec_anger, 
        v=0.2, 
        T=0.5, 
        born=time.time()
    )
    p_c = Point(
        id="C", 
        emotion_vector=vec_happy, 
        v=0.5, 
        T=0.5, 
        born=time.time()
    )

    storage.upsert_point(p_a)
    storage.upsert_point(p_b)
    storage.upsert_point(p_c)
    print("✓ 数据插入成功 (A, B, C)")

    # 3. 构造查询点 (愤怒, 强)
    print("\n3. 构造查询点...")
    query = Point(
        id="Q", 
        emotion_vector=vec_anger, 
        v=0.9, 
        T=0.5, 
        born=time.time()
    )
    print("✓ 查询点构造成功")

    # 4. 执行检索
    print("\n4. 执行检索...")
    results = retrieval.search(
        query, 
        top_k=3, 
        cone_width=50,
        neighbor_penalty=1.2
    )
    print(f"✓ 检索完成，找到 {len(results)} 个结果")

    # 5. 输出结果
    print("\n5. 检索结果：")
    print("-" * 60)
    for i, res in enumerate(results, 1):
        print(
            f"{i}. ID: {res.id:3s} | "
            f"Score: {res.score:.4f} | "
            f"Type: {res.match_type:8s} | "
            f"v: {res.metadata['v']:.2f}"
        )
    print("-" * 60)
    
    print("\n✓ 测试完成！")

if __name__ == "__main__":
    main()

