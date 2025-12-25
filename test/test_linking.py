"""
测试链接构建功能

验证基于双曲距离的邻域构建是否正常工作。
"""
import time
import torch
import logging
from poincare.types import Point
from poincare.physics import ParticleProjector
from poincare.linking import build_hyperbolic_links, auto_link_points

# 配置日志
logging.basicConfig(level=logging.INFO)

def test_hyperbolic_linking():
    """测试基于双曲距离的链接构建"""
    print("=" * 60)
    print("测试：基于双曲距离的邻域构建")
    print("=" * 60)
    
    # 1. 初始化投影器
    projector = ParticleProjector(curvature=1.0, scaling_factor=2.0)
    
    # 2. 创建测试点
    # A: 愤怒, 强 (v=0.9)
    # B: 愤怒, 弱 (v=0.2) - 应该与 A 有链接（相同向量，不同强度）
    # C: 开心, 中 (v=0.5) - 可能没有链接（不同向量）
    # D: 愤怒, 中 (v=0.6) - 应该与 A、B 有链接
    vec_anger = torch.tensor([0.9, 0.1, 0.0])
    vec_happy = torch.tensor([0.0, 0.1, 0.9])
    
    points = [
        Point(id="A", emotion_vector=vec_anger, v=0.9, T=0.5, born=time.time()),
        Point(id="B", emotion_vector=vec_anger, v=0.2, T=0.5, born=time.time()),
        Point(id="C", emotion_vector=vec_happy, v=0.5, T=0.5, born=time.time()),
        Point(id="D", emotion_vector=vec_anger, v=0.6, T=0.5, born=time.time()),
    ]
    
    print(f"\n创建了 {len(points)} 个测试点:")
    for p in points:
        print(f"  - {p.id}: v={p.v:.2f}, vector={'愤怒' if p.id != 'C' else '开心'}")
    
    # 3. 构建链接（使用默认阈值 1.5）
    print("\n使用默认阈值 (1.5) 构建链接...")
    edges = build_hyperbolic_links(
        points=points,
        projector=projector,
        distance_threshold=1.5
    )
    
    print("\n构建的链接关系:")
    for point_id, neighbors in edges.items():
        if neighbors:
            print(f"  {point_id} -> {neighbors}")
        else:
            print(f"  {point_id} -> [] (无邻居)")
    
    # 4. 使用 auto_link_points 自动更新
    print("\n使用 auto_link_points 自动更新链接...")
    updated_points = auto_link_points(
        points=points,
        projector=projector,
        distance_threshold=1.5
    )
    
    print("\n更新后的 Point 对象:")
    for p in updated_points:
        print(f"  {p.id}: links={p.links}")
    
    # 5. 测试不同的阈值
    print("\n" + "=" * 60)
    print("测试不同阈值的影响:")
    print("=" * 60)
    
    for threshold in [0.5, 1.0, 1.5, 2.0]:
        edges = build_hyperbolic_links(
            points=points,
            projector=projector,
            distance_threshold=threshold
        )
        total_links = sum(len(n) for n in edges.values())
        print(f"阈值 {threshold:.1f}: 共 {total_links} 条链接")
        for point_id, neighbors in edges.items():
            if neighbors:
                print(f"  {point_id} -> {len(neighbors)} 个邻居")
    
    print("\n✓ 测试完成！")

if __name__ == "__main__":
    test_hyperbolic_linking()

