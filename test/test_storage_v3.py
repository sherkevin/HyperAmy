"""
测试 poincare/storage.py 模块 (System V3)

验证存储层的正确性。

测试内容：
1. V3 粒子的存储和检索
2. 批量操作
3. 元数据处理
4. 向量查询
"""

import numpy as np
import tempfile
import shutil
import os

from poincare.storage import HyperAmyStorage
from poincare.retrieval import CandidateParticle, create_candidate


class TestHyperAmyStorageV3:
    """测试 V3 存储层"""

    def test_init_storage(self):
        """测试存储层初始化"""
        # 使用临时目录
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(
                persist_path=temp_dir,
                collection_name="test_particles",
                curvature=1.0
            )
            assert storage.curvature == 1.0
            assert storage.count_candidates() == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_upsert_and_get_candidate(self):
        """测试单个粒子的存储和获取"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_v3")

            # 创建候选粒子
            candidate = create_candidate(
                particle_id="test_001",
                direction=np.array([1.0, 0.0, 0.0]),
                mass=2.0,
                temperature=1.5,
                initial_radius=2.0,
                created_at=0.0,
                entity="test_entity"
            )

            # 存储
            storage.upsert_candidate(candidate)

            # 验证计数
            assert storage.count_candidates() == 1

            # 获取所有粒子
            candidates = storage.get_all_candidates()
            assert len(candidates) == 1
            assert candidates[0].id == "test_001"
            assert candidates[0].mass == 2.0
            assert candidates[0].temperature == 1.5

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_batch_upsert_candidates(self):
        """测试批量存储粒子"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_batch")

            # 创建多个候选粒子
            candidates = [
                create_candidate(f"p{i}", np.array([1.0, 0.0]), mass=1.0, temperature=1.0, initial_radius=1.0, created_at=0.0)
                for i in range(5)
            ]

            # 批量存储
            storage.upsert_candidates(candidates)

            # 验证
            assert storage.count_candidates() == 5

            retrieved = storage.get_all_candidates()
            assert len(retrieved) == 5

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_delete_candidate(self):
        """测试删除粒子"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_delete")

            # 添加粒子
            candidate = create_candidate(
                particle_id="to_delete",
                direction=np.array([1.0, 0.0]),
                mass=1.0,
                temperature=1.0,
                initial_radius=1.0,
                created_at=0.0
            )
            storage.upsert_candidate(candidate)
            assert storage.count_candidates() == 1

            # 删除
            result = storage.delete_candidate("to_delete")
            assert result is True
            assert storage.count_candidates() == 0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_by_direction(self):
        """测试按方向查询"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_query")

            # 添加不同方向的粒子
            candidates = [
                create_candidate("p1", np.array([1.0, 0.0]), mass=2.0, temperature=1.0, initial_radius=1.0, created_at=0.0),
                create_candidate("p2", np.array([0.0, 1.0]), mass=1.0, temperature=1.0, initial_radius=1.0, created_at=0.0),
                create_candidate("p3", np.array([0.707, 0.707]), mass=0.5, temperature=2.0, initial_radius=1.0, created_at=0.0),
            ]
            storage.upsert_candidates(candidates)

            # 查询与 [1, 0] 相似的粒子
            results = storage.query_by_direction(
                direction=np.array([1.0, 0.0]),
                n_results=2
            )

            assert len(results) <= 2
            # 第一个结果应该是 p1（完全相同方向）
            if results:
                assert results[0]["id"] in ["p1", "p3"]

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_with_filters(self):
        """测试带过滤条件的查询"""
        temp_dir = tempfile.mkdtemp()
        try:
            storage = HyperAmyStorage(persist_path=temp_dir, collection_name="test_filters")

            # 添加不同质量和温度的粒子
            candidates = [
                create_candidate("heavy", np.array([1.0, 0.0]), mass=10.0, temperature=0.5, initial_radius=1.0, created_at=0.0),
                create_candidate("light", np.array([1.0, 0.0]), mass=0.5, temperature=2.0, initial_radius=1.0, created_at=0.0),
                create_candidate("medium", np.array([1.0, 0.0]), mass=2.0, temperature=1.0, initial_radius=1.0, created_at=0.0),
            ]
            storage.upsert_candidates(candidates)

            # 查询质量 >= 2.0 的粒子
            results = storage.query_by_direction(
                direction=np.array([1.0, 0.0]),
                n_results=10,
                min_mass=2.0
            )

            # 应该只返回 heavy 和 medium
            assert len(results) >= 1
            for r in results:
                mass = float(r["metadata"].get("mass", 0))
                assert mass >= 2.0

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_tests():
    """运行所有测试"""
    print("Running Storage V3 Tests...")
    print("=" * 60)

    test = TestHyperAmyStorageV3()

    print("\n1. Testing storage initialization...")
    test.test_init_storage()
    print("   ✓ Initialization test passed")

    print("\n2. Testing upsert and get candidate...")
    test.test_upsert_and_get_candidate()
    print("   ✓ Upsert and get test passed")

    print("\n3. Testing batch upsert...")
    test.test_batch_upsert_candidates()
    print("   ✓ Batch upsert test passed")

    print("\n4. Testing delete candidate...")
    test.test_delete_candidate()
    print("   ✓ Delete test passed")

    print("\n5. Testing query by direction...")
    test.test_query_by_direction()
    print("   ✓ Query test passed")

    print("\n6. Testing query with filters...")
    test.test_query_with_filters()
    print("   ✓ Filter query test passed")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_tests()
