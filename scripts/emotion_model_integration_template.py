#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情绪嵌入模型集成模板

提供情绪嵌入模型的接口规范和示例实现，供合作者参考。
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionEmbeddingModelInterface:
    """
    情绪嵌入模型接口规范
    
    这是接口规范，不是具体实现。合作者需要实现具体的模型类。
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化模型
        
        Args:
            model_path: 模型文件路径
            device: 设备（"cpu" 或 "cuda"）
        
        Raises:
            FileNotFoundError: 模型文件不存在
            ValueError: 模型文件格式错误
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def extract(self, text: str) -> np.ndarray:
        """
        提取单个文本的情绪向量
        
        Args:
            text: 输入文本（字符串）
            
        Returns:
            28维情绪向量（numpy array，dtype=float32）
            每个维度在[0, 1]范围内（不强制归一化）
            
        Raises:
            ValueError: 输入文本为空或无效
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def extract_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量提取情绪向量
        
        Args:
            texts: 文本列表
            
        Returns:
            情绪向量列表，每个向量为28维numpy array
            
        Raises:
            ValueError: 输入列表为空或包含无效文本
        """
        raise NotImplementedError("子类必须实现此方法")
    
    def extract_with_intensity(self, text: str) -> Dict[str, Any]:
        """
        提取情绪向量和强度
        
        Args:
            text: 输入文本
            
        Returns:
            {
                "emotion_vector": np.ndarray,  # 28维向量
                "intensity": float  # L2-norm of vector
            }
        """
        vector = self.extract(text)
        intensity = float(np.linalg.norm(vector))
        return {
            "emotion_vector": vector,
            "intensity": intensity
        }


class EmotionEmbeddingModelExample(EmotionEmbeddingModelInterface):
    """
    情绪嵌入模型示例实现
    
    这是一个示例实现，展示如何实现接口。
    合作者需要根据实际模型框架（PyTorch/TensorFlow/ONNX等）进行修改。
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化模型
        
        示例：使用PyTorch加载模型
        """
        import torch
        from pathlib import Path
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 示例：加载PyTorch模型
        # self.model = torch.load(model_path, map_location=device)
        # self.model.eval()
        # self.device = device
        
        # 示例：加载ONNX模型
        # import onnxruntime as ort
        # self.session = ort.InferenceSession(model_path)
        
        # 这里只是示例，实际实现需要根据模型格式调整
        logger.info(f"模型加载示例（需要根据实际模型实现）: {model_path}")
        self.model_path = model_path
        self.device = device
    
    def extract(self, text: str) -> np.ndarray:
        """
        提取单个文本的情绪向量
        
        示例实现（需要根据实际模型修改）
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")
        
        # 示例：使用模型进行推理
        # 1. 文本预处理（tokenization等）
        # processed_text = self._preprocess(text)
        
        # 2. 模型推理
        # with torch.no_grad():
        #     output = self.model(processed_text)
        #     vector = output.cpu().numpy().flatten()
        
        # 3. 后处理（确保维度为28，值在[0,1]范围）
        # vector = self._postprocess(vector)
        
        # 这里返回一个示例向量（实际应使用模型输出）
        vector = np.random.rand(28).astype(np.float32)
        vector = np.clip(vector, 0.0, 1.0)  # 确保值在[0,1]范围
        
        # 验证维度
        if len(vector) != 28:
            raise ValueError(f"输出向量维度错误: {len(vector)} != 28")
        
        return vector
    
    def extract_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        批量提取情绪向量
        
        示例实现（需要根据实际模型修改）
        """
        if not texts:
            raise ValueError("输入文本列表不能为空")
        
        # 示例：批量处理
        # processed_texts = [self._preprocess(text) for text in texts]
        # with torch.no_grad():
        #     outputs = self.model(processed_texts)
        #     vectors = [self._postprocess(output.cpu().numpy().flatten()) 
        #                for output in outputs]
        
        # 这里返回示例向量（实际应使用模型输出）
        vectors = []
        for text in texts:
            vector = self.extract(text)  # 可以优化为批量处理
            vectors.append(vector)
        
        return vectors
    
    def _preprocess(self, text: str) -> Any:
        """
        文本预处理（示例）
        
        根据实际模型需求进行tokenization等预处理
        """
        # 示例：简单的预处理
        return text.strip().lower()
    
    def _postprocess(self, vector: np.ndarray) -> np.ndarray:
        """
        后处理（示例）
        
        确保输出符合要求：
        - 维度为28
        - 值在[0,1]范围
        """
        # 确保维度
        if len(vector) != 28:
            if len(vector) > 28:
                vector = vector[:28]
            else:
                vector = np.pad(vector, (0, 28 - len(vector)), 'constant')
        
        # 确保值在[0,1]范围
        vector = np.clip(vector, 0.0, 1.0)
        
        return vector.astype(np.float32)


def test_model_interface(model: EmotionEmbeddingModelInterface):
    """
    测试模型接口
    
    Args:
        model: 实现了EmotionEmbeddingModelInterface的模型实例
    """
    print("=" * 80)
    print("测试情绪嵌入模型接口")
    print("=" * 80)
    
    # 测试1: 单文本提取
    print("\n测试1: 单文本提取")
    text = "I am very happy today!"
    try:
        vector = model.extract(text)
        print(f"✅ 提取成功")
        print(f"   维度: {vector.shape}")
        print(f"   数据类型: {vector.dtype}")
        print(f"   值范围: [{vector.min():.4f}, {vector.max():.4f}]")
        print(f"   强度 (L2-norm): {np.linalg.norm(vector):.4f}")
        
        # 验证
        assert vector.shape == (28,), f"维度错误: {vector.shape} != (28,)"
        assert vector.dtype == np.float32, f"数据类型错误: {vector.dtype} != float32"
        assert np.all(vector >= 0) and np.all(vector <= 1), "值超出[0,1]范围"
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试2: 批量提取
    print("\n测试2: 批量提取")
    texts = ["I am happy", "I am sad", "I am angry"]
    try:
        vectors = model.extract_batch(texts)
        print(f"✅ 批量提取成功")
        print(f"   文本数: {len(texts)}")
        print(f"   向量数: {len(vectors)}")
        print(f"   所有向量维度正确: {all(v.shape == (28,) for v in vectors)}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试3: 提取带强度
    print("\n测试3: 提取带强度")
    try:
        result = model.extract_with_intensity(text)
        print(f"✅ 提取成功")
        print(f"   包含emotion_vector: {'emotion_vector' in result}")
        print(f"   包含intensity: {'intensity' in result}")
        print(f"   intensity值: {result['intensity']:.4f}")
        print(f"   intensity = L2-norm: {abs(result['intensity'] - np.linalg.norm(result['emotion_vector'])) < 0.001}")
        
        # 验证
        assert "emotion_vector" in result
        assert "intensity" in result
        assert abs(result["intensity"] - np.linalg.norm(result["emotion_vector"])) < 0.001
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ 所有测试通过！")
    print("=" * 80)
    return True


def main():
    """主函数：演示如何使用接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="情绪嵌入模型集成模板测试")
    parser.add_argument(
        "--model-path",
        type=str,
        help="模型文件路径（可选，用于实际测试）"
    )
    parser.add_argument(
        "--test-example",
        action="store_true",
        help="测试示例实现"
    )
    
    args = parser.parse_args()
    
    if args.test_example:
        # 测试示例实现
        print("测试示例实现（使用随机向量）...")
        model = EmotionEmbeddingModelExample("dummy_path")
        test_model_interface(model)
    elif args.model_path:
        # 测试实际模型（需要合作者实现）
        print(f"测试实际模型: {args.model_path}")
        print("注意：需要实现具体的模型类")
        # model = YourEmotionModel(args.model_path)
        # test_model_interface(model)
    else:
        print("情绪嵌入模型集成模板")
        print("\n使用方法：")
        print("1. 实现 EmotionEmbeddingModelInterface 接口")
        print("2. 参考 EmotionEmbeddingModelExample 示例")
        print("3. 运行测试: python emotion_model_integration_template.py --test-example")
        print("\n详细文档请参考: docs/EMOTION_MODEL_INTEGRATION.md")


if __name__ == "__main__":
    main()
