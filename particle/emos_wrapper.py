"""
Emos模型包装类

用于在HyperAmy项目中使用emos模型进行实体级情绪向量提取。
支持批量处理，直接在一个句子中提取多个实体的情绪向量（64维hidden_state）。
"""
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)

# 尝试导入emos模型
emos_path = os.environ.get('EMOS_PATH', '')
if emos_path:
    sys.path.insert(0, emos_path)
else:
    # 尝试从项目目录查找
    project_root = Path(__file__).parent.parent
    for emos_dir in ["emos", "emos-master"]:
        emos_dir_path = project_root / emos_dir
        if emos_dir_path.exists():
            sys.path.insert(0, str(emos_dir_path))
            break

try:
    from src.model import GbertPredictor, ProbabilisticGBERTV4
    EMOS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入emos模型: {e}")
    EMOS_AVAILABLE = False
    GbertPredictor = None


class EmosWrapper:
    """
    Emos模型包装类
    
    功能：
    - 加载训练好的emos模型
    - 支持在一个句子中提取多个实体的情绪向量（64维hidden_state）
    - 不需要改写句子，直接使用原始句子+实体span
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        model_name: Optional[str] = None,  # 如果为None，从checkpoint自动读取
        use_8bit_quantization: bool = False  # 8-bit量化（用于8B模型节省显存）
    ):
        """
        初始化EmosWrapper
        
        Args:
            checkpoint_path: 模型checkpoint路径，如果为None则尝试默认路径
            device: 设备 (cpu/cuda)
            model_name: 基座模型名称（如果为None，从checkpoint自动读取）
            use_8bit_quantization: 是否使用8-bit量化（推荐用于8B模型）
        """
        if not EMOS_AVAILABLE:
            raise ImportError("emos模型不可用，请检查导入路径")
        
        # 确定checkpoint路径
        if checkpoint_path is None:
            project_root = Path(__file__).parent.parent
            default_paths = [
                # Qwen3-8B模型（优先）
                Path("/public/jiangh/emos/checkpoints/qwen3_8b/last_checkpoint.pt"),  # 服务器路径
                Path("/public/jiangh/emos/checkpoints/qwen3_8b/best_model.pt"),  # 服务器路径（如果存在）
                # Qwen-7B模型（备选）
                project_root / "outputs" / "stage2_training_remote" / "checkpoints" / "best_model_stage2.pt",
                project_root / "outputs" / "stage2_training" / "checkpoints" / "best_model_stage2.pt",
                Path.home() / "Desktop" / "best_model.pt",
            ]
            for path in default_paths:
                if path.exists():
                    checkpoint_path = str(path)
                    logger.info(f"使用默认模型路径: {checkpoint_path}")
                    break
            else:
                raise FileNotFoundError(f"未找到模型checkpoint，尝试过的路径: {default_paths}")
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型checkpoint不存在: {checkpoint_path}")
        
        # 如果是8B模型且使用GPU，默认启用8-bit量化以节省显存
        if model_name and ("8b" in model_name.lower() or "embedding-8b" in model_name.lower()):
            if device.startswith("cuda") and not use_8bit_quantization:
                logger.info("检测到8B模型，推荐使用8-bit量化以节省显存。如需要，请设置use_8bit_quantization=True")
        
        # 加载模型
        logger.info(f"加载emos模型: {checkpoint_path}")
        logger.info(f"  设备: {device}")
        logger.info(f"  8-bit量化: {use_8bit_quantization}")
        try:
            # 传递use_8bit_quantization参数到from_checkpoint
            # 如果为None，from_checkpoint会自动检测（8B模型+GPU时默认启用）
            self.predictor = GbertPredictor.from_checkpoint(
                checkpoint_path=str(checkpoint_path),
                model_name=model_name,  # 如果为None，从checkpoint自动读取
                device=device,
                load_in_8bit=use_8bit_quantization if use_8bit_quantization else None  # None=auto-detect
            )
            self.device = device
            logger.info(f"✅ emos模型加载成功 (device: {device})")
            
            # 检查模型的实际embedding维度
            # 通过运行一次预测来获取向量维度
            try:
                test_result = self.predictor.predict("test", span_text="test")
                if 'mu' in test_result:
                    embedding_dim = len(test_result['mu'])
                    logger.info(f"✅ 模型embedding维度: {embedding_dim}")
            except:
                logger.warning("无法自动检测embedding维度，默认使用256维")
                
        except Exception as e:
            logger.error(f"加载emos模型失败: {e}")
            raise
    
    def extract_entity_emotion_vector(
        self,
        text: str,
        entity_text: str,
        entity_start: Optional[int] = None,
        entity_end: Optional[int] = None
    ) -> np.ndarray:
        """
        提取单个实体在句子中的情绪向量（64维hidden_state）
        
        Args:
            text: 完整句子文本
            entity_text: 实体文本span
            entity_start: 实体在文本中的字符起始位置（可选，用于精确定位）
            entity_end: 实体在文本中的字符结束位置（可选，用于精确定位）
        
        Returns:
            np.ndarray: 64维情绪向量
        """
        try:
            # 使用emos模型的predict方法
            # 如果提供了entity_text，模型会自动定位实体并提取情绪向量
            result = self.predictor.predict(text, span_text=entity_text)
            
            # 获取mu（64维embedding向量）
            mu = np.array(result['mu'])
            
            # 支持动态维度（64, 128, 256等）
            expected_dims = [64, 128, 256, 512]
            if len(mu) not in expected_dims:
                logger.warning(f"情绪向量维度异常: {len(mu)}，预期为 {expected_dims}")
            
            return mu
            
        except Exception as e:
            logger.error(f"提取实体情绪向量失败: text='{text[:50]}...', entity='{entity_text}', error={e}")
            # 返回零向量作为fallback
            # 尝试从predictor获取embedding维度
            try:
                # 通过检查模型的semantic_head输出维度来确定
                if hasattr(self.predictor, 'model') and hasattr(self.predictor.model, 'semantic_head'):
                    embedding_dim = self.predictor.model.semantic_head.out_features
                else:
                    embedding_dim = 256  # 默认256维（Qwen3-8B和Qwen-7B都是256维）
            except:
                embedding_dim = 256
            logger.warning(f"提取失败，返回零向量（维度: {embedding_dim}）")
            return np.zeros(embedding_dim, dtype=np.float32)
    
    def extract_entities_emotion_vectors(
        self,
        text: str,
        entities: List[str],
        entity_positions: Optional[List[Dict[str, int]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        批量提取多个实体在句子中的情绪向量
        
        Args:
            text: 完整句子文本
            entities: 实体文本列表
            entity_positions: 可选的实体位置信息列表，每个元素包含{'char_start': int, 'char_end': int}
        
        Returns:
            Dict[str, np.ndarray]: 实体到情绪向量的映射 {entity: 64维向量}
        """
        results = {}
        
        for i, entity in enumerate(entities):
            try:
                # 如果提供了位置信息，使用更精确的定位
                if entity_positions and i < len(entity_positions):
                    pos = entity_positions[i]
                    # 注意：emos模型的predict方法只支持span_text，不支持直接传入char_start/char_end
                    # 所以这里还是使用entity_text进行定位
                    vector = self.extract_entity_emotion_vector(
                        text=text,
                        entity_text=entity,
                        entity_start=pos.get('char_start'),
                        entity_end=pos.get('char_end')
                    )
                else:
                    vector = self.extract_entity_emotion_vector(
                        text=text,
                        entity_text=entity
                    )
                
                results[entity] = vector
                
            except Exception as e:
                logger.error(f"提取实体 '{entity}' 的情绪向量失败: {e}")
                # 获取正确的embedding维度
                try:
                    if hasattr(self.predictor, 'model') and hasattr(self.predictor.model, 'semantic_head'):
                        embedding_dim = self.predictor.model.semantic_head.out_features
                    else:
                        embedding_dim = 256  # 默认256维
                except:
                    embedding_dim = 256
                results[entity] = np.zeros(embedding_dim, dtype=np.float32)
        
        logger.debug(f"成功提取 {len([v for v in results.values() if v.sum() != 0])}/{len(entities)} 个实体的情绪向量")
        return results
    
    def get_token_level_hidden_states(
        self,
        text: str,
        entity_start: int,
        entity_end: int
    ) -> Tuple[np.ndarray, Dict]:
        """
        获取实体对应的token-level hidden states（64维向量）
        
        这个方法直接访问模型的semantic_head输出，获取token级别的hidden states。
        然后对属于实体的tokens进行pooling，得到实体级别的向量。
        
        Args:
            text: 完整句子文本
            entity_start: 实体在文本中的字符起始位置
            entity_end: 实体在文本中的字符结束位置
        
        Returns:
            Tuple[np.ndarray, Dict]: (实体向量64维, encoding信息)
        """
        # 获取tokenizer（已经在predictor中初始化）
        tokenizer = self.predictor.tokenizer
        
        # Tokenize with offset mapping
        encoding = tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        
        # 获取offsets
        offsets = encoding["offset_mapping"]
        if isinstance(offsets, list):
            if len(offsets) > 0 and isinstance(offsets[0], list):
                offsets = offsets[0]
            offsets = torch.tensor(offsets)
        else:
            if offsets.dim() > 2:
                offsets = offsets.squeeze(0)
        
        attention_mask = encoding["attention_mask"]
        if isinstance(attention_mask, list):
            if len(attention_mask) > 0 and isinstance(attention_mask[0], list):
                attention_mask = attention_mask[0]
            attention_mask = torch.tensor(attention_mask)
        else:
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)
        
        # 创建entity mask
        token_starts = offsets[:, 0]
        token_ends = offsets[:, 1]
        entity_mask = (token_starts < entity_end) & (token_ends > entity_start) & attention_mask.bool()
        
        # 前向传播获取token-level hidden states
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask_tensor = attention_mask.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取backbone输出
            outputs = self.predictor.model.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask_tensor
            )
            last_hidden = outputs.last_hidden_state  # (1, L, 768)
            
            # 通过semantic_head获取token-level vectors (64维)
            token_vectors = self.predictor.model.semantic_head(last_hidden)  # (1, L, 64)
            token_vectors = token_vectors.squeeze(0)  # (L, 64)
        
        # 对属于实体的tokens进行mean pooling
        if entity_mask.sum() > 0:
            entity_vectors = token_vectors[entity_mask]  # (num_entity_tokens, 64)
            entity_vector = entity_vectors.mean(dim=0)  # (64,)
        else:
            # 如果没有匹配的tokens，使用所有有效tokens的均值
            valid_mask = attention_mask.bool()
            if valid_mask.sum() > 0:
                entity_vector = token_vectors[valid_mask].mean(dim=0)
            else:
                # 使用token_vectors的实际维度
                embedding_dim = token_vectors.shape[1]
                entity_vector = torch.zeros(embedding_dim, device=self.device)
        
        return entity_vector.cpu().numpy(), encoding
