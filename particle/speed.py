"""
Speed (Surprise) 类

计算 chunk 的惊讶值（surprise value）：输入 chunk，输出惊讶值
越惊讶代表该段 chunk 越重要
"""
import numpy as np
import math

from llm import create_client
from llm.config import API_URL_COMPLETIONS, DEFAULT_MODEL
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class Speed:
    """
    惊讶值计算类
    
    功能：输入 chunk，输出惊讶值（surprise value）
    惊讶值基于信息论中的 surprisal：surprisal = -log(p)
    值越大表示该 chunk 越意外/重要
    """
    
    def __init__(self, model_name=None):
        """
        初始化 Speed 类
        
        Args:
            model_name: 使用的模型名称，如果为 None 则使用默认模型
        """
        self.model_name = model_name or DEFAULT_MODEL
        
        # 创建 LLM 客户端（使用 specific 模式，Completion API）
        self.client = create_client(
            model_name=self.model_name,
            api_url=API_URL_COMPLETIONS,
            mode="specific"
        )
        
        logger.info(f"Speed initialized with model: {self.model_name}")
    
    def extract(self, chunk: str, aggregation: str = "mean") -> float:
        """
        提取 chunk 的惊讶值
        
        Args:
            chunk: 输入文本片段
            aggregation: 聚合方式
                - "mean": 平均惊讶值（推荐，对长度不敏感）
                - "sum": 总惊讶值（对长度敏感）
                - "max": 最大惊讶值（关注最意外的 token）
                - "geometric_mean": 几何平均概率的负对数（等价于 mean）
        
        Returns:
            float: 惊讶值，值越大表示越意外/重要
        """
        try:
            # 使用 specific 模式获取 token 概率
            # echo=True 表示返回 prompt 部分的 token 概率
            # 设置一个小的 max_tokens，因为 API 不支持 max_tokens=0
            # 当 echo=True 时，prompt 的 token 概率已经在响应中
            result = self.client.complete(
                query=chunk,
                echo=True,  # 回显 prompt，这样 prompt_tokens 包含 chunk 的所有 token
                max_tokens=1,  # 设置一个小的值（API 不支持 0），但我们只使用 prompt_tokens
                temperature=0.0,  # 温度设为 0 保证确定性
                logprobs=1  # 获取 top-1 的 logprob
            )
            
            # 获取 prompt 部分的 token 信息（即 chunk 的 token）
            prompt_tokens = result.prompt_tokens
            
            if not prompt_tokens:
                logger.warning(f"No tokens found for chunk: {chunk[:50]}...")
                return 0.0
            
            # 提取 logprobs（如果为 None，则设为负无穷，表示完全意外）
            logprobs = [
                token.logprob if token.logprob is not None else float('-inf')
                for token in prompt_tokens
            ]
            
            # 计算每个 token 的惊讶值：surprisal = -log(p) = -logprob
            surprisals = [-logprob if logprob != float('-inf') else float('inf') 
                         for logprob in logprobs]
            
            # 过滤掉无穷大值（如果所有 token 都是 None，返回 0）
            valid_surprisals = [s for s in surprisals if s != float('inf')]
            if not valid_surprisals:
                logger.warning(f"All token logprobs are None for chunk: {chunk[:50]}...")
                return 0.0
            
            # 根据聚合方式计算最终惊讶值
            if aggregation == "mean":
                surprise_value = np.mean(valid_surprisals)
            elif aggregation == "sum":
                surprise_value = np.sum(valid_surprisals)
            elif aggregation == "max":
                surprise_value = np.max(valid_surprisals)
            elif aggregation == "geometric_mean":
                # 几何平均概率的负对数 = -log(exp(mean(logprob))) = -mean(logprob)
                # 等价于 mean(-logprob) = mean(surprisal)
                surprise_value = np.mean(valid_surprisals)
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
            
            logger.debug(
                f"Chunk surprise value: {surprise_value:.4f} "
                f"(aggregation={aggregation}, tokens={len(valid_surprisals)})"
            )
            
            return float(surprise_value)
            
        except Exception as e:
            logger.error(f"Failed to extract surprise value: {e}")
            raise

