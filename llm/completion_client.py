"""
LLM Completion API Client

封装了使用 Completion 接口调用大语言模型的功能，支持获取 token 级别的概率信息。
"""

import requests
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class TokenInfo:
    """Token 信息"""
    token: str
    logprob: Optional[float]
    probability: Optional[float]
    
    def __repr__(self):
        prob_str = f"{self.probability:.2%}" if self.probability is not None else "N/A"
        logprob_str = f"{self.logprob:.4f}" if self.logprob is not None else "None"
        return f"Token(token={repr(self.token)}, logprob={logprob_str}, prob={prob_str})"


@dataclass
class CompletionResult:
    """Completion 结果"""
    prompt_tokens: List[TokenInfo]
    answer_tokens: List[TokenInfo]
    answer_text: str
    usage: Dict[str, int]
    raw_response: Dict[str, Any]
    
    def get_prompt_text(self) -> str:
        """获取完整的 prompt 文本"""
        return "".join([t.token for t in self.prompt_tokens])
    
    def get_answer_text(self) -> str:
        """获取完整的回答文本"""
        return self.answer_text
    
    def print_analysis(self):
        """打印详细的分析结果"""
        print(f"原始输入: {self.get_prompt_text()}")
        print("=" * 60)
        
        # Prompt 部分
        print(f"【Part 1: Prompt Analysis】 (长度: {len(self.prompt_tokens)})")
        print(f"{'Token':<15} | {'Logprob':<10} | {'Prob':<10}")
        print("-" * 45)
        for token_info in self.prompt_tokens:
            prob_str = f"{token_info.probability:.2%}" if token_info.probability is not None else "N/A"
            logprob_str = f"{token_info.logprob:.4f}" if token_info.logprob is not None else "None"
            print(f"{repr(token_info.token):<15} | {logprob_str:<10} | {prob_str:<10}")
        
        print("\n" + "=" * 60)
        
        # Answer 部分
        print(f"【Part 2: Answer Analysis】 (长度: {len(self.answer_tokens)})")
        print(f"回答内容: {self.answer_text.strip()}")
        print("-" * 45)
        print(f"{'Token':<15} | {'Logprob':<10} | {'Prob':<10}")
        print("-" * 45)
        for token_info in self.answer_tokens:
            prob_str = f"{token_info.probability:.2%}" if token_info.probability is not None else "N/A"
            logprob_str = f"{token_info.logprob:.4f}" if token_info.logprob is not None else "None"
            print(f"{repr(token_info.token):<15} | {logprob_str:<10} | {prob_str:<10}")


class CompletionClient:
    """LLM Completion API 客户端"""
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://llmapi.paratera.com/v1/completions",
        model_name: str = "DeepSeek-V3.2",
        default_max_tokens: int = 100,
        default_temperature: float = 0.7,
        default_stop: Optional[List[str]] = None
    ):
        """
        初始化 Completion 客户端
        
        Args:
            api_key: API 密钥
            api_url: API 地址，默认为 paratera.com
            model_name: 模型名称，默认为 DeepSeek-V3.2
            default_max_tokens: 默认最大 token 数
            default_temperature: 默认温度参数
            default_stop: 默认停止词列表
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_stop = default_stop or ["User:", "\n\nUser", "<|endoftext|>", "<end_of_text>"]
    
    def _build_prompt(self, query: str, prompt_template: Optional[str] = None) -> str:
        """
        构建 prompt
        
        Args:
            query: 用户查询
            prompt_template: Prompt 模板，如果为 None 则使用默认模板
        
        Returns:
            构建好的 prompt
        """
        if prompt_template is None:
            # 默认使用简单的对话格式
            return f"User: {query}\nAssistant:"
        else:
            return prompt_template.format(query=query)
    
    def _parse_tokens(
        self,
        tokens: List[str],
        logprobs: List[Optional[float]]
    ) -> List[TokenInfo]:
        """
        解析 token 列表为 TokenInfo 列表
        
        Args:
            tokens: Token 列表
            logprobs: Log 概率列表
        
        Returns:
            TokenInfo 列表
        """
        result = []
        for token, logprob in zip(tokens, logprobs):
            prob = math.exp(logprob) if logprob is not None else None
            result.append(TokenInfo(
                token=token,
                logprob=logprob,
                probability=prob
            ))
        return result
    
    def complete(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        logprobs: int = 1,
        echo: bool = True,
        **kwargs
    ) -> CompletionResult:
        """
        调用 Completion API
        
        Args:
            query: 用户查询
            prompt_template: Prompt 模板，如果为 None 则使用默认模板
            max_tokens: 最大 token 数，如果为 None 则使用默认值
            temperature: 温度参数，如果为 None 则使用默认值
            stop: 停止词列表，如果为 None 则使用默认值
            logprobs: Log 概率数量，默认为 1
            echo: 是否回显 prompt，默认为 True（必须为 True 才能获取 prompt 的概率）
            **kwargs: 其他 API 参数
        
        Returns:
            CompletionResult 对象
        
        Raises:
            requests.RequestException: 请求失败时抛出
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        full_prompt = self._build_prompt(query, prompt_template)
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "logprobs": logprobs,
            "echo": echo,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "stop": stop if stop is not None else self.default_stop,
            **kwargs
        }
        
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # 获取 token 数量信息
        usage = data.get('usage', {})
        prompt_token_count = usage.get('prompt_tokens', 0)
        
        # 获取 token 和 logprob 数据
        choices = data['choices'][0]
        all_tokens = choices['logprobs']['tokens']
        all_logprobs = choices['logprobs']['token_logprobs']
        
        # 切分 prompt 和 answer
        prompt_tokens = all_tokens[:prompt_token_count]
        prompt_logprobs = all_logprobs[:prompt_token_count]
        answer_tokens = all_tokens[prompt_token_count:]
        answer_logprobs = all_logprobs[prompt_token_count:]
        
        # 解析为 TokenInfo
        prompt_token_infos = self._parse_tokens(prompt_tokens, prompt_logprobs)
        answer_token_infos = self._parse_tokens(answer_tokens, answer_logprobs)
        
        # 构建回答文本
        answer_text = "".join(answer_tokens)
        
        return CompletionResult(
            prompt_tokens=prompt_token_infos,
            answer_tokens=answer_token_infos,
            answer_text=answer_text,
            usage=usage,
            raw_response=data
        )
    
    def get_answer(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        快速获取回答文本（不返回详细概率信息）
        
        Args:
            query: 用户查询
            prompt_template: Prompt 模板
            **kwargs: 其他参数传递给 complete 方法
        
        Returns:
            回答文本
        """
        result = self.complete(query, prompt_template=prompt_template, **kwargs)
        return result.get_answer_text().strip()


# 便捷函数：创建默认客户端
def create_client(
    api_key: str,
    model_name: str = "DeepSeek-V3.2",
    api_url: str = "https://llmapi.paratera.com/v1/completions"
) -> CompletionClient:
    """
    创建默认的 Completion 客户端
    
    Args:
        api_key: API 密钥
        model_name: 模型名称
        api_url: API 地址
    
    Returns:
        CompletionClient 实例
    """
    return CompletionClient(
        api_key=api_key,
        api_url=api_url,
        model_name=model_name
    )

