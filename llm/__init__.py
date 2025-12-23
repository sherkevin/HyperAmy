"""
LLM 客户端模块

提供各种 LLM API 的客户端封装
"""

from .completion_client import (
    CompletionClient,
    CompletionResult,
    TokenInfo,
    create_client
)
from .config import (
    API_KEY,
    API_URL_COMPLETIONS,
    API_URL_CHAT,
    API_URL_EMBEDDINGS,
    DEFAULT_MODEL
)

__all__ = [
    'CompletionClient',
    'CompletionResult',
    'TokenInfo',
    'create_client',
    'API_KEY',
    'API_URL_COMPLETIONS',
    'API_URL_CHAT',
    'API_URL_EMBEDDINGS',
    'DEFAULT_MODEL',
]

