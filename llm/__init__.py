"""
LLM 客户端模块

提供各种 LLM API 的客户端封装
"""

from .completion_client import (
    CompletionClient,
    CompletionResult,
    ChatResult,
    TokenInfo,
    create_client
)
from .config import (
    API_KEY,
    BASE_URL,
    API_URL_COMPLETIONS,
    API_URL_CHAT,
    API_URL_EMBEDDINGS,
    DEFAULT_MODEL,
    DEFAULT_EMBEDDING_MODEL
)

__all__ = [
    'CompletionClient',
    'CompletionResult',
    'ChatResult',
    'TokenInfo',
    'create_client',
    'API_KEY',
    'BASE_URL',
    'API_URL_COMPLETIONS',
    'API_URL_CHAT',
    'API_URL_EMBEDDINGS',
    'DEFAULT_MODEL',
    'DEFAULT_EMBEDDING_MODEL',
]

