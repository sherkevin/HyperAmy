"""
LLM 配置模块

统一管理 API 配置，使用 dotenv 加载环境变量
"""

import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# API 配置
API_KEY = os.getenv("API_KEY")
API_URL_COMPLETIONS = os.getenv("API_URL_COMPLETIONS", "https://llmapi.paratera.com/v1/completions")
API_URL_CHAT = os.getenv("API_URL_CHAT", "https://llmapi.paratera.com/v1/chat/completions")
API_URL_EMBEDDINGS = os.getenv("API_URL_EMBEDDINGS", "https://llmapi.paratera.com/v1/embeddings")

# 默认模型
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "DeepSeek-V3.2")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file.")

