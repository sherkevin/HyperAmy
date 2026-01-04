"""
LLM 配置模块

统一管理 API 配置，使用 dotenv 加载环境变量
"""

import os
from dotenv import load_dotenv
load_dotenv()


# API 配置
API_KEY = os.getenv("API_KEY")
BASE_URL_RAW = os.getenv("BASE_URL", "https://llmapi.paratera.com/v1").strip().strip('"').strip("'").rstrip('/')

# 处理 BASE_URL：如果以 /chat/ 结尾，则提取基础 URL
if BASE_URL_RAW.endswith("/chat"):
    BASE_URL = BASE_URL_RAW[:-5]  # 移除 /chat
elif BASE_URL_RAW.endswith("/chat/"):
    BASE_URL = BASE_URL_RAW[:-6]  # 移除 /chat/
else:
    BASE_URL = BASE_URL_RAW

# 构建完整的 API URL
API_URL_COMPLETIONS = os.getenv("API_URL_COMPLETIONS", f"{BASE_URL}/completions")
API_URL_CHAT = os.getenv("API_URL_CHAT", f"{BASE_URL}/chat/completions")
API_URL_EMBEDDINGS = os.getenv("API_URL_EMBEDDINGS", f"{BASE_URL}/embeddings")

# 默认模型（仅作为默认值，不从环境变量读取，由外部传入）
DEFAULT_MODEL = "DeepSeek-V3.2"
DEFAULT_EMBEDDING_MODEL = "GLM-Embedding-3"

# GoT 实验配置
BETA_WARPING = 10  # 庞加莱畸变参数
MASS_THRESHOLD = 0.8  # 高质量块阈值
CHUNK_SIZE = 300  # 分块大小（词数）
CHUNK_OVERLAP = 50  # 重叠大小（词数）

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file.")
