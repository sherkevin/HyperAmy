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

# 处理 BASE_URL：提取基础 URL（去除所有端点路径）
# 如果包含 /chat/completions，提取基础部分
if "/chat/completions" in BASE_URL_RAW:
    BASE_URL = BASE_URL_RAW.split("/chat/completions")[0].rstrip('/')
elif BASE_URL_RAW.endswith("/chat"):
    BASE_URL = BASE_URL_RAW[:-5]  # 移除 /chat
elif BASE_URL_RAW.endswith("/chat/"):
    BASE_URL = BASE_URL_RAW[:-6]  # 移除 /chat/
elif "/chat/" in BASE_URL_RAW:
    BASE_URL = BASE_URL_RAW.split("/chat/")[0].rstrip('/')
else:
    BASE_URL = BASE_URL_RAW

# 确保BASE_URL是基础URL（不包含v1后面的路径）
if BASE_URL.endswith("/v1"):
    pass  # 正确
elif "/v1/" in BASE_URL or BASE_URL.endswith("/v1"):
    # 已经是正确的格式
    pass
else:
    # 如果不是以/v1结尾，添加/v1
    if not BASE_URL.endswith("/v1"):
        if "/v1" not in BASE_URL:
            BASE_URL = f"{BASE_URL}/v1"

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
