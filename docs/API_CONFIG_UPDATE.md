# API配置更新说明

## 更新内容

用户提供了新的API配置：

- **API_KEY**: `sk-7870u-nMQ69cSLRmIAxt2A`
- **BASE_URL**: `https://llmapi.paratera.com/v1/chat/`

## 更新步骤

### 1. 更新 .env 文件

编辑项目根目录下的 `.env` 文件，更新以下内容：

```env
API_KEY=sk-7870u-nMQ69cSLRmIAxt2A
BASE_URL=https://llmapi.paratera.com/v1/chat/
```

**注意：**
- `BASE_URL` 末尾可以包含 `/chat/`，代码会自动处理
- 或者设置为 `https://llmapi.paratera.com/v1`，代码会自动添加 `/chat/completions`

### 2. 验证配置

运行以下命令验证API连接：

```bash
python3 -c "
from llm.completion_client import CompletionClient
client = CompletionClient()
result = client.complete(query='测试：1+1=?', mode='normal', max_tokens=10)
print('✅ API连接成功')
print('响应:', result.get_answer_text())
"
```

### 3. 配置说明

`llm/config.py` 会自动处理 BASE_URL：

- 如果 BASE_URL 以 `/chat` 或 `/chat/` 结尾，会自动移除
- 最终构建的 URL 格式：`{BASE_URL}/chat/completions`

例如：
- `BASE_URL=https://llmapi.paratera.com/v1/chat/` → `https://llmapi.paratera.com/v1/chat/completions`
- `BASE_URL=https://llmapi.paratera.com/v1` → `https://llmapi.paratera.com/v1/chat/completions`

## 问题排查

如果仍然出现 401 Unauthorized：

1. **检查 API_KEY 是否正确**
   ```bash
   python3 -c "from llm.config import API_KEY; print(f'API_KEY: {API_KEY[:10]}...')"
   ```

2. **检查 BASE_URL 是否正确**
   ```bash
   python3 -c "from llm.config import BASE_URL, API_URL_CHAT; print(f'BASE_URL: {BASE_URL}'); print(f'API_URL_CHAT: {API_URL_CHAT}')"
   ```

3. **确认 API 服务是否可用**
   ```bash
   curl -X POST https://llmapi.paratera.com/v1/chat/completions \
     -H "Authorization: Bearer sk-7870u-nMQ69cSLRmIAxt2A" \
     -H "Content-Type: application/json" \
     -d '{"model":"DeepSeek-V3.2","messages":[{"role":"user","content":"test"}]}'
   ```

