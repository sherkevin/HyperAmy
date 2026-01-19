#!/usr/bin/env python3
"""
Vibe Search / Emotional Retrieval 数据集生成脚本

利用LLM API从《基督山伯爵》文本中挖掘情绪金矿，生成专门用于验证HyperAmy优势的数据集。

数据集特点：
- 以情绪为核心，而非事实性问答
- 查询不包含人名、地名，只描述情绪氛围
- 专门针对大仲马的三种标志性情绪：绝望、冷静复仇、救赎/希望
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llm.completion_client import CompletionClient
from llm.config import API_KEY, API_URL_CHAT, DEFAULT_MODEL

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generate_vibe_dataset.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class VibeQuery:
    """Vibe查询数据类"""
    query: str
    gold_chunk_id: str
    gold_text: str
    emotion_tag: str
    emotion_intensity: float
    type: str = "vibe_search"


class VibeDatasetGenerator:
    """Vibe数据集生成器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_workers: int = 5,
        emotion_intensity_threshold: float = 8.0
    ):
        """
        初始化生成器
        
        Args:
            api_key: API密钥
            api_url: API地址
            model_name: 模型名称
            max_workers: 并发工作线程数
            emotion_intensity_threshold: 情绪密度阈值（1-10）
        """
        self.client = CompletionClient(
            api_key=api_key or API_KEY,
            chat_api_url=api_url or API_URL_CHAT,
            model_name=model_name or DEFAULT_MODEL,
            mode="normal",
            default_max_tokens=500,
            default_temperature=0.7
        )
        self.max_workers = max_workers
        self.emotion_intensity_threshold = emotion_intensity_threshold
        
        # 大仲马的三种标志性情绪
        self.emotion_categories = [
            "The Despair (绝望)",
            "The Calculated Revenge (冷静的复仇)",
            "The Redemption/Wait and Hope (救赎/等待与希望)"
        ]
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """
        调用LLM API
        
        Args:
            prompt: 提示词
            max_retries: 最大重试次数
        
        Returns:
            LLM返回的文本
        """
        for attempt in range(max_retries):
            try:
                result = self.client.complete(
                    query=prompt,
                    max_tokens=500,
                    temperature=0.7
                )
                return result.get_answer_text().strip()
            except Exception as e:
                logger.warning(f"LLM调用失败（尝试{attempt+1}/{max_retries}）: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"LLM调用最终失败: {e}")
                    return ""
        return ""
    
    def score_emotion_intensity(self, chunk_text: str, chunk_id: str) -> Tuple[float, str]:
        """
        对chunk进行情绪密度评分并识别情绪类型
        
        Args:
            chunk_text: 文本片段
            chunk_id: 片段ID
        
        Returns:
            (情绪密度分数, 情绪标签) 或 (None, None) 如果评分失败
        """
        prompt = f"""你是一位专业的文学情感分析专家。请分析以下来自大仲马《基督山伯爵》的文本片段，评估其情绪密度并识别主要情绪类型。

文本片段：
"{chunk_text}"

请按以下格式回答：
1. 情绪密度评分（1-10的整数，10表示情绪最强烈）：
2. 主要情绪类型（从以下三种中选择一种，或说明其他类型）：
   - The Despair (绝望)
   - The Calculated Revenge (冷静的复仇)
   - The Redemption/Wait and Hope (救赎/等待与希望)
   - 其他（请说明）

回答格式：
评分：[1-10的整数]
情绪类型：[情绪标签]

请直接给出评分和情绪类型，不要添加其他解释。"""
        
        response = self._call_llm(prompt)
        
        if not response:
            return None, None
        
        # 解析响应
        try:
            lines = response.split('\n')
            score = None
            emotion_tag = None
            
            for line in lines:
                line = line.strip()
                if '评分：' in line or 'Score:' in line or line.startswith('评分') or line.startswith('Score'):
                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = float(numbers[0])
                elif '情绪类型：' in line or 'Emotion:' in line or '情绪' in line:
                    # 提取情绪标签
                    if '绝望' in line or 'Despair' in line:
                        emotion_tag = "Despair"
                    elif '复仇' in line or 'Revenge' in line:
                        emotion_tag = "Calculated Revenge"
                    elif '救赎' in line or '希望' in line or 'Redemption' in line or 'Hope' in line:
                        emotion_tag = "Redemption/Wait and Hope"
                    else:
                        emotion_tag = line.split('：')[-1].split(':')[-1].strip()
            
            if score is None:
                # 尝试从响应中直接提取数字
                import re
                numbers = re.findall(r'\b(10|[1-9])\b', response)
                if numbers:
                    score = float(numbers[0])
                else:
                    logger.warning(f"无法从响应中提取评分: {response[:200]}")
                    return None, None
            
            if emotion_tag is None:
                emotion_tag = "Unknown"
            
            return float(score), emotion_tag
        
        except Exception as e:
            logger.warning(f"解析情绪评分失败（chunk_id={chunk_id}）: {e}, 响应: {response[:200]}")
            return None, None
    
    def generate_vibe_query(self, chunk_text: str, emotion_tag: str) -> str:
        """
        为高情绪密度的chunk生成"情绪氛围查询"
        
        Args:
            chunk_text: 文本片段
            emotion_tag: 情绪标签
        
        Returns:
            生成的查询字符串
        """
        prompt = f"""你是一位专业的文学搜索引擎设计专家。用户想要找到以下这段文字，但他不记得具体的人名或地点，只记得当时那种强烈的情绪氛围或心理描写。

文本片段：
"{chunk_text}"

情绪类型：{emotion_tag}

请生成一个搜索查询，要求：
1. **不包含人名、地名**（如"唐泰斯"、"伊夫堡"等）
2. **不包含具体情节细节**（如"被关在监狱"、"收到信"等）
3. **只描述情绪氛围、心理状态、内心独白**
4. **侧重**：{emotion_tag}

反例（不要这样）：
- "唐泰斯在监狱里说了什么？"（包含人名和地点）
- "主角如何逃离伊夫堡？"（包含地名和具体情节）

正例（要这样）：
- "找一段描写主角在长期监禁后，内心从希望转变为彻底虚无和麻木的心理独白。"（只描述情绪和心理）
- "描写那种复仇前的冰冷冷静，表面礼貌但内心充满杀意的心理状态。"（只描述情绪氛围）

请直接给出查询，不要添加任何解释或前缀。"""
        
        response = self._call_llm(prompt)
        
        if not response:
            return None
        
        # 清理响应（移除可能的引号、前缀等）
        query = response.strip().strip('"').strip("'").strip()
        
        # 移除常见的前缀
        prefixes = ["查询：", "Query:", "搜索查询：", "Search Query:", "查询", "Query"]
        for prefix in prefixes:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        return query if query else None
    
    def process_chunk(self, chunk: Dict, chunk_idx: int, total: int) -> Optional[VibeQuery]:
        """
        处理单个chunk：评分 → 筛选 → 生成查询
        
        Args:
            chunk: chunk数据字典
            chunk_idx: chunk索引
            total: 总chunk数
        
        Returns:
            VibeQuery对象，如果chunk不符合要求则返回None
        """
        chunk_text = chunk.get('input') or chunk.get('text') or chunk.get('content') or chunk.get('chunk_text', '')
        chunk_id = chunk.get('chunk_id') or chunk.get('id') or f'chunk_{chunk_idx}'
        
        if not chunk_text or len(chunk_text.strip()) < 50:
            return None
        
        logger.info(f"[{chunk_idx+1}/{total}] 处理chunk: {chunk_id[:50]}...")
        
        # 步骤1：情绪密度评分
        score, emotion_tag = self.score_emotion_intensity(chunk_text, chunk_id)
        
        if score is None or emotion_tag is None:
            logger.warning(f"  ⚠️  情绪评分失败，跳过")
            return None
        
        logger.info(f"  情绪评分: {score}/10, 情绪类型: {emotion_tag}")
        
        # 步骤2：筛选（只保留情绪密度>=threshold的）
        if score < self.emotion_intensity_threshold:
            logger.info(f"  ❌ 情绪密度不足（{score} < {self.emotion_intensity_threshold}），跳过")
            return None
        
        logger.info(f"  ✅ 情绪密度足够（{score} >= {self.emotion_intensity_threshold}），生成查询...")
        
        # 步骤3：生成查询
        query = self.generate_vibe_query(chunk_text, emotion_tag)
        
        if not query:
            logger.warning(f"  ⚠️  查询生成失败，跳过")
            return None
        
        logger.info(f"  ✅ 查询生成成功: {query[:100]}...")
        
        return VibeQuery(
            query=query,
            gold_chunk_id=chunk_id,
            gold_text=chunk_text,
            emotion_tag=emotion_tag,
            emotion_intensity=score,
            type="vibe_search"
        )
    
    def generate_dataset(
        self,
        chunks_file: str,
        output_file: str,
        max_queries: int = 50,
        sample_size: Optional[int] = None
    ) -> Dict:
        """
        生成完整的数据集
        
        Args:
            chunks_file: chunks文件路径（JSONL格式）
            output_file: 输出文件路径（JSON格式）
            max_queries: 最大生成查询数量
            sample_size: 采样chunk数量（None表示使用全部）
        
        Returns:
            生成的数据集字典
        """
        logger.info("=" * 80)
        logger.info("Vibe Search / Emotional Retrieval 数据集生成")
        logger.info("=" * 80)
        logger.info(f"输入文件: {chunks_file}")
        logger.info(f"输出文件: {output_file}")
        logger.info(f"情绪密度阈值: {self.emotion_intensity_threshold}/10")
        logger.info(f"目标查询数量: {max_queries}")
        logger.info("=" * 80)
        
        # 加载chunks
        logger.info(f"\n【步骤1】加载chunks...")
        chunks = []
        chunks_path = Path(chunks_file)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chunks文件不存在: {chunks_file}")
        
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON解析失败，跳过: {e}")
        
        logger.info(f"✅ 加载了 {len(chunks)} 个chunks")
        
        # 采样（如果指定）
        if sample_size and sample_size < len(chunks):
            import random
            random.seed(42)
            chunks = random.sample(chunks, sample_size)
            logger.info(f"✅ 采样为 {len(chunks)} 个chunks")
        
        # 步骤2：并发处理chunks（评分 → 筛选 → 生成查询）
        logger.info(f"\n【步骤2】并发处理chunks（情绪评分 + 查询生成）...")
        logger.info(f"并发工作线程数: {self.max_workers}")
        
        vibe_queries = []
        processed = 0
        successful = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, idx, len(chunks)): (idx, chunk)
                for idx, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in as_completed(future_to_chunk):
                chunk_idx, chunk = future_to_chunk[future]
                processed += 1
                
                try:
                    result = future.result()
                    if result:
                        vibe_queries.append(result)
                        successful += 1
                        logger.info(f"✅ [{successful}/{max_queries}] 成功生成查询（chunk_idx={chunk_idx}）")
                        
                        # 如果达到目标数量，停止处理
                        if successful >= max_queries:
                            logger.info(f"✅ 已达到目标查询数量（{max_queries}），停止处理")
                            # 取消剩余任务
                            for f in future_to_chunk:
                                if not f.done():
                                    f.cancel()
                            break
                except Exception as e:
                    logger.error(f"处理chunk失败（chunk_idx={chunk_idx}）: {e}")
                
                # 进度报告
                if processed % 10 == 0:
                    logger.info(f"进度: {processed}/{len(chunks)} 已处理, {successful} 个成功生成查询")
        
        logger.info(f"\n【步骤3】处理完成！")
        logger.info(f"总处理: {processed} 个chunks")
        logger.info(f"成功生成: {len(vibe_queries)} 个查询")
        
        # 步骤4：构建数据集
        dataset = {
            "dataset_name": "monte_cristo_vibe_search",
            "description": "Vibe Search / Emotional Retrieval数据集 - 专门用于验证HyperAmy优势",
            "version": "1.0",
            "emotion_intensity_threshold": self.emotion_intensity_threshold,
            "total_queries": len(vibe_queries),
            "data": [
                {
                    "query": q.query,
                    "gold_chunk_id": q.gold_chunk_id,
                    "gold_text": q.gold_text,
                    "emotion_tag": q.emotion_tag,
                    "emotion_intensity": q.emotion_intensity,
                    "type": q.type
                }
                for q in vibe_queries
            ]
        }
        
        # 步骤5：保存数据集
        logger.info(f"\n【步骤4】保存数据集...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ 数据集已保存到: {output_file}")
        
        # 统计信息
        logger.info(f"\n【数据集统计】")
        emotion_counts = {}
        for q in vibe_queries:
            emotion_counts[q.emotion_tag] = emotion_counts.get(q.emotion_tag, 0) + 1
        
        logger.info(f"情绪类型分布:")
        for emotion, count in emotion_counts.items():
            logger.info(f"  - {emotion}: {count} 个查询")
        
        avg_intensity = sum(q.emotion_intensity for q in vibe_queries) / len(vibe_queries) if vibe_queries else 0
        logger.info(f"平均情绪密度: {avg_intensity:.2f}/10")
        
        logger.info("=" * 80)
        logger.info("✅ 数据集生成完成！")
        logger.info("=" * 80)
        
        return dataset


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成Vibe Search数据集")
    parser.add_argument(
        "--chunks_file",
        type=str,
        default=str(project_root / "data" / "training" / "monte_cristo_train_full.jsonl"),
        help="输入chunks文件路径（JSONL格式）"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(project_root / "data" / "public_benchmark" / "monte_cristo_vibe_search.json"),
        help="输出数据集文件路径（JSON格式）"
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=50,
        help="最大生成查询数量（默认50）"
    )
    parser.add_argument(
        "--emotion_threshold",
        type=float,
        default=8.0,
        help="情绪密度阈值（1-10，默认8.0）"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="并发工作线程数（默认5）"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="采样chunk数量（None表示使用全部，用于快速测试）"
    )
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = VibeDatasetGenerator(
        emotion_intensity_threshold=args.emotion_threshold,
        max_workers=args.max_workers
    )
    
    # 生成数据集
    dataset = generator.generate_dataset(
        chunks_file=args.chunks_file,
        output_file=args.output_file,
        max_queries=args.max_queries,
        sample_size=args.sample_size
    )
    
    logger.info(f"\n✅ 数据集生成完成！共生成 {len(dataset['data'])} 个查询")
    logger.info(f"输出文件: {args.output_file}")


if __name__ == "__main__":
    main()
