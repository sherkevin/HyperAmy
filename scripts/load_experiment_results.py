#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一加载实验结果数据

提供统一的API加载所有实验结果，包括：
- Fusion策略网格搜索结果
- 三种方法对比结果（Monte Cristo）
- GoT跨数据集验证结果
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentDataLoader:
    """实验结果数据加载器"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        初始化数据加载器
        
        Args:
            base_dir: 项目根目录，默认为脚本所在目录的父目录
        """
        if base_dir is None:
            base_dir = project_root
        self.base_dir = Path(base_dir)
        self.outputs_dir = self.base_dir / "outputs"
        self.results_dir = self.base_dir / "results"
        self.docs_dir = self.base_dir / "docs"
    
    def load_fusion_grid_search_results(self) -> Dict[str, Any]:
        """
        加载Fusion策略网格搜索结果
        
        Returns:
            包含所有配置结果的字典
        """
        results = {
            "summary": None,
            "individual_results": [],
            "analysis_report": None
        }
        
        # 尝试加载汇总文件
        summary_file = self.outputs_dir / "fusion_strategy_grid_search" / "grid_search_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    results["summary"] = json.load(f)
                logger.info(f"加载Fusion网格搜索汇总: {summary_file}")
            except Exception as e:
                logger.warning(f"加载汇总文件失败: {e}")
        
        # 尝试加载单个结果文件
        results_dir = self.outputs_dir / "fusion_strategy_grid_search" / "results"
        if results_dir.exists():
            for result_file in results_dir.glob("result_*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        results["individual_results"].append(result_data)
                except Exception as e:
                    logger.warning(f"加载结果文件 {result_file} 失败: {e}")
            logger.info(f"加载了 {len(results['individual_results'])} 个Fusion配置结果")
        
        # 加载分析报告（从已有的markdown文件提取关键信息）
        analysis_file = self.docs_dir / "FUSION_STRATEGY_GRID_SEARCH_ANALYSIS.md"
        if analysis_file.exists():
            results["analysis_report"] = str(analysis_file)
        
        return results
    
    def load_three_methods_comparison_results(self) -> Dict[str, Any]:
        """
        加载三种方法对比实验结果（Monte Cristo）
        
        Returns:
            包含三种方法评估结果的字典
        """
        results = {
            "hipporag": None,
            "fusion": None,
            "hyperamy": None,
            "analysis_report": None
        }
        
        # 尝试从输出目录加载
        mc_dir = self.outputs_dir / "monte_cristo_comparison"
        
        # 查找结果文件（可能需要根据实际脚本输出位置调整）
        # 这里提供一个框架，实际文件位置需要根据脚本确认
        
        # 加载分析报告
        analysis_file = self.docs_dir / "EXPERIMENT_RESULTS_ANALYSIS_THREE_METHODS_FINAL.md"
        if analysis_file.exists():
            results["analysis_report"] = str(analysis_file)
            logger.info(f"找到分析报告: {analysis_file}")
        
        # 尝试从分析报告中提取关键数据
        if analysis_file.exists():
            try:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 从markdown中提取关键指标（简单提取，可能需要更复杂的解析）
                    results["extracted_metrics"] = self._extract_metrics_from_markdown(content)
            except Exception as e:
                logger.warning(f"解析分析报告失败: {e}")
        
        return results
    
    def load_got_experiment_results(self) -> Dict[str, Any]:
        """
        加载GoT跨数据集验证结果
        
        Returns:
            包含GoT实验结果的字典
        """
        results = {
            "data": None,
            "analysis_report": None
        }
        
        # 尝试从输出目录加载
        # GoT结果的位置需要根据实际脚本确认
        
        # 查找相关的分析文档
        got_docs = list(self.docs_dir.glob("*GOT*.md"))
        if got_docs:
            results["analysis_report"] = str(got_docs[0])
            logger.info(f"找到GoT分析文档: {got_docs[0]}")
        
        return results
    
    def _extract_metrics_from_markdown(self, content: str) -> Dict[str, Any]:
        """
        从Markdown分析报告中提取关键指标
        
        Args:
            content: Markdown文件内容
            
        Returns:
            提取的指标字典
        """
        metrics = {}
        
        # 简单的指标提取（可以后续改进）
        import re
        
        # 提取Recall@K指标
        recall_pattern = r'Recall@(\d+)[:\s]+(\d+\.?\d*)%'
        recalls = re.findall(recall_pattern, content)
        if recalls:
            metrics["recall"] = {f"@{k}": float(v) for k, v in recalls}
        
        # 提取MRR
        mrr_pattern = r'MRR[:\s]+(\d+\.?\d+)'
        mrr_match = re.search(mrr_pattern, content)
        if mrr_match:
            metrics["mrr"] = float(mrr_match.group(1))
        
        return metrics
    
    def get_all_experiment_data(self) -> Dict[str, Any]:
        """
        加载所有实验数据
        
        Returns:
            包含所有实验数据的字典
        """
        all_data = {
            "fusion_grid_search": self.load_fusion_grid_search_results(),
            "three_methods_comparison": self.load_three_methods_comparison_results(),
            "got_experiment": self.load_got_experiment_results(),
            "metadata": {
                "base_dir": str(self.base_dir),
                "load_time": None
            }
        }
        
        from datetime import datetime
        all_data["metadata"]["load_time"] = datetime.now().isoformat()
        
        return all_data


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="加载实验结果数据")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["fusion", "three_methods", "got", "all"],
        default="all",
        help="要加载的实验类型"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="输出JSON文件路径（可选）"
    )
    
    args = parser.parse_args()
    
    loader = ExperimentDataLoader()
    
    if args.experiment == "fusion" or args.experiment == "all":
        fusion_data = loader.load_fusion_grid_search_results()
        print(f"\nFusion网格搜索结果:")
        print(f"  汇总文件: {'存在' if fusion_data['summary'] else '不存在'}")
        print(f"  单个结果: {len(fusion_data['individual_results'])} 个")
        if args.experiment == "fusion":
            result_data = {"fusion_grid_search": fusion_data}
    
    if args.experiment == "three_methods" or args.experiment == "all":
        three_methods_data = loader.load_three_methods_comparison_results()
        print(f"\n三种方法对比结果:")
        print(f"  分析报告: {'存在' if three_methods_data['analysis_report'] else '不存在'}")
        if args.experiment == "three_methods":
            result_data = {"three_methods_comparison": three_methods_data}
    
    if args.experiment == "got" or args.experiment == "all":
        got_data = loader.load_got_experiment_results()
        print(f"\nGoT实验结果:")
        print(f"  分析报告: {'存在' if got_data['analysis_report'] else '不存在'}")
        if args.experiment == "got":
            result_data = {"got_experiment": got_data}
    
    if args.experiment == "all":
        result_data = loader.get_all_experiment_data()
    
    # 保存到文件
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
    else:
        print("\n使用 --output 参数保存结果到文件")


if __name__ == "__main__":
    main()
