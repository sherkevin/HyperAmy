"""
Prompts 模块

统一管理项目中的 prompt 模板
"""
try:
    from .prompt_template_manager import PromptTemplateManager
    __all__ = ['PromptTemplateManager']
except ImportError:
    # 如果导入失败，可能是路径问题，不影响使用
    __all__ = []

