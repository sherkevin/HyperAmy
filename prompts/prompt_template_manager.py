"""
Prompt Template Manager

管理项目中的 prompt 模板，参照 hipporag 的组织形式
"""
import os
import importlib.util
from string import Template
from typing import Dict, List, Union, Any, Optional
from hipporag.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptTemplateManager:
    """
    Prompt 模板管理器
    
    从 templates 目录加载 prompt 模板，支持字符串模板和聊天历史模板
    """
    
    def __init__(self, role_mapping: Optional[Dict[str, str]] = None):
        """
        初始化 PromptTemplateManager
        
        Args:
            role_mapping: 角色映射，用于调整不同 LLM 提供商的角色定义
        """
        self.role_mapping = role_mapping or {
            "system": "system",
            "user": "user",
            "assistant": "assistant"
        }
        
        # 获取 templates 目录路径
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        self.templates_dir = os.path.join(package_dir, "templates")
        
        self.templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = {}
        
        self._load_templates()
    
    def _load_templates(self) -> None:
        """从 templates 目录加载所有模板"""
        if not os.path.exists(self.templates_dir):
            logger.warning(f"Templates directory '{self.templates_dir}' does not exist. Creating it.")
            os.makedirs(self.templates_dir, exist_ok=True)
            return
        
        logger.info(f"Loading templates from directory: {self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]
                script_path = os.path.join(self.templates_dir, filename)
                
                try:
                    spec = importlib.util.spec_from_file_location(script_name, script_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if not hasattr(module, "prompt_template"):
                        logger.warning(f"Module '{script_name}' does not define a 'prompt_template'. Skipping.")
                        continue
                    
                    prompt_template = module.prompt_template
                    
                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item 
                        for item in prompt_template
                    ):
                        # 调整角色映射
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(item["role"], item["role"])
                            if isinstance(item["content"], str):
                                item["content"] = Template(item["content"])
                        self.templates[script_name] = prompt_template
                    else:
                        logger.warning(
                            f"Invalid prompt_template format in '{script_name}.py'. "
                            f"Must be a Template, str, or List[Dict]. Skipping."
                        )
                        continue
                    
                    logger.debug(f"Successfully loaded template '{script_name}' from '{script_path}'.")
                    
                except Exception as e:
                    logger.error(f"Failed to load template from '{script_path}': {e}")
    
    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        获取指定名称的模板
        
        Args:
            name: 模板名称
        
        Returns:
            模板对象
        
        Raises:
            ValueError: 如果模板不存在
        """
        if name not in self.templates:
            available = ", ".join(self.templates.keys())
            raise ValueError(
                f"Template '{name}' not found. Available templates: {available}"
            )
        return self.templates[name]
    
    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        渲染模板
        
        Args:
            name: 模板名称
            **kwargs: 模板变量
        
        Returns:
            渲染后的字符串或聊天历史列表
        """
        template = self.get_template(name)
        
        if isinstance(template, Template):
            # 渲染字符串模板
            try:
                result = template.substitute(**kwargs)
                logger.debug(f"Successfully rendered template '{name}' with variables: {kwargs}.")
                return result
            except KeyError as e:
                logger.error(f"Missing variable for template '{name}': {e}")
                raise ValueError(f"Missing variable for template '{name}': {e}")
        elif isinstance(template, list):
            # 渲染聊天历史模板
            try:
                rendered_list = [
                    {
                        "role": item["role"],
                        "content": item["content"].substitute(**kwargs) 
                        if isinstance(item["content"], Template) 
                        else item["content"]
                    }
                    for item in template
                ]
                logger.debug(f"Successfully rendered chat history template '{name}' with variables: {kwargs}.")
                return rendered_list
            except KeyError as e:
                logger.error(f"Missing variable in chat history template '{name}': {e}")
                raise ValueError(f"Missing variable in chat history template '{name}': {e}")
    
    def list_template_names(self) -> List[str]:
        """列出所有可用的模板名称"""
        return list(self.templates.keys())

