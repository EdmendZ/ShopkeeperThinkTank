"""
LLM 配置模块，负责读取对话模型与视觉模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_float, env_str


@dataclass
class LLMConfig:
    """保存文本模型和视觉模型的连接与推理参数。

    ``base_url``/``api_key`` 用于连接 OpenAI 兼容接口，``lv_model`` 与
    ``llm_model`` 分别选择视觉和文本模型，温度控制生成随机性。
    """

    base_url: str
    api_key: str
    lv_model: str
    llm_model: str
    llm_temperature: float


# 创建共享配置实例时完成环境变量到 float 等目标类型的转换。
lm_config = LLMConfig(
    base_url=env_str("OPENAI_BASE_URL"),
    api_key=env_str("OPENAI_API_KEY"),
    lv_model=env_str("VL_MODEL"),
    llm_model=env_str("LLM_DEFAULT_MODEL"),
    llm_temperature=env_float("LLM_DEFAULT_TEMPERATURE"),
)
