"""
Embedding 配置模块，负责读取向量模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


@dataclass
class EmbeddingConfig:
    """保存 BGE-M3 向量模型的加载参数。

    字段依次描述本地模型路径/名称、运行设备以及是否使用 FP16。``@dataclass``
    根据这些类型注解自动生成 ``__init__`` 等方法。
    """

    bge_m3_path: str
    bge_m3: str
    bge_device: str
    bge_fp16: bool

# 模块导入时把字符串环境变量转换为带类型的配置对象，供模型加载器复用。
embedding_config = EmbeddingConfig(
    bge_m3_path=env_str("BGE_M3_PATH"),
    bge_m3=env_str("BGE_M3"),
    bge_device=env_str("BGE_DEVICE"),
    bge_fp16=env_bool("BGE_FP16"),
)
