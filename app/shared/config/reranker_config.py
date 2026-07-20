"""
Reranker 配置模块，负责读取重排模型相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str

@dataclass
class RerankerConfig:
    """保存 BGE Reranker 重排模型的路径、设备和精度配置。"""

    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool

# 模块级实例在查询重排阶段加载模型时使用。
reranker_config = RerankerConfig(
    bge_reranker_large=env_str("BGE_RERANKER_LARGE"),
    bge_reranker_device=env_str("BGE_RERANKER_DEVICE"),
    bge_reranker_fp16=env_bool("BGE_RERANKER_FP16"),
)
