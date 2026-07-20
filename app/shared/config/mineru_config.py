"""
MinerU 配置模块，负责读取文档解析服务相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MinerUConfig:
    """保存 MinerU 文档解析服务的地址和鉴权令牌。"""

    base_url: str
    api_key: str


# 模块级实例作为 PDF 解析网关的统一配置来源。
mineru_config = MinerUConfig(
    base_url=env_str("MINERU_BASE_URL"),
    api_key=env_str("MINERU_API_TOKEN"),
)
