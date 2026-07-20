"""
Milvus 配置模块，负责读取向量库相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MilvusConfig:
    """保存 Milvus 地址和项目使用的 Collection 名称。

    Collection 类似关系型数据库中的表；切块、实体名称和商品主体名称分别写入
    不同 Collection，避免字段结构与查询方式相互干扰。
    """

    milvus_url: str
    chunks_collection: str
    entity_name_collection: str
    item_name_collection: str

# 环境变量只在模块加载阶段读取一次，后续代码复用该实例。
milvus_config = MilvusConfig(
    milvus_url=env_str("MILVUS_URL"),
    chunks_collection=env_str("CHUNKS_COLLECTION"),
    entity_name_collection=env_str("ENTITY_NAME_COLLECTION"),
    item_name_collection=env_str("ITEM_NAME_COLLECTION"),
)
