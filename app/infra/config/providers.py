"""基础设施配置聚合器。

把分散在 ``app.shared.config`` 中的配置实例组合成一个对象，供基础设施适配器
统一取用。模块末尾会在导入时创建聚合实例，并打印应用名称。
"""

from dataclasses import dataclass, field

from app.shared.config.embedding_config import embedding_config
from app.shared.config.lm_config import lm_config
from app.shared.config.bailian_mcp_config import mcp_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.mineru_config import mineru_config
from app.shared.config.minio_config import minio_config
from app.shared.config.reranker_config import reranker_config
from app.shared.config.settings_config import settings

@dataclass
class InfraConfig:
    """集中持有应用、模型、存储和数据库配置对象。

    ``@dataclass`` 会根据字段声明自动生成初始化方法；每个 ``default_factory``
    都在创建 ``InfraConfig`` 实例时返回已经加载好的共享配置对象。
    """

    # default_factory 接收“无参数函数”，因此用 lambda 延迟返回模块级配置实例。
    app: object = field(default_factory=lambda: settings)
    llm: object = field(default_factory=lambda: lm_config)
    embedding: object = field(default_factory=lambda: embedding_config)
    reranker: object = field(default_factory=lambda: reranker_config)
    mcp: object = field(default_factory=lambda: mcp_config)
    milvus: object = field(default_factory=lambda: milvus_config)
    mineru: object = field(default_factory=lambda: mineru_config)
    minio: object = field(default_factory=lambda: minio_config)

    # 这是保留的兼容字段；当前实现与 app 一样指向 settings。
    infra_config: object = field(default_factory=lambda: settings)

# 模块级实例相当于配置单例：其他模块导入后复用同一个聚合对象。
infra_config = InfraConfig()

# 普通模块被导入时也会执行顶层语句，因此这里会产生控制台输出副作用。
print(infra_config.app.import_app_name)
