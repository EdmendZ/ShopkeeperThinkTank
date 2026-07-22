"""基础设施层的向量库访问入口。

业务代码从本包取得 ``milvus_gateway``，无需直接依赖底层 Milvus 客户端工具。
"""

# 重新导出网关对象，统一调用方的导入路径。
from app.infra.vectorstore.milvus_gateway import milvus_gateway

# __all__ 明确本包对外公开的名称，避免内部实现细节被星号导入。
__all__ = ["milvus_gateway"]
