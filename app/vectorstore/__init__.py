"""向量库访问入口。

外部业务代码只需从本包导入 ``milvus_gateway``，不必了解底层客户端、集合名称和
混合检索工具分别放在哪个文件中。
"""

# 重新导出网关对象，让调用方使用更短、更稳定的导入路径。
from app.infra.vectorstore.milvus_gateway import milvus_gateway

# __all__ 规定 ``from app.vectorstore import *`` 时允许导出的公开名称。
# 下划线开头的内部实现不会因为新增文件而意外暴露给调用方。
__all__ = ["milvus_gateway"]
