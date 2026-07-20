"""对象存储基础设施包，对外只公开预先创建的 ``minio_gateway`` 门面。"""
from app.infra.object_storage.minio_gateway import minio_gateway

# 使用列表声明本包刻意暴露的名称，避免调用方依赖内部实现工具。
__all__ = ["minio_gateway"]
