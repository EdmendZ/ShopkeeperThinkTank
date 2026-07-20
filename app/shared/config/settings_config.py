"""
应用基础配置模块，负责读取导入服务与查询服务的启动配置。
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv


# 把项目 .env 文件加载到进程环境中；已有同名系统环境变量默认不会被覆盖。
load_dotenv()


@dataclass
class AppSettings:
    """保存两个 Web 服务共用的名称、监听地址和跨域配置。

    这些默认值在类定义执行时读取环境变量，再由 ``@dataclass`` 生成的初始化方法
    复制到实例。端口显式转为 ``int``，便于直接传给 Web 服务器。
    """

    import_app_name: str = os.getenv("IMPORT_APP_NAME", "Enterprise RAG Import Service")
    query_app_name: str = os.getenv("QUERY_APP_NAME", "Enterprise RAG Query Service")
    app_env: str = os.getenv("APP_ENV", "dev")
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    import_app_port: int = int(os.getenv("IMPORT_APP_PORT", "8000"))
    query_app_port: int = int(os.getenv("QUERY_APP_PORT", "8001"))
    # 生成器表达式逐项去除空白并过滤空字符串，tuple() 再将结果固化为不可变元组。
    cors_origins: tuple[str, ...] = tuple(
        item.strip() for item in os.getenv("CORS_ORIGINS", "*").split(",") if item.strip()
    )

# 其他模块导入并复用这个应用级配置实例。
settings = AppSettings()
