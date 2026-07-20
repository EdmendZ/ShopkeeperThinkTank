"""
MCP 配置模块，负责读取联网检索相关环境变量。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class McpConfig:
    """声明 MCP 地址和密钥的数据类版本。

    注意：文件下方再次用同名普通类赋值给 ``McpConfig``，因此运行时最终可见的
    是第二个定义；这里主要保留为两种写法的学习对照。
    """

    mcp_base_url: str
    api_key: str

# 不使用 @dataclass 的普通类写法：属性初始化效果相同，但不会自动生成比较等方法。
class McpConfig:
    """运行时实际使用的 MCP 配置类。"""

    def __init__(self, mcp_base_url: str, api_key: str):
        """保存 MCP 服务地址与鉴权密钥。

        Args:
            mcp_base_url: 百炼 MCP 服务的基础地址。
            api_key: 调用联网检索服务使用的 API Key。
        """

        self.mcp_base_url = mcp_base_url  # 实例属性赋值
        self.api_key = api_key           # 实例属性赋值

# 模块导入时读取环境变量并构造共享配置对象。
mcp_config = McpConfig(
    mcp_base_url=env_str("MCP_DASHSCOPE_BASE_URL"),
    api_key=env_str("OPENAI_API_KEY"),
)
