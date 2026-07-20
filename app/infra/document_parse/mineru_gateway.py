"""
MinerU 门面模块，统一封装 PDF 解析服务的连接配置访问。
"""
from app.infra.config import infra_config


class MinerUGateway:
    """向业务层提供 MinerU 连接配置，避免业务代码直接依赖配置对象的结构。"""

    @property
    def base_url(self) -> str:
        """
        获取 MinerU 服务基础地址。

        Returns:
            str: MinerU 接口基础 URL。
        """
        return infra_config.mineru.base_url

    @property
    def api_key(self) -> str:
        """
        获取 MinerU 服务 API Token。

        Returns:
            str: MinerU 调用所需的 Token。
        """
        return infra_config.mineru.api_key


# 模块级单例：其他模块导入后可直接使用，不必重复创建无状态门面对象。
mineru_gateway = MinerUGateway()
