"""
MinIO 门面模块，统一封装对象存储客户端与桶配置访问。
"""
from minio import Minio

from app.shared.clients.minio_utils import get_minio_client
from app.infra.config import infra_config


class MinIOGateway:
    """统一暴露 MinIO 配置、客户端和图片 URL 拼接能力。"""

    @property
    def bucket_name(self) -> str:
        """返回存放知识库文件的桶名称。"""
        return infra_config.minio.bucket_name

    @property
    def image_dir(self) -> str:
        """返回桶内图片对象使用的公共目录前缀。"""
        return infra_config.minio.minio_img_dir

    def client(self) -> Minio:
        """获取已经按项目配置初始化的 MinIO 客户端。"""
        return get_minio_client()

    def build_image_url(self, stem: str, image_name: str) -> str:
        """
        构建图片对象的公开访问地址。

        Args:
            stem: 当前文档或任务的目录名。
            image_name: 图片文件名。

        Returns:
            str: 对应图片在 MinIO 中的访问 URL。
        """
        protocol = "https" if infra_config.minio.minio_secure else "http"
        return (
            f"{protocol}://{infra_config.minio.endpoint}/"
            f"{self.bucket_name}{self.image_dir}/{stem}/{image_name}"
        )


# 模块级单例让上层只依赖门面，不需要分别导入配置和底层客户端工具。
minio_gateway = MinIOGateway()
