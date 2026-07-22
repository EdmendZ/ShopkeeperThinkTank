"""
导入链配置模块。

集中管理 Markdown 切块、主体识别、图片处理、MinerU 调用、向量化与
Milvus 入库相关的策略参数。这里的时间配置单位均为秒。
"""


# ------------------------- Markdown 文本切块 -------------------------
# 单个切块允许达到的最大字符数，用于限制异常超长块。
CHUNK_MAX_SIZE: int = 1000
# 常规切块的目标字符数。
CHUNK_SIZE: int = 600
# 相邻切块之间保留的重叠字符数，用于减少上下文被截断的问题。
CHUNK_OVERLAP: int = 20


# ------------------------- 商品主体名称识别 -------------------------
# 识别商品主体名称时，最多选取的上下文切块数量。
ITEM_NAME_CONTEXT_CHUNK_K: int = 5
# 提交给名称识别模型的上下文总字符数上限。
ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS: int = 10000


# ------------------------- 向量化与图片处理 -------------------------
# 每次提交给 Embedding 模型的文本数量；过大可能增加显存或接口压力。
EMBEDDING_BATCH_SIZE: int = 5
# 文档导入流程允许处理的图片扩展名，统一使用小写形式匹配。
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".webp",
}


# ------------------------- MinerU PDF 解析 -------------------------
# MinerU 使用的解析模型版本；"vlm" 表示视觉语言模型解析模式。
MINERU_MODEL_VERSION: str = "vlm"
# 从提交任务开始到轮询结束允许等待的最长时间，超时后停止轮询。
MINERU_POLL_TIMEOUT_SECONDS: int = 600
# 两次查询 MinerU 解析状态之间的等待时间。
MINERU_POLL_INTERVAL_SECONDS: int = 3
# 下载 MinerU 结果 ZIP 时的网络超时时间。
# 该参数只控制等待时长，不控制代理和 TLS/SSL 证书行为。
MINERU_DOWNLOAD_TIMEOUT_SECONDS: int = 30


# ------------------------- Milvus 字段约束 -------------------------
# Milvus 普通 VARCHAR 字段的默认最大长度。
MILVUS_DEFAULT_VARCHAR_MAX_LENGTH: int = 512
# 保存 Markdown 切块正文的 VARCHAR 字段最大长度。
MILVUS_CHUNK_CONTENT_MAX_LENGTH: int = 65535
# 向量字段维度，必须与当前 Embedding 模型的输出维度一致。
MILVUS_VECTOR_DIM: int = 1024

