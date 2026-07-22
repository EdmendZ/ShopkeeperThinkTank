"""导入接口使用的响应数据模型。

FastAPI 会根据这些 Pydantic 模型校验返回值，并自动生成 OpenAPI（Swagger）接口文档。
客户端通常先上传文件取得 ``task_ids``，再按任务 ID 轮询状态接口。
"""
from pydantic import BaseModel

# 继承 BaseModel 后，Pydantic 会按字段类型构造响应 JSON，并让 FastAPI 将字段展示到接口文档。
class UploadResponse(BaseModel):
    """``POST /upload`` 的返回结构。"""

    # 响应 JSON 内的业务状态码；它与 HTTP 状态码是两个概念。
    code: int = 200
    # 给调用方阅读的简短提示文本。
    message: str
    # 每个上传文件对应一个 UUID，后续查询进度时使用其中某一个值。
    task_ids: list[str]


class ImportStatusResponse(BaseModel):
    """``GET /status/{task_id}`` 的进度查询结果。"""

    code: int = 200
    # 与上传接口返回值对应的任务标识。
    task_id: str
    # 例如 processing、completed 或 failed；尚未写入状态时可能为空。
    status: str | None = None
    # 已完成节点的中文展示名，列表顺序就是节点完成顺序。
    done_list: list[str]
    # 当前正在执行节点的中文展示名。
    running_list: list[str]
