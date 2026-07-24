"""查询 API 的 Pydantic request/response schemas。"""

from typing import Any
from pydantic import BaseModel,Field

class QueryRequest(BaseModel):
    """查询请求；未提供 ``session_id`` 时由 API 创建新会话。"""
    session_id: str = Field(None, description="会话ID")
    query: str | None = Field(..., description="原始查询")
    is_stream: bool = Field(False, description="是否流式返回")

class AsyncQueryResponse(BaseModel):
    """流式查询受理结果；客户端据此订阅对应 session 的 SSE endpoint。"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应信息")



class QueryResponse(BaseModel):
    """非流式查询的最终答案、图片引用与已完成节点列表。"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="响应信息")
    answer: str = Field("", description="答案")
    image_urls: list[str] = Field(description="图片URL列表" ,default_factory=list)
    done_list: list[str]  = Field(description="已完成任务列表", default_factory=list)


class ClearHistoryResponse(BaseModel):
    """清空聊天记录后的操作摘要。"""
    message: str = Field(..., description="操作提示信息")
    deleted_count: int = Field(..., description="成功删除的消息条数")


class HistoryItem(BaseModel):
    """一条规范化聊天消息；list 字段使用 factory 避免实例间共享状态。"""
    id: str = Field(default="", description="消息ID")
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)
    ts: Any = None


class HistoryResponse(BaseModel):
    """指定 session 的聊天历史集合。"""
    session_id: str
    items: list[HistoryItem] = Field(default_factory=list)
