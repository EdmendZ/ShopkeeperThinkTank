"""聊天历史 Repository，为业务层隔离 MongoDB helper 的具体实现。"""
from app.shared.clients.mongo_history_utils import (
    clear_history,
    get_recent_messages,
    save_chat_message,
    update_message_item_names,
)


class HistoryRepository:
    """提供会话消息的查询、写入、清理和主体名称回填操作。

    当前实现直接委托给 ``mongo_history_utils``；保留这一层可避免上层工作流依赖
    MongoDB collection、字段转换和 ID 处理细节。
    """

    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        """按时间正序返回会话最近的消息。

        Args:
            session_id: 用于隔离聊天记录的会话标识。
            limit: 最多读取的消息数量。
        """
        return get_recent_messages(session_id, limit=limit)

    def save_message(
        self,
        *,
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        item_names: list[str] | None = None,
        image_urls: list[str] | None = None,
        message_id: str | None = None,
    ) -> str:
        """保存一条聊天消息并返回 MongoDB document ID。

        Optional list 参数由底层 helper 规范化为空列表，避免将 ``None`` 写入需要数组
        语义的字段。
        """
        return save_chat_message(
            session_id=session_id,
            role=role,
            text=text,
            rewritten_query=rewritten_query,
            item_names=item_names,
            image_urls=image_urls,
            message_id=message_id,
        )

    def clear_session(self, session_id: str) -> int:
        """删除指定会话的全部消息，并返回 MongoDB 删除数量。"""
        return clear_history(session_id)

    def update_item_names(self, ids: list[str], item_names: list[str]) -> int:
        """批量更新消息关联的主体名称，并返回实际更新数量。"""
        return update_message_item_names(ids, item_names)


# 模块级实例保持调用方式稳定，并为未来替换 Repository 实现保留注入边界。
history_repository = HistoryRepository()
