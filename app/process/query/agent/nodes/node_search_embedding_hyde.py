"""LangGraph HyDE 检索 node：用 hypothetical answer 扩展本地召回。"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.hyde_search_service import search_embedding_hyde
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_search_embedding_hyde")
def node_search_embedding_hyde(state):
    """执行 HyDE 增强检索，产出 ``hyde_embedding_chunks`` partial state。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    chunks = search_embedding_hyde(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"hyde_embedding_chunks": chunks}
