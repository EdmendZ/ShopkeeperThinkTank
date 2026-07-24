"""LangGraph rerank node：为 RRF 与 web 候选精排并返回 partial state。"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rerank_service import rerank_documents
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rerank")
def node_rerank(state):
    """使用 Cross-Encoder 重排候选，产出 ``reranked_docs`` partial state。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    docs = rerank_documents(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"reranked_docs": docs}
