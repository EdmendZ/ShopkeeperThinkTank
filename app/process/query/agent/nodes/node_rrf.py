"""LangGraph RRF node：融合两路本地检索排名并返回 partial state。"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.rrf_service import fuse_by_rrf
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_rrf")
def node_rrf(state):
    """融合 Embedding 与 HyDE 排名，产出 ``rrf_chunks`` partial state。

    Web documents 不参与此处 RRF，而是在后续 rerank service 中合并。
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    chunks = fuse_by_rrf(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"rrf_chunks": chunks}
