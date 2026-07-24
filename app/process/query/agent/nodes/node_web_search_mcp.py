"""LangGraph web search node：通过 MCP 补充外部文档。"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.web_search_service import search_by_web
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_web_search_mcp")
def node_web_search_mcp(state):
    """执行外网检索，产出 ``web_search_docs`` partial state。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    pages = search_by_web(state)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {"web_search_docs": pages}
