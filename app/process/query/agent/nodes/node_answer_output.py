"""LangGraph answer node：生成最终答案并更新 session 任务进度。"""

import sys

from app.shared.runtime.logger import node_log
from app.rag.query.answer_service import generate_answer
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_answer_output")
def node_answer_output(state):
    """生成或复用最终答案，返回更新后的完整 query state。"""
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    state = generate_answer(state)
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state["is_stream"])
    return state
