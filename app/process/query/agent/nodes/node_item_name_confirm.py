"""LangGraph 主体确认 node：改写 query、识别主体并决定是否需要澄清。"""

import json
import sys

from app.shared.runtime.logger import node_log
from app.rag.query.item_name_confirm_service import confirm_item_name
from app.shared.utils.task_utils import add_done_task, add_running_task

@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    """运行主体确认 service，并返回更新后的完整 query state。

    除 ``item_names`` 外，service 还会更新 ``rewritten_query``、``history``，并可能
    写入用于提前结束检索的 ``answer``。
    """
    # 先登记节点开始，前端进度区可以立即感知"主体确认"已启动。
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    state = confirm_item_name(state)

    # 识别完成后写入完成列表，方便前端展示当前节点已结束。
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])

    return state
