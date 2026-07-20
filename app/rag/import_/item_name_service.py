"""导入流程的主体识别占位模块，预留名称抽取与主体索引接口。"""

from app.process.import_.agent.state import ImportGraphState


def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    """原样返回导入状态。

    函数名体现了未来计划，但当前函数体只有 ``return state``，因此不会调用 LLM、
    修改切块或写入主体名称索引。
    """
    return state
