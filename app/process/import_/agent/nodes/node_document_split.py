"""文档切分节点适配器：更新任务进度，并把状态交给切分服务。"""

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.split_service import split_document

# @node_log(...) 等价于先用装饰函数包装 node_document_split，再把包装结果绑定回原名称。
@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    """
    add_running_task(state["task_id"], "node_document_split")
    state = split_document(state)
    # service 抛异常时不会走到这里，因此失败步骤不会被误记为已完成。
    add_done_task(state["task_id"], "node_document_split")
    return state
