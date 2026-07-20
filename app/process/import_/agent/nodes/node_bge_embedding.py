"""向量化节点适配器：记录任务进度，并把状态交给 Embedding 领域服务。"""

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.embedding_service import generate_chunk_embeddings

# 装饰器负责统一记录节点调用日志；函数体负责业务进度和状态传递。
@node_log("node_bge_embedding")
def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    """
    # 使用 [] 取 task_id：任务状态缺少该必需键时会立即抛出 KeyError。
    add_running_task(state["task_id"], "node_bge_embedding")
    state = generate_chunk_embeddings(state)
    # 只有 service 正常返回才会执行完成标记；中途异常会交给上层图处理。
    add_done_task(state["task_id"], "node_bge_embedding")
    return state
