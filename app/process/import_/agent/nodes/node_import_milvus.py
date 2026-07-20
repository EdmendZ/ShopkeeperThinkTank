"""Milvus 入库节点适配器：记录进度，并委托入库服务处理切块。"""

from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.index_service import index_chunks

@node_log("node_import_milvus")
def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    """
    # 本节点不直接操作数据库，保持图编排层与基础设施细节解耦。
    add_running_task(state["task_id"], "node_import_milvus")
    state = index_chunks(state)
    # index_chunks 正常返回后才把该步骤从 running 移入 done。
    add_done_task(state["task_id"], "node_import_milvus")
    return state
