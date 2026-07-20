"""导入流程的向量入库占位模块，预留 Milvus 集合维护和写入接口。"""

from app.process.import_.agent.state import ImportGraphState


def index_chunks(state: ImportGraphState) -> ImportGraphState:
    """原样返回导入状态。

    当前版本只是保持 LangGraph 节点接口完整，尚未创建集合、删除旧数据或插入
    ``chunks``；学习时要区分“注释描述的规划”与函数真实执行的语句。
    """
    return state
