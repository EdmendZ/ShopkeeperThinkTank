"""导入流程的向量化服务占位模块，预留为切块补充稠密、稀疏向量的接口。"""

from app.process.import_.agent.state import ImportGraphState


def generate_chunk_embeddings(state: ImportGraphState) -> ImportGraphState:
    """原样返回导入状态。

    这是为后续向量化步骤预留的服务接口；当前项目尚未在这里调用模型，也不会
    给 ``chunks`` 增加 ``dense_vector`` 或 ``sparse_vector`` 字段。
    """
    return state
