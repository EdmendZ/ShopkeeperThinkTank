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

if __name__ == '__main__':
    # --- 本地集成测试 ---
    # 本段会连接 .env 中配置的真实 Milvus，并可能删除同名 item_name 的旧记录；
    # 因此它不是隔离外部依赖的“单元测试”，不要在生产集合中随意运行。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 1024 必须与 MILVUS_VECTOR_DIM 一致；[数值] * dim 用重复值构造指定长度的假向量。
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "item_name":"测试项目_Milvus",
        "file_title": "test.pdf",
        "embeddings_content": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 同名记录会先被删除，再插入本次数据。
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 稠密向量的长度必须等于集合 schema 的维度。
                "sparse_vector": {1: 0.5, 10: 0.8}  # 键是词元编号，值是对应的稀疏权重。
            }
,
            {
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 与第一条相同，便于验证一次批量写入。
                "parent_title": "test.pdf2",
                "part": 1,
                "file_title": "test.pdf",
                "dense_vector": [0.2] * dim,  # 用不同数值区分两条测试记录。
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟由 BGE-M3 生成的稀疏表示。
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 节点会先登记任务进度，再委托 index_chunks 处理集合和数据。
        result_state = node_import_milvus(test_state)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
