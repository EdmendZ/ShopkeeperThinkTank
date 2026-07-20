"""组装知识导入 LangGraph，定义文件类型分支、节点执行顺序并编译应用。"""

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus

# 导入模块时加载 .env；后续配置对象和节点即可从环境变量读取连接信息。
load_dotenv()

# StateGraph 接收状态类型，用于约定所有节点之间传递的共享字典结构。
workflow = StateGraph(ImportGraphState)

# add_node 把“图中的节点名”绑定到实际执行的 Python 函数。
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_item_name_recognition", node_item_name_recognition)
workflow.add_node("node_bge_embedding", node_bge_embedding)
workflow.add_node("node_import_milvus", node_import_milvus)

# 每次 invoke/stream 都从入口节点开始。
workflow.set_entry_point("node_entry")

def after_entry_node(state: ImportGraphState):
    """
    入口节点后的路由函数：
    - Markdown 文件：直接进入图片处理节点
    - PDF 文件：先进入 PDF 转 Markdown 节点
    - 其他类型：直接结束
    """
    if state["is_md_read_enabled"]:
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        return "node_pdf_to_md"
    else:
        return END

workflow.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        END: END,
    },
)

# 以下普通边表示前一节点完成后无条件执行后一节点。
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_item_name_recognition")
workflow.add_edge("node_item_name_recognition", "node_bge_embedding")
workflow.add_edge("node_bge_embedding", "node_import_milvus")
workflow.add_edge("node_import_milvus", END)

# compile 把“节点 + 边”的声明转换为可 invoke 或 stream 的可执行图。
kb_import_app = workflow.compile()

# 该语句位于模块顶层，所以导入 main_graph 时也会打印一次 ASCII 流程图。
kb_import_app.get_graph().print_ascii()
# 如需在 Notebook 中查看图片，可使用 LangGraph 的 Mermaid PNG 绘图接口；当前代码
# 只保留轻量的终端 ASCII 输出。
