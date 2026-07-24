"""构建查询 LangGraph：主体确认后并行检索，再执行融合、重排与答案生成。"""

from langgraph.graph import StateGraph, END

from app.process.query.agent.nodes.node_answer_output import node_answer_output
from app.process.query.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.process.query.agent.nodes.node_rerank import node_rerank
from app.process.query.agent.nodes.node_rrf import node_rrf
from app.process.query.agent.nodes.node_search_embedding import node_search_embedding
from app.process.query.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.process.query.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger

# StateGraph 以 QueryGraphState 作为跨节点 data contract。
query_graph = StateGraph(QueryGraphState)

query_graph.add_node("node_item_name_confirm", node_item_name_confirm)
query_graph.add_node("node_search_embedding", node_search_embedding)
query_graph.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
query_graph.add_node("node_web_search_mcp", node_web_search_mcp)
query_graph.add_node("node_rrf", node_rrf)
query_graph.add_node("node_rerank", node_rerank)
query_graph.add_node("node_answer_output", node_answer_output)

query_graph.set_entry_point("node_item_name_confirm")


def node_item_name_confirm_after_router(state: QueryGraphState):
    """根据主体确认结果选择短路回答或三路并行检索。

    ``answer`` 非空表示上游已经生成澄清/兜底文本，无需继续检索；否则同时启动普通
    Embedding、HyDE 和 web search 分支。
    """
    if state['answer']:
        logger.warning(f"node_item_name_confirm_无法继续向后执行: {state['answer']}")
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde", "node_web_search_mcp"

query_graph.add_conditional_edges(
    "node_item_name_confirm",
    node_item_name_confirm_after_router,
    {
        "node_answer_output": "node_answer_output",
        "node_search_embedding": "node_search_embedding",
        "node_search_embedding_hyde": "node_search_embedding_hyde",
        "node_web_search_mcp": "node_web_search_mcp"
    }
)

# 两路本地检索先做 RRF；web documents 在 rerank service 中与本地结果合并。
query_graph.add_edge("node_search_embedding", "node_rrf")
query_graph.add_edge("node_search_embedding_hyde", "node_rrf")
query_graph.add_edge("node_web_search_mcp", "node_rrf")
query_graph.add_edge("node_rrf", "node_rerank")
query_graph.add_edge("node_rerank", "node_answer_output")
query_graph.add_edge("node_answer_output", END)

# 模块导入时编译 graph，HTTP 层复用同一 runnable。
query_app = query_graph.compile()
