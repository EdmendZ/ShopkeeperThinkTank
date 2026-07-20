"""工作流编排层包。

用 LangGraph 的 State、Node 和 Graph 描述业务执行顺序；节点只负责流程衔接，
具体业务计算由 :mod:`app.rag` 下的服务完成。
"""
