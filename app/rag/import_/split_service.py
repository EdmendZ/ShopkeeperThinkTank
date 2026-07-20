"""导入流程的文档切分占位模块，预留从 Markdown 生成知识切块的接口。"""

from app.process.import_.agent.state import ImportGraphState


def split_document(state: ImportGraphState) -> ImportGraphState:
    """原样返回导入状态。

    当前实现尚未读取 Markdown、执行标题切分或回写 ``chunks``，仅为导入图保留
    类型一致的服务边界。
    """
    return state
