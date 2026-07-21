"""
LangGraph 的文档切分节点。

Node 负责“流程调度”：登记节点开始和完成，并把共享 state 交给 Service。
真正的标题切分、长短块整理和 JSON 备份都在 split_service.py 中完成。
直接运行本文件时，还可以用本地 Markdown 手工观察“图片处理 -> 文档切分”的结果。
"""

import os

from app.shared.runtime.logger import node_log, logger
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.split_service import split_document

# @node_log 的作用和 @step_log 类似，但它专门用于 LangGraph 节点：
# 会从 state 中取得 task_id，并记录节点开始、结束、耗时和异常。
# 装饰器语法等价于 node_document_split = node_log(...)(node_document_split)。
@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点：文档切分（node_document_split）。

    这个 Node 自己不实现切分算法，只完成三件事：
    1. 把 node_document_split 登记为“正在运行”；
    2. 调用 split_document，让 Service 处理正文并写入 state["chunks"]；
    3. Service 成功返回后，把当前节点登记为“已完成”。

    Args:
        state: ImportGraphState 定义的导入流程共享字典。

    Returns:
        经过文档切分后的 state，供 LangGraph 的下一个节点继续使用。
    """
    # state["task_id"] 使用方括号严格取值：缺少 task_id 时会抛出 KeyError，
    # 因为任务进度必须绑定到一个明确的任务。
    add_running_task(state["task_id"], "node_document_split")

    # Service 会读取 md_content、md_path、file_title，并把 chunks 写回同一个 state。
    # 这里仍使用 state = ... 接收返回值，使 Node 的写法与其他 Service 调用保持一致。
    state = split_document(state)

    # 如果 Service 抛出异常，程序不会执行到这一行，因此失败节点不会被误记为完成。
    add_done_task(state["task_id"], "node_document_split")
    return state

# -------------------------- 本地手工演示入口 --------------------------
# Python 直接运行本文件时，特殊变量 __name__ 的值是 "__main__"；
# 被 main_graph.py 导入时则是完整模块名，下面的演示代码不会执行。
if __name__ == '__main__':
    # 这两个导入只供演示使用，放在 if 内可避免正常启动导入图时额外加载。
    from app.shared.utils.path_util import PROJECT_ROOT
    from app.process.import_.agent.nodes.node_md_img import node_md_img

    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")

    # ----------------------- Arrange：准备测试路径 -----------------------
    # os.path.join 会使用当前操作系统的路径分隔符拼接各部分，避免手写完整绝对路径。
    test_md_name = os.path.join(r"output\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # 本示例依赖已经存在的 Markdown 和 images 目录，不会自动生成测试文件。
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
        # ----------------------- Arrange：准备最小流程状态 -----------------------
        # 普通 dict 就是 LangGraph 在运行时传递的 state；这里手动准备两个节点会用到的字段。
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": "",
            "file_title": "hak180产品安全手册",
            "local_dir": os.path.join(PROJECT_ROOT, "output"),
        }

        # ----------------------- Act：按真实流程顺序执行 -----------------------
        # 先处理 Markdown 图片，得到增强后的 md_content；再执行文档切分。
        result_state = node_md_img(test_state)
        final_state = node_document_split(result_state)

        # ----------------------- Inspect：人工查看结果 -----------------------
        # get("chunks", []) 在字段缺失时返回空列表，避免打印阶段抛出 KeyError。
        final_chunks = final_state.get("chunks", [])
        # 这里只打印并记录数量，没有 assert，因此它是手工集成演示，不是自动化单元测试。
        print(final_chunks)
        logger.info(f"测试成功：最终生成{len(final_chunks)}个有效Chunk")
