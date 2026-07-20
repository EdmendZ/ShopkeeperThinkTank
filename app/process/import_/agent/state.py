"""定义知识导入图的共享状态结构，并提供互不共享可变列表的默认状态工厂。"""

from typing import TypedDict
import copy


class ImportGraphState(TypedDict):
    """描述所有导入节点共同读取和修改的字典字段。

    ``TypedDict`` 只帮助编辑器和静态类型检查器理解字典键，并不会在运行时自动
    校验数据。实际对象仍是普通 ``dict``，可用 ``state['task_id']`` 严格取值，
    也可用 ``state.get('chunks')`` 在键缺失时取得 ``None``。
    """
    task_id: str

    # --- 流程控制标记 ---
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # --- 路径相关 ---
    local_dir: str  # PDF 解析结果及中间文件的输出目录。
    local_file_path: str  # 外部传入的原始文件路径，入口节点再判断是 MD 还是 PDF。
    file_title: str  # 去掉目录和扩展名后的文件标题。
    pdf_path: str  # 确认为 PDF 输入后保存的路径。
    md_path: str  # 直接输入或由 PDF 转换得到的 Markdown 路径。

    # --- 内容数据 ---
    md_content: str
    chunks: list
    item_name: str

    # --- 数据库相关 ---
    embeddings_content: list


graph_default_state: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "embeddings_content": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """深拷贝默认模板，并用关键字参数覆盖指定字段。

    ``**overrides`` 把任意关键字参数收集为字典，例如
    ``create_default_state(task_id='t1', local_file_path='a.pdf')``。
    """
    # deepcopy 会连内部的 chunks 等列表一起复制，避免多个任务共享同一个列表。
    state = copy.deepcopy(graph_default_state)
    # update 用 overrides 中同名的键替换默认值，也会接纳模板之外的新键。
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """返回默认状态的深拷贝，防止调用方修改全局模板。"""

    return copy.deepcopy(graph_default_state)
