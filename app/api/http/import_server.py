"""知识导入服务的 HTTP 入口。

客户端先调用 ``POST /upload`` 上传文件，服务端立即返回每个文件的 ``task_id``；随后客户端
可调用 ``GET /status/{task_id}`` 查询导入图进度。这里使用 FastAPI 的 BackgroundTasks，
不是可跨进程持久化的任务队列：应用重启后，内存中的任务状态会丢失。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.import_ import ImportStatusResponse, UploadResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state
from app.infra.config import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status, add_done_task, add_running_task,
)

app = FastAPI(
    title=settings.import_app_name,
    description="企业化 RAG 导入服务，负责文件上传、导入执行与状态查询。",
    version="0.2.0",
)

# CORS 决定哪些浏览器网页可以调用本服务。配置为空时退回 ``["*"]``，表示不限制来源。
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/html")
def import_html():
    """返回项目内置的导入演示页面文件。"""

    # ``Path / "子路径"`` 会按操作系统规则拼接路径，避免手写反斜杠或斜杠。
    html_path = PROJECT_ROOT / "app" / "process" / "import_" / "page" / "import.html"
    # guess_type 根据扩展名推测 MIME 类型，让浏览器按 HTML 页面而不是普通附件处理。
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])

def run_graph_task(task_id: str, local_dir: str, local_file_path: str):
    """在响应返回后执行导入图，并把进度写入内存任务表。

    ``kb_import_app.stream`` 会在每个节点完成时产出一个事件字典，字典的键是节点名称。
    本函数记录这些节点名称，供 ``/status`` 接口轮询。异常不会重新抛给已经返回的上传
    请求，而是记录日志并将任务状态改为 ``failed``。

    Args:
        task_id: 当前文件的唯一任务 ID。
        local_dir: 当前任务保存源文件和中间产物的目录。
        local_file_path: 已保存到本地的上传文件绝对路径。
    """
    try:
        # 任务状态和节点列表都保存在当前进程内存中；状态接口会从同一处读取它们。
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行LangGraph全流程，本地文件路径：{local_file_path}")

        # get_default_state 返回新的深拷贝，避免不同上传任务共用 chunks 等可变列表。
        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path

        # stream 逐节点返回事件，避免等所有节点结束后才知道处理进度。
        for event in kb_import_app.stream(init_state):
            for node_name, node_result in event.items():
                # node_result 是该节点产出的状态片段；此接口只需要节点名，所以不读取它。
                logger.info(f"[{task_id}] LangGraph节点执行完成：{node_name}")
                # add_done_task 会避免重复记录，并把同名节点从“运行中”列表移除。
                add_done_task(task_id, node_name)

        # 只有图的所有事件正常迭代完成，任务才标记为 completed。
        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] LangGraph全流程执行完毕，任务完成")

    except Exception as e:
        # 后台任务的异常无法直接返回给已结束的 HTTP 请求，需由状态接口和日志对外体现。
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)

@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(
        ...,
        json_schema_extra={
        "items": {
            "type": "string",
            "format": "binary",
        }
    },
    ),
):
    """保存上传文件，为每个文件创建任务，并登记后台导入流程。

    ``files`` 是 FastAPI 从 multipart/form-data 表单注入的文件列表。接口不会等待导入完成；
    它返回任务 ID 后，由客户端通过状态接口继续获取进度。
    """

    # 以日期分组中间文件，方便人工清理；每个文件稍后还会拥有独立的 UUID 子目录。
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str

    task_ids = []

    # 一个 HTTP 请求可带多个文件，因此每次循环都要单独创建 task_id 和目录。
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)

        # 任务工具会把内部节点名转换为中文，供状态接口直接展示。
        add_running_task(task_id, "upload_file")

        # UUID 使同一天的同名文件也不会覆盖彼此。
        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)

        # UploadFile.file 是可读取的二进制文件对象；以 wb 打开目标文件用于写入字节。
        local_file_abs_path: Path = task_local_dir / file.filename
        with local_file_abs_path.open("wb") as file_buffer:
            # copyfileobj 会分块读取并写入，不会一次把整个文件读入内存。
            # 它仍是同步文件 I/O；超大文件场景应评估异步存储或专门的上传服务。
            shutil.copyfileobj(file.file, file_buffer)

        # add_done_task 同时会把 upload_file 从运行中列表移除。
        add_done_task(task_id, "upload_file")

        # FastAPI 在响应发送后执行该任务；这里只登记函数和参数，不会立刻等待导入结束。
        background_tasks.add_task(
            run_graph_task,
            task_id,
            str(task_local_dir),
            str(local_file_abs_path)
        )

    return UploadResponse(
        code=200,
        message=f"Files uploaded successfully, total: {len(files)}",
        task_ids=task_ids
    )

@app.get("/status/{task_id}", summary="任务状态查询", response_model=ImportStatusResponse)
async def get_task_progress(task_id: str):
    """返回一个任务的当前状态和节点进度。

    任务追踪数据保存在当前进程内存中。未知任务 ID 会得到任务工具的默认值，而不会在这里
    自动查询数据库或恢复历史任务。
    """

    # 三个读取函数返回的节点名已由任务工具转换为适合前端展示的中文名称。
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)

    logger.info(f"[{task_id}] 任务状态查询，当前状态：{status}，已完成节点：{done_list}")

    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list
    )

if __name__ == "__main__":
    # 直接运行此文件时启动本地开发服务；生产环境通常由 ASGI 服务器加载 ``app`` 对象。
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
