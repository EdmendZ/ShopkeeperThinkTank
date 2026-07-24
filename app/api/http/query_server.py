"""查询 HTTP API：协调同步响应、后台 LangGraph 执行与 SSE 进度流。"""

from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schemas.query import QueryRequest, QueryResponse, AsyncQueryResponse
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_app as query_graph_app, query_app
from app.process.query.agent.state import create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

# 该 app 会由根应用挂载，也可直接运行以便独立调试查询服务。
app = FastAPI(
    title=settings.query_app_name,
    description="描述,进行rag查询的服务对象",
    version="0.2.0"
)

# 当前前端可能从独立开发端口访问，因此开放 CORS；生产部署应由网关收紧来源。
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_methods = ['*'],
    allow_headers = ['*']
)

@app.get("/html")
def query_html():
    """返回内置聊天页面；media type 根据文件扩展名推断。"""
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


def run_query_graph(query: str, session_id: str, is_stream: bool):
    """执行一次查询图，并把生命周期状态写入 session task store。

    流式模式下，task status 和异常还会通过 SSE queue 推送；函数本身不返回答案，
    最终结果由 answer node 写入 task store。
    """
    # session_id 可被客户端复用；启动新任务前必须清除同 ID 的旧进度和结果。
    clear_task(session_id)
    update_task_status(session_id, "processing", is_stream)

    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        update_task_status(session_id, "failed", is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})


@app.post("/query")
async def query(request: QueryRequest,background_tasks: BackgroundTasks):
    """提交查询，并根据 ``is_stream`` 选择立即响应或同步等待。

    流式请求先创建 session queue，再通过 FastAPI ``BackgroundTasks`` 执行查询图；
    客户端使用返回的 ``session_id`` 订阅 ``/stream/{session_id}``。非流式请求在当前
    调用中完成图执行，并从 task store 读取答案。
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    if is_stream:
        # queue 必须先于 background task 创建，避免图启动后的首个事件无处投递。
        create_sse_queue(session_id)
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        logger.info(f"query:{query}已经开启了异步和流式处理！！")
        return AsyncQueryResponse(
            session_id=session_id,
            message="本次查询处理中...."
        )
    else:
        run_query_graph(query, session_id, is_stream)
        # ``node_answer_output`` 负责把最终 answer 写入 task store。
        answer = get_task_result(session_id,"answer")
        logger.info(f"query:{query}开启同步处理！处理结果为：{answer}!")
        return QueryResponse(
            answer=answer,
            session_id=session_id,
            message="本次查询完毕!",
            done_list=get_done_task_list(session_id)
        )

@app.get("/stream/{session_id}")
async def stream_query_result(session_id: str, request: Request):
    """以 ``text/event-stream`` 持续输出指定 session 的进度和结果事件。"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/health")
async def health():
    """返回轻量健康状态，不探测模型或外部存储连接。"""
    logger.info("健康检查接口调用成功")
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    # 直接执行本模块时启动独立查询服务；被根应用导入时不会创建第二个 server。
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
