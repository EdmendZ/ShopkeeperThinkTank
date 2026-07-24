"""导入向量化 service：分批生成 dense/sparse vectors 并回写文档 chunks。"""
from app.shared.runtime.logger import logger, step_log
from app.infra.llm import llm_provider
from app.rag.import_.config import EMBEDDING_BATCH_SIZE


@step_log("generate_chunk_embeddings")
def generate_chunk_embeddings(state: dict) -> dict:
    """
    向量化服务总入口
    功能：校验切块数据 → 批量生成稠密/稀疏向量 → 回写到 state
    输出：更新后的 state，包含带有 dense_vector 和 sparse_vector 的 chunks
    """
    # 先确认 chunks 存在，再批量写回 dense/sparse 向量字段
    state["chunks"] = embed_chunks(require_chunks(state))
    return state

@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """
    校验导入状态中是否已经生成切块结果
    功能：确保后续流程有有效的输入数据，缺失时抛出异常
    :param state: LangGraph 流程状态字典
    :return: 已通过校验的切块列表
    """
    # 从 state 中获取核心数据
    chunks = state.get("chunks", [])

    # 空输入没有可入库内容，直接终止导入，避免下游创建空集合记录。
    if not chunks:
        logger.error("chunks为空,无法继续业务处理!")
        raise ValueError("chunks为空,无法继续业务处理!")

    # 返回校验后的数据
    return chunks

@step_log("embed_chunks")
def embed_chunks(chunks: list[dict], *, step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """
    批量为文本切片生成稠密和稀疏向量
    功能：分批拼接文本 → 调用 Embedding 模型 → 绑定向量字段 → 异常隔离
    :param chunks: 文档切片列表，每个元素包含 item_name、content 等字段
    :param step: 批次大小，默认由配置 EMBEDDING_BATCH_SIZE 控制
    :return: 带向量字段的切片列表
    """
    chunks_vector: list[dict] = []
    total = len(chunks)

    # 分批限制单次 Embedding 请求的显存和 API payload。
    for index in range(0, total, step):
        try:
            step_chunks = chunks[index:index + step]

            # 主体名作为显式前缀，为 query 阶段的主体过滤保留一致语义。
            vector_str_list = []
            for item in step_chunks:
                item_name = item.get("item_name")
                content = item.get("content", "")
                # 有主体名称则拼接，无则直接使用内容
                vector_str_list.append(f"主体:{item_name},内容:{content}" if item_name else content)

            # Provider 返回与输入顺序一致的 dense/sparse vector arrays。
            result = llm_provider.embed_documents(vector_str_list)

            # 复制 chunk，避免把向量字段写入上游仍可能复用的对象。
            for i, chunk in enumerate(step_chunks):
                chunk_new = chunk.copy()
                chunk_new["dense_vector"] = result["dense"][i]  # 绑定稠密向量
                chunk_new["sparse_vector"] = result["sparse"][i]  # 绑定稀疏向量
                chunks_vector.append(chunk_new)

        except Exception as exc:
            # 单批失败时保留其他批次结果；warning 包含区间便于后续补偿。
            # 捕获异常，记录警告信息并跳过当前批次
            logger.warning(f"index={index}步骤,发生错误,跳过,继续生成向量!!,错误信息:{str(exc)}")
            # 跳过当前批次，继续处理下一批次，保证整体流程不中断
            continue

    # 返回所有带向量的切片列表
    return chunks_vector
