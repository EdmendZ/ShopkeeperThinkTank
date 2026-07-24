"""把已向量化的文档切块写入 Milvus。

本模块负责集合的首次创建、同一主体的旧数据清理和新数据批量写入。调用前的每个
Chunk 应已包含内容、标题、主体名称以及稠密/稀疏向量；向量生成并不在本模块中完成。
"""

from pymilvus import DataType

from app.shared.runtime.logger import logger, step_log
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.rag.import_.config import (
    MILVUS_CHUNK_CONTENT_MAX_LENGTH,
    MILVUS_DEFAULT_VARCHAR_MAX_LENGTH,
    MILVUS_VECTOR_DIM,
)


@step_log("index_chunks")
def index_chunks(state: dict) -> dict:
    """按“校验 → 建集合 → 删旧数据 → 插新数据”的顺序完成一次入库。

    返回的仍是原 state 字典。集合使用 Milvus 自动主键，当前实现只记录服务端返回的
    ID 日志，不会把 ``chunk_id`` 回填到每个 Chunk 字典。
    """
    # 先校验切片存在，避免空列表仍触发集合创建或删除旧数据。
    chunks = require_chunks(state)

    # 集合不存在时先自动创建，保证第一次导入不需要人工初始化数据库。
    prepare_chunks_collection()

    # item_name 是本次导入的“业务主键”，用于判断哪些旧记录需要被覆盖。
    item_name = state.get("item_name", "")

    # 同一主体重复导入时先删旧数据，保持当前导入结果覆盖旧版本。
    # item_name 为空时跳过删除，重复导入可能累积多份记录。
    if item_name:
        remove_old_chunks(item_name)

    # 一次传入列表可减少客户端和 Milvus 之间的网络往返。
    insert_chunks(chunks)

    return state


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    """读取并校验 state 中的 ``chunks`` 列表。

    ``dict.get`` 让字段缺失时得到空列表；随后统一抛出 ValueError，使后续函数不用
    反复处理“没有任何切块”的分支。
    """
    # 从 state 中获取核心数据；这里不校验每个 Chunk 的字段完整性。
    chunks = state.get("chunks", [])

    # 空列表不得触发 collection 创建或同主体旧数据删除。
    if not chunks:
        logger.error("chunks为空,无法继续业务!!")
        raise ValueError("chunks为空,无法继续业务!!")

    # 返回校验后的数据
    return chunks


@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    """确保文档切块集合及其向量索引已经存在。

    集合已存在时直接返回，不会根据后续代码自动迁移 schema；调整字段或索引后需单独
    设计迁移流程。
    """
    # 获取 Milvus 客户端
    milvus_client = milvus_gateway.client()

    # 获取集合名称（从配置中读取）
    collection_name = milvus_gateway.chunks_collection

    # 如果集合已存在，直接返回；避免重复 create_collection 触发错误。
    if milvus_client.has_collection(collection_name=collection_name):
        return

    # Schema 只在 collection 首次创建时生效，后续字段变更需要显式迁移。
    # auto_id=True 让 Milvus 生成主键；enable_dynamic_field=True 允许写入未声明的额外字段。
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 主键由服务端生成，所以待写入的 Chunk 不应自己填写 chunk_id。
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)

    # 添加文件标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加主体名称字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加切片标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加父标题字段：VARCHAR 类型，最大长度 512
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=MILVUS_DEFAULT_VARCHAR_MAX_LENGTH)

    # 添加切片序号字段：INT8 类型
    schema.add_field(field_name="part", datatype=DataType.INT8)

    # 内容字段长度来自项目配置；超过该长度的 Chunk 会在插入阶段被 Milvus 拒绝。
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=MILVUS_CHUNK_CONTENT_MAX_LENGTH)

    # 添加稠密向量字段：FLOAT_VECTOR 类型，维度 1024
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=MILVUS_VECTOR_DIM)

    # 添加稀疏向量字段：SPARSE_FLOAT_VECTOR 类型
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    # dense 与 sparse 字段使用各自的检索 metric 和 index type。
    # 准备索引参数
    index_params = milvus_client.prepare_index_params()

    # 稠密向量采用 HNSW 图索引；COSINE 与 BGE-M3 文本向量查询参数保持一致。
    # M 控制每个节点的最大连接数，efConstruction 控制建图时的候选范围。
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100},
    )

    # 为稀疏向量创建索引：使用 SPARSE_INVERTED_INDEX，算法为 DAAT_MAXSCORE
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    # create_collection 会同时保存字段定义和上面准备好的两个索引。
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)

@step_log("remove_old_chunks")
def remove_old_chunks(item_name: str) -> None:
    """删除某个主体名称对应的已有切块，实现“重复导入覆盖旧版本”。"""
    # filter 是 Milvus 的标量过滤表达式。此处把 item_name 作为字符串字面量拼入表达式，
    # 因而调用方应保证名称不含单引号，否则需要先进行转义处理。
    milvus_gateway.client().delete(
        collection_name=milvus_gateway.chunks_collection,
        filter=f"item_name=='{item_name}'",
    )


@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]) -> None:
    """把带向量的 Chunk 列表批量写入切块集合。"""
    # Milvus 返回插入数量和自动生成的主键列表；本函数只记录日志，不修改原 chunks。
    result = milvus_gateway.client().insert(
        collection_name=milvus_gateway.chunks_collection,
        data=chunks,
    )

    # get 给出默认值，兼容结果字典缺少某个字段的情况。
    logger.info(f"插入数据成功! 总条数:{result.get('insert_count', 0)}")
    logger.info(f"插入数据主键回显:{result.get('ids', [])}")
