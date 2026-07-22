"""Milvus 向量库的门面（Gateway）模块。

本文件把“读取集合配置”“取得客户端”“构造混合检索请求”和“执行检索”集中到一个
对象中。业务层只依赖本网关，因此底层客户端工具或配置位置变化时，修改范围更小。
"""
from typing import Any

from app.shared.clients.milvus_utils import (
    create_hybrid_search_requests,
    get_milvus_client,
    hybrid_search,
)
from app.infra.config import infra_config


class MilvusGateway:
    """为业务层提供 Milvus 相关能力的轻量入口，不在这里保存业务数据。"""

    @property
    def chunks_collection(self) -> str:
        """
        获取文档切块集合名称。

        Returns:
            str: Milvus 中存放知识切块的集合名。
        """
        # 配置对象集中保存环境差异，调用方无需自己拼接集合名。
        return infra_config.milvus.chunks_collection

    @property
    def item_name_collection(self) -> str:
        """
        获取主体名称集合名称。

        Returns:
            str: Milvus 中存放主体名称向量的集合名。
        """
        return infra_config.milvus.item_name_collection

    def client(self):
        """
        获取 Milvus 客户端实例。

        Returns:
            Any: 底层 Milvus 客户端对象。
        """
        # 底层函数会复用已创建的客户端，避免每次检索都重新建立连接。
        return get_milvus_client()

    def create_requests(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        *,
        expr: str = None,
        limit: int = 5,
    ):
        """
        创建 Milvus 混合检索请求对象。

        Args:
            dense_vector: 稠密向量表示。
            sparse_vector: 稀疏向量表示。
            expr: 可选过滤表达式，用于限定检索范围。
            limit: 单路检索返回条数上限。

        Returns:
            Any: 底层 Milvus 混合检索请求列表。
        """
        # * 之后的 expr、limit 必须写成“参数名=值”，调用时不容易把过滤条件和数量写反。
        # 稠密向量负责语义相近召回，稀疏向量负责关键词匹配；底层会把它们变成两路请求。
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
        )

    def hybrid_search(
        self,
        *,
        collection_name: str,
        reqs: list[Any],
        ranker_weights: tuple[float, float] = (0.5, 0.5),
        norm_score: bool = False,
        limit: int = 5,
        output_fields: list[str] | None = None,
        search_params: dict | None = None,
    ):
        """
        执行 Milvus 混合检索。

        Args:
            collection_name: 目标集合名称。
            reqs: 检索请求列表。
            ranker_weights: 稠密与稀疏路召回结果的融合权重。
            norm_score: 是否对分数做归一化。
            limit: 最终返回条数上限。
            output_fields: 需要返回的字段列表。
            search_params: 额外检索参数。

        Returns:
            Any: Milvus 返回的原始检索结果。
        """
        # 网关只转交参数，不改变底层返回结果；这样调用方仍可获得 Milvus 的原始命中信息。
        # ranker_weights 的两个值依次对应稠密、稀疏两路结果，通常应保持为非负数。
        return hybrid_search(
            client=self.client(),
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params,
        )


# 模块级单例：业务代码导入并复用同一个无状态网关对象即可。
milvus_gateway = MilvusGateway()
