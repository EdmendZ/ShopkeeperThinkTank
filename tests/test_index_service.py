"""Milvus chunk collection schema/index configuration regression tests。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.import_ import index_service


class RecordingIndexParams:
    """记录 ``add_index`` 调用参数的 minimal fake。"""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def add_index(self, **kwargs) -> None:
        """按调用顺序保存 index definition，供断言检索。"""
        self.entries.append(kwargs)


class PrepareChunksCollectionIndexTests(unittest.TestCase):
    """验证首次建 collection 时生成的 dense/sparse index contract。"""

    def test_dense_and_sparse_indexes_match_project3_build_intent(self) -> None:
        """防止 COSINE/HNSW 与 sparse IP index 配置在同步时发生漂移。"""
        fake_client = MagicMock()
        fake_client.has_collection.return_value = False
        fake_schema = MagicMock()
        fake_client.create_schema.return_value = fake_schema
        index_params = RecordingIndexParams()
        fake_client.prepare_index_params.return_value = index_params

        with patch.object(index_service.milvus_gateway, "client", return_value=fake_client):
            index_service.prepare_chunks_collection()

        dense = next(
            entry for entry in index_params.entries
            if entry["field_name"] == "dense_vector"
        )
        sparse = next(
            entry for entry in index_params.entries
            if entry["field_name"] == "sparse_vector"
        )

        self.assertEqual(
            dense,
            {
                "field_name": "dense_vector",
                "index_type": "HNSW",
                "index_name": "dense_vector_index",
                "metric_type": "COSINE",
                "params": {"M": 64, "efConstruction": 100},
            },
        )
        self.assertEqual(
            sparse,
            {
                "field_name": "sparse_vector",
                "index_type": "SPARSE_INVERTED_INDEX",
                "index_name": "sparse_vector_index",
                "metric_type": "IP",
                "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
            },
        )
        fake_client.create_collection.assert_called_once_with(
            collection_name=index_service.milvus_gateway.chunks_collection,
            schema=fake_schema,
            index_params=index_params,
        )


if __name__ == "__main__":
    unittest.main()
