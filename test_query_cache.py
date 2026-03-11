import sys
import types
import unittest
from unittest.mock import MagicMock

from query_cache import QueryCache


class _DummyResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class QueryCacheEmbeddingTests(unittest.TestCase):
    def _mock_engine_with_connection(self, conn):
        engine = MagicMock()
        ctx = MagicMock()
        ctx.__enter__.return_value = conn
        ctx.__exit__.return_value = False
        engine.connect.return_value = ctx
        return engine

    def _module_patches(self, embedding_fn):
        return {
            "vector_utils_v2": types.SimpleNamespace(get_embedding=embedding_fn),
            "sqlalchemy": types.SimpleNamespace(text=lambda sql: sql),
        }

    def test_successful_semantic_insert_serializes_embedding(self):
        conn = MagicMock()
        conn.execute.return_value = _DummyResult(None)
        engine = self._mock_engine_with_connection(conn)
        cache = QueryCache(vector_engine=engine, enabled=False)
        cache.vector_engine = engine

        with unittest.mock.patch.dict(
            sys.modules, self._module_patches(lambda _: [0.1, 0.2, 3])
        ):
            ok = cache._db_store(
                cache_key="k1",
                question_original="Q",
                question_normalized="q",
                sql="SELECT 1",
                schema_version="s1",
                rules_version="r1",
                dialect="postgresql",
            )

        self.assertTrue(ok)
        params = conn.execute.call_args.args[1]
        self.assertEqual(params["embedding"], "[0.1,0.2,3.0]")

    def test_successful_semantic_lookup_serializes_embedding_and_returns_result(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            _DummyResult(("SELECT 1", "easy", 10, 2, True, "k1", 0.99)),
            _DummyResult(None),
        ]
        engine = self._mock_engine_with_connection(conn)
        cache = QueryCache(
            vector_engine=engine,
            enabled=False,
            semantic_threshold=0.95,
        )
        cache.vector_engine = engine

        with unittest.mock.patch.dict(
            sys.modules, self._module_patches(lambda _: [1, 2, 3])
        ):
            result = cache._db_semantic_lookup("q", "s1", "r1", "postgresql")

        self.assertIsNotNone(result)
        self.assertEqual(result["sql"], "SELECT 1")
        first_call_params = conn.execute.call_args_list[0].args[1]
        self.assertEqual(first_call_params["embedding"], "[1.0,2.0,3.0]")

    def test_set_falls_back_when_embedding_generation_fails(self):
        conn = MagicMock()
        conn.execute.return_value = _DummyResult(None)
        engine = self._mock_engine_with_connection(conn)
        cache = QueryCache(vector_engine=engine, enabled=False)
        cache.vector_engine = engine

        def _raise(_):
            raise RuntimeError("embed failure")

        with unittest.mock.patch.dict(sys.modules, self._module_patches(_raise)):
            ok = cache._db_store(
                cache_key="k2",
                question_original="Q",
                question_normalized="q",
                sql="SELECT 2",
                schema_version="s1",
                rules_version="r1",
                dialect="postgresql",
            )

        self.assertTrue(ok)
        params = conn.execute.call_args.args[1]
        self.assertNotIn("embedding", params)


if __name__ == "__main__":
    unittest.main()
