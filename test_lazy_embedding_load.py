"""Sanity check for lazy embedding model initialization in vector_utils_v2."""

import importlib
import sys
import types


class FakeSentenceTransformer:
    init_count = 0

    def __init__(self, model_name: str):
        type(self).init_count += 1
        self.model_name = model_name

    def encode(self, text: str, convert_to_tensor: bool = False):
        return FakeVector([0.1, 0.2, 0.3])


class FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


def _install_fake_dependencies() -> None:
    fake_st_module = types.ModuleType("sentence_transformers")
    fake_st_module.SentenceTransformer = FakeSentenceTransformer
    sys.modules["sentence_transformers"] = fake_st_module

    fake_sqlalchemy = types.ModuleType("sqlalchemy")
    fake_sqlalchemy.text = lambda value: value
    sys.modules["sqlalchemy"] = fake_sqlalchemy

    sys.modules["numpy"] = types.ModuleType("numpy")


def run_check() -> None:
    sys.modules.pop("vector_utils_v2", None)
    _install_fake_dependencies()

    module = importlib.import_module("vector_utils_v2")
    assert FakeSentenceTransformer.init_count == 0, (
        "SentenceTransformer should not be instantiated at import time"
    )

    embedding = module.get_embedding("hello")
    assert embedding == [0.1, 0.2, 0.3]
    assert FakeSentenceTransformer.init_count == 1, (
        "SentenceTransformer should be instantiated on first embedding request"
    )

    module.get_embedding("world")
    assert FakeSentenceTransformer.init_count == 1, "Model should be cached after first use"


if __name__ == "__main__":
    run_check()
    print("✅ Lazy embedding model check passed")
