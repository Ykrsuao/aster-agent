# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for swappable vector backends (numpy / sqlite-vec / chroma).

The numpy backend always runs; the others ``importorskip`` their optional
dependency so the suite stays green whether or not they are installed.
"""

import shutil
import sys
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from aster_memory import (
    MemoryConfig,
    MemoryManager,
    MemoryToolsMixin,
    NumpyVectorIndex,
    VectorConfig,
    make_vector_index,
)
from aster_memory.embeddings import HashingEmbedder
from aster_memory.schema import Memory, MemoryType
from aster_memory.store import MemoryStore

from aster import Agent
from aster.unifiedllm import FakeLLMClient

DIM = 64
ALL_BACKENDS = ["numpy", "sqlite_vec", "chroma_embedded", "chroma_http"]


def _require(backend: str) -> None:
    if backend == "sqlite_vec":
        pytest.importorskip("sqlite_vec")
    if backend in ("chroma_embedded", "chroma_http"):
        pytest.importorskip("chromadb")
    if backend == "chroma_http":
        pytest.skip("chroma_http needs a running server")


def _store(backend: str) -> MemoryStore:
    _require(backend)
    return MemoryStore(":memory:", vector_config=VectorConfig(backend=backend), embedding_dim=DIM)


@pytest.fixture
def emb():
    return HashingEmbedder(dim=DIM)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_factory_returns_working_index(backend):
    store = _store(backend)
    emb = HashingEmbedder(dim=DIM)
    m = Memory(content="deploy ship release", type=MemoryType.INFO)
    store.add(m, emb.embed(m.embedding_text()))
    ranked = store.knn(emb.embed("deploy ship"), 1)
    assert ranked and ranked[0][0] == m.id
    assert 0.0 < ranked[0][1] <= 1.0  # cosine score
    store.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_ranks_relevant_first(backend, emb):
    store = _store(backend)
    target = Memory(content="deploy ship release production rollout", type=MemoryType.INFO)
    others = [
        Memory(content="the cat sat on the mat", type=MemoryType.INFO),
        Memory(content="kubernetes pod crashloop backoff", type=MemoryType.INFO),
    ]
    for m in [target, *others]:
        store.add(m, emb.embed(m.embedding_text()))
    ranked = store.knn(emb.embed("how do I deploy and ship a release"), 3)
    assert ranked[0][0] == target.id
    store.close()


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backend_archive_removes_from_index(backend, emb):
    store = _store(backend)
    m = Memory(content="ephemeral", type=MemoryType.INFO)
    store.add(m, emb.embed(m.embedding_text()))
    assert len(store._index) == 1
    store.archive(m.id)
    assert m.id not in [i for i, _ in store.knn(emb.embed("ephemeral"), 5)]
    store.close()


def test_all_available_backends_agree_on_ranking(emb):
    """The whole point of the protocol: identical data -> identical top-k order."""
    texts = [
        "deploy ship release production",
        "the cat sat on the mat",
        "kubernetes pod crashloop backoff",
        "rollback with make undeploy",
    ]
    query = "how to deploy and ship a release to production"

    rankings = {}
    for backend in ALL_BACKENDS:
        try:
            _require(backend)
        except Exception:
            continue
        store = _store(backend)
        ids = []
        for t in texts:
            m = Memory(content=t, type=MemoryType.INFO)
            store.add(m, emb.embed(m.embedding_text()))
            ids.append(m.id)
        order = [mid for mid, _ in store.knn(emb.embed(query), len(texts))]
        # normalise to positions within this store's own ids
        rankings[backend] = [ids.index(mid) for mid in order]
        store.close()

    assert "numpy" in rankings
    baseline = rankings["numpy"]
    for backend, order in rankings.items():
        assert order == baseline, f"{backend} ranking {order} != numpy {baseline}"


def test_factory_unknown_backend_raises():
    # VectorConfig.backend is a Literal, so build a stand-in to reach the factory's guard.
    import types

    with pytest.raises(ValueError, match="Unknown vector backend"):
        make_vector_index(types.SimpleNamespace(backend="bogus"))  # type: ignore[arg-type]


def test_factory_numpy_is_default():
    assert isinstance(make_vector_index(VectorConfig()), NumpyVectorIndex)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_manager_wires_backend_end_to_end(backend):
    """Flipping MemoryConfig.vector.backend changes the store's index, no code change."""
    _require(backend)

    class MemAgent(MemoryToolsMixin, Agent, llm=FakeLLMClient()):
        pass

    agent = MemAgent()
    mgr = MemoryManager.install(
        agent,
        config=MemoryConfig(enabled=True, path=":memory:", vector=VectorConfig(backend=backend)),
    )
    agent.remember("the deploy command is make ship", type="info")
    res = agent.recall("how do I deploy", k=1)
    assert res and "make ship" in res[0].content
    mgr.uninstall()


def test_numpy_dedup_on_write_still_works_per_backend():
    """sanity: dedup-on-write (which uses knn) holds for the numpy default."""
    store = _store("numpy")
    emb = HashingEmbedder(dim=DIM)
    a = Memory(content="identical fact", type=MemoryType.INFO)
    store.add(a, emb.embed(a.embedding_text()))
    hits = store.knn(emb.embed("identical fact"), 1)
    assert hits[0][1] > 0.99  # near-identical cosine -> dedup would trigger
    store.close()


def test_npindex_unit():
    idx = NumpyVectorIndex()
    idx.add("a", np.array([1.0, 0.0], dtype=np.float32))
    idx.add("b", np.array([0.0, 1.0], dtype=np.float32))
    assert idx.query(np.array([1.0, 0.0], dtype=np.float32), 1)[0][0] == "a"


@pytest.fixture
def fake_chroma(monkeypatch):
    class Collection:
        def __init__(self, name):
            self.name = name
            self.index = NumpyVectorIndex()

        def upsert(self, *, ids, embeddings):
            for id, vector in zip(ids, embeddings, strict=True):
                self.index.add(id, np.asarray(vector))

        def delete(self, *, ids):
            for id in ids:
                self.index.remove(id)

        def query(self, *, query_embeddings, n_results):
            ranked = self.index.query(np.asarray(query_embeddings[0]), n_results)
            return {
                "ids": [[id for id, _ in ranked]],
                "distances": [[1.0 - score for _, score in ranked]],
            }

        def count(self):
            return len(self.index)

    collections = {}
    client = Mock()

    def get_collection(name, metadata):
        return collections.setdefault(name, Collection(name))

    client.get_or_create_collection.side_effect = get_collection
    module = SimpleNamespace(
        HttpClient=Mock(return_value=client),
        PersistentClient=Mock(return_value=client),
        EphemeralClient=Mock(return_value=client),
    )
    monkeypatch.setitem(sys.modules, "chromadb", module)
    monkeypatch.setitem(sys.modules, "chromadb.config", SimpleNamespace(Settings=Mock()))
    return module


@pytest.fixture
def fake_sqlite_vec(monkeypatch):
    module = SimpleNamespace(load=Mock(), serialize_float32=Mock())
    monkeypatch.setitem(sys.modules, "sqlite_vec", module)
    return module


@pytest.mark.parametrize("backend", ["chroma_embedded", "chroma_http"])
def test_chroma_namespace_isolates_stores_and_reuses_the_same_store(fake_chroma, tmp_path, backend):
    config = VectorConfig(backend=backend, collection="shared")
    first = make_vector_index(
        config, dim=2, path=str(tmp_path / "memory.sqlite"), namespace="a" * 32
    )
    reopened = make_vector_index(
        config, dim=2, path=str(tmp_path / "memory.sqlite"), namespace="a" * 32
    )
    second = make_vector_index(config, dim=3, path=str(tmp_path / "memory.db"), namespace="b" * 32)
    assert first._collection is reopened._collection
    assert first._collection is not second._collection
    assert first._collection.name.startswith("shared-")
    assert second._collection.name.startswith("shared-")
    if backend == "chroma_embedded":
        paths = [call.kwargs["path"] for call in fake_chroma.PersistentClient.call_args_list]
        assert len(set(paths)) == 1  # The SQLite suffixes map to the same Chroma directory.


def test_chroma_without_namespace_keeps_persistent_names_and_isolates_ephemeral(fake_chroma):
    config = VectorConfig(backend="chroma_embedded", collection="shared")
    persistent = make_vector_index(config, dim=2, path="memory.sqlite")
    first = make_vector_index(config, dim=2)
    second = make_vector_index(config, dim=2)
    assert persistent._collection.name == "shared"
    assert first._collection is not second._collection


def test_chroma_bound_ephemeral_namespace_is_stable(fake_chroma):
    config = VectorConfig(backend="chroma_embedded", collection="shared")
    first = make_vector_index(config, dim=2, namespace="a" * 32)
    reopened = make_vector_index(config, dim=2, namespace="A" * 32)
    assert first._collection is reopened._collection


def test_sqlite_vec_namespace_isolates_tables(fake_sqlite_vec):
    conn = Mock()
    config = VectorConfig(backend="sqlite_vec")
    first = make_vector_index(config, dim=2, conn=conn, namespace="a" * 32)
    reopened = make_vector_index(config, dim=2, conn=conn, namespace="A" * 32)
    second = make_vector_index(config, dim=3, conn=conn, namespace="b" * 32)
    assert first._table == reopened._table == f"vec_memories_{'a' * 32}"
    assert second._table == f"vec_memories_{'b' * 32}"
    statements = [call.args[0] for call in conn.execute.call_args_list]
    assert first._table in statements[0]
    assert first._table in statements[1]
    assert second._table in statements[2]
    assert "embedding float[3]" in statements[2]


def test_sqlite_vec_without_namespace_keeps_table_name(fake_sqlite_vec):
    index = make_vector_index(VectorConfig(backend="sqlite_vec"), dim=2, conn=Mock())
    assert index._table == "vec_memories"


@pytest.mark.parametrize("backend", ["sqlite_vec", "chroma_embedded", "chroma_http"])
@pytest.mark.parametrize(
    "namespace", ["", "g" * 32, "a" * 31, "a" * 33, "a'; DROP TABLE memories; --"]
)
def test_persistent_backend_rejects_invalid_namespace_before_loading_dependencies(
    backend, namespace
):
    with pytest.raises(ValueError, match="namespace"):
        make_vector_index(VectorConfig(backend=backend), dim=2, conn=Mock(), namespace=namespace)


def test_numpy_ignores_namespace():
    assert isinstance(make_vector_index(VectorConfig(), namespace="a" * 32), NumpyVectorIndex)


@pytest.mark.parametrize("backend", ["chroma_embedded", "chroma_http"])
@pytest.mark.parametrize("copy_database", [False, True], ids=["fresh", "copied"])
def test_chroma_store_isolates_copied_or_fresh_databases(
    fake_chroma, tmp_path, backend, copy_database
):
    """Exercise the real store/adapter with an offline Chroma protocol double."""
    source_path = tmp_path / "memory.sqlite"
    other_path = tmp_path / "memory.db"
    config = VectorConfig(backend=backend, collection="shared")
    options = {
        "vector_config": config,
        "embedding_identity": "test:v1",
        "embedding_dim": 2,
    }
    with closing(MemoryStore(source_path, **options)) as initial:
        initial.add(Memory(id="original", content="original"), np.array([1.0, 0.0]))
        initial_collection = initial._index._collection.name
    if copy_database:
        shutil.copyfile(source_path, other_path)

    with (
        closing(MemoryStore(source_path, **options)) as source,
        closing(MemoryStore(other_path, **options)) as other,
    ):
        assert source._index._collection.name == initial_collection
        other.add(Memory(id="other-only", content="other only"), np.array([0.0, 1.0]))

        assert source.get("other-only") is None
        assert source.knn(np.array([0.0, 1.0]), 1) == [("original", 0.0)]
        assert other.knn(np.array([0.0, 1.0]), 1) == [("other-only", 1.0)]
        assert source._index._collection is not other._index._collection
        if backend == "chroma_embedded":
            paths = [call.kwargs["path"] for call in fake_chroma.PersistentClient.call_args_list]
            assert len(set(paths)) == 1


@pytest.mark.parametrize("backend", ["chroma_embedded", "chroma_http"])
@pytest.mark.parametrize("alias", ["relative", "windows_case"])
def test_chroma_namespace_normalizes_store_paths(
    fake_chroma, monkeypatch, tmp_path, backend, alias
):
    if alias == "windows_case" and sys.platform != "win32":
        pytest.skip("Windows paths are case insensitive")
    monkeypatch.chdir(tmp_path)
    path = str(tmp_path / "memory.sqlite")
    alias_path = "./memory.sqlite" if alias == "relative" else path.upper()
    config = VectorConfig(backend=backend, collection="shared")
    first = make_vector_index(config, dim=2, path=path, namespace="a" * 32)
    reopened = make_vector_index(config, dim=2, path=alias_path, namespace="a" * 32)
    assert first._collection is reopened._collection
