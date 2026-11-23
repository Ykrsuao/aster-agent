# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding contracts reject incompatible vectors without losing canonical data."""

import json
import sqlite3
from contextlib import closing

import numpy as np
import pytest
from aster_memory.config import VectorConfig
from aster_memory.embeddings import EmbeddingMismatchError, HashingEmbedder
from aster_memory.schema import EdgeType, Memory, MemoryType
from aster_memory.store import MemoryStore


class _Embedder(HashingEmbedder):
    def __init__(self, identity, dim):
        super().__init__(dim=dim)
        self._test_identity = identity
        self.seen = []

    @property
    def identity(self):
        return self._test_identity

    def embed_batch(self, texts):
        self.seen.extend(texts)
        return super().embed_batch(texts)


@pytest.fixture
def stores():
    opened = []

    def open_store(path=":memory:", **kwargs):
        store = MemoryStore(path, **kwargs)
        opened.append(store)
        return store

    yield open_store
    for store in reversed(opened):
        store.close()


def _records():
    active = Memory(
        id="active",
        title="Original title",
        content="Keep the complete fact, including 中文 and a second line.\nSecond line.",
        owner="owner-a",
        importance=8.0,
        salience=0.7,
        confidence=0.9,
        strength=3,
        created_at=10.0,
        last_accessed_at=20.0,
        tags=["release", "retained"],
        entities=["aster"],
        related_files=["src/example.py"],
        source_task_ref="task-a",
        injected_count=2,
    )
    archived = Memory(
        id="archived",
        type=MemoryType.TODO,
        content="An archived completed task must also be re-embedded.",
        status="done",
        archived=True,
        owner="owner-b",
        created_at=11.0,
        last_accessed_at=21.0,
    )
    without_vector = Memory(
        id="without-vector",
        content="A text-only record must not disappear during a rebuild.",
        created_at=12.0,
        last_accessed_at=22.0,
    )
    active.add_edge(archived.id, EdgeType.RELATED, weight=0.4)
    archived.add_edge(without_vector.id, EdgeType.RELATED, weight=0.6)
    return [active, archived, without_vector]


def _legacy_database(path, records):
    # Freeze the pre-contract format; do not create it through today's Store.
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            PRAGMA user_version = 2;
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
                importance REAL NOT NULL DEFAULT 5.0, salience REAL NOT NULL DEFAULT 0.0,
                strength INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL,
                last_accessed REAL NOT NULL, access_count INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0, embedding BLOB, data TEXT NOT NULL,
                owner TEXT NOT NULL DEFAULT '', status TEXT
            );
            CREATE TABLE memory_edges (
                src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0, created_at REAL NOT NULL,
                PRIMARY KEY (src, dst, type)
            );
            """
        )
        vectors = [np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32), None]
        for memory, vector in zip(records, vectors, strict=True):
            conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory.id,
                    memory.type.value,
                    memory.content,
                    memory.importance,
                    memory.salience,
                    memory.strength,
                    memory.created_at,
                    memory.last_accessed_at,
                    memory.access_count,
                    int(memory.archived),
                    vector.tobytes() if vector is not None else None,
                    json.dumps(memory.model_dump(mode="json", exclude={"edges"})),
                    memory.owner,
                    memory.status,
                ),
            )
            for edge in memory.edges:
                conn.execute(
                    "INSERT INTO memory_edges VALUES (?,?,?,?,?)",
                    (memory.id, edge.target_id, edge.type.value, edge.weight, edge.created_at),
                )
        conn.commit()


def _seed_store(stores, path, *, legacy=False):
    records = _records()
    if legacy:
        _legacy_database(path, records)
        store = stores(path)
    else:
        embedder = _Embedder("model-a", 2)
        store = stores(path, embedding_identity=embedder.identity, embedding_dim=embedder.dim)
        for index, memory in enumerate(records):
            vector = embedder.embed(memory.embedding_text()) if index < 2 else None
            store.add(memory, vector)
    store.log_maintenance("reflect", {"created": 2, "notes": ["retain this report"]})
    store.log_maintenance("prune", {"archived": 1})
    return store


def _canonical_snapshot(path):
    with closing(sqlite3.connect(path)) as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("memories", "memory_edges", "maintenance_log")
        }


@pytest.mark.parametrize("identity, dim", [("model-b", 2), ("model-a", 3)])
def test_named_contract_rejects_incompatible_reopen(stores, tmp_path, identity, dim):
    path = tmp_path / "memory.db"
    store = _seed_store(stores, path)
    before = _canonical_snapshot(path)

    with pytest.raises(EmbeddingMismatchError):
        stores(path, embedding_identity=identity, embedding_dim=dim)

    reopened = stores(path, embedding_identity="model-a", embedding_dim=2)
    assert reopened.get("active").model_dump() == store.get("active").model_dump()
    assert reopened.knn(reopened.get_embedding("active"), 1)[0][0] == "active"
    assert _canonical_snapshot(path) == before


def test_raw_store_binds_dimension_and_reopens_without_claiming_a_model(stores, tmp_path):
    path = tmp_path / "raw.db"
    store = stores(path)
    vector = np.array([1.0, 0.0], dtype=np.float32)
    store.add(Memory(id="raw", content="User-supplied vectors"), vector)

    reopened = stores(path)
    np.testing.assert_array_equal(reopened.get_embedding("raw"), vector)
    assert reopened.knn(vector, 1)[0][0] == "raw"
    with pytest.raises(EmbeddingMismatchError):
        reopened.add(Memory(id="bad", content="Wrong dimensions"), np.ones(3))
    with pytest.raises(EmbeddingMismatchError):
        reopened.bind_embeddings("model-a", 2)
    with pytest.raises(EmbeddingMismatchError):
        stores(path, embedding_identity="model-a", embedding_dim=2)
    assert reopened.get("bad") is None


def test_empty_store_can_be_explicitly_bound_before_its_first_vector(stores, tmp_path):
    path = tmp_path / "empty.db"
    store = stores(path)
    store.bind_embeddings("model-a", 2)
    store.add(Memory(id="bound", content="Explicit space"), np.ones(2))

    reopened = stores(path, embedding_identity="model-a", embedding_dim=2)
    assert reopened.knn(np.ones(2), 1)[0][0] == "bound"


def test_omitted_identity_opens_named_store_for_inspection_only(stores, tmp_path):
    path = tmp_path / "named.db"
    source = _seed_store(stores, path)
    before = _canonical_snapshot(path)
    inspector = stores(path)

    assert inspector.get("active").model_dump() == source.get("active").model_dump()
    np.testing.assert_array_equal(inspector.get_embedding("active"), source.get_embedding("active"))
    with pytest.raises(EmbeddingMismatchError):
        inspector.knn(np.ones(2), 1)
    with pytest.raises(EmbeddingMismatchError):
        inspector.add(Memory(content="Identity was not declared"), np.ones(2))
    assert _canonical_snapshot(path) == before


@pytest.mark.parametrize("method", ["add", "save"])
@pytest.mark.parametrize(
    "invalid_vector",
    [np.ones(3), np.ones((1, 2)), np.array([np.nan, 1]), np.array([np.inf, 1])],
    ids=["wrong-dimension", "matrix", "nan", "infinity"],
)
def test_invalid_vector_keeps_text_metadata_edges_and_existing_vector(
    stores, tmp_path, method, invalid_vector
):
    path = tmp_path / "memory.db"
    store = _seed_store(stores, path)
    before = _canonical_snapshot(path)
    changed = store.get("active")
    changed.content = "This failed update must never replace the original text."
    changed.owner = "wrong-owner"
    changed.edges.clear()

    with pytest.raises(EmbeddingMismatchError):
        getattr(store, method)(changed, embedding=invalid_vector)

    assert _canonical_snapshot(path) == before
    assert store.knn(store.get_embedding("active"), 1)[0][0] == "active"


@pytest.mark.parametrize(
    "invalid_vector",
    [np.array([]), np.ones((1, 2)), np.array([np.nan, 1])],
    ids=["empty", "matrix", "nan"],
)
def test_failed_first_vector_does_not_bind_an_empty_store(stores, tmp_path, invalid_vector):
    path = tmp_path / "memory.db"
    store = stores(path)

    with pytest.raises(EmbeddingMismatchError):
        store.add(Memory(id="invalid", content="Must not be written"), invalid_vector)

    assert store.count(include_archived=True) == 0
    named = stores(path, embedding_identity="model-b", embedding_dim=3)
    named.add(Memory(id="valid", content="A different space can still be bound"), np.ones(3))
    assert named.knn(np.ones(3), 1)[0][0] == "valid"


def test_failed_transaction_rolls_back_the_first_vector_and_its_contract(stores, tmp_path):
    path = tmp_path / "memory.db"
    store = stores(path)

    with pytest.raises(EmbeddingMismatchError), store.transaction():
        store.add(Memory(id="first", content="Roll back this record too"), np.ones(2))
        store.add(Memory(id="invalid", content="Wrong dimension"), np.ones(3))

    assert store.count(include_archived=True) == 0
    store.bind_embeddings("model-b", 3)
    store.add(Memory(id="valid", content="The rolled-back contract must not remain"), np.ones(3))
    reopened = stores(path, embedding_identity="model-b", embedding_dim=3)
    assert reopened.knn(np.ones(3), 10)[0][0] == "valid"


@pytest.mark.parametrize("identity,dim", [("model-b", 2), ("model-a", 3)])
def test_failed_bind_keeps_the_original_handle_usable(stores, tmp_path, identity, dim):
    path = tmp_path / "memory.db"
    store = _seed_store(stores, path)
    before = _canonical_snapshot(path)

    with pytest.raises(EmbeddingMismatchError):
        store.bind_embeddings(identity, dim)

    assert _canonical_snapshot(path) == before
    assert store.knn(store.get_embedding("active"), 1)[0][0] == "active"
    store.add(Memory(id="valid", content="The original contract remains usable"), np.ones(2))
    assert store.get("valid") is not None


def test_query_recovers_after_a_temporary_index_initialization_failure(
    stores, tmp_path, monkeypatch
):
    path = tmp_path / "memory.db"
    source = _seed_store(stores, path)
    inspection = stores(path, vector_config=VectorConfig(backend="numpy"))
    before = _canonical_snapshot(path)

    def unavailable_backend(*args, **kwargs):
        raise RuntimeError("temporary backend initialization failure")

    with monkeypatch.context() as patch:
        patch.setattr("aster_memory.store.make_vector_index", unavailable_backend)
        with pytest.raises(RuntimeError, match="temporary backend initialization failure"):
            inspection.bind_embeddings("model-a", 2)

    # Recovery must load the existing records instead of returning an empty dummy index.
    assert inspection.knn(source.get_embedding("active"), 10)[0][0] == "active"
    assert _canonical_snapshot(path) == before


def test_mixed_dimension_legacy_store_is_inspectable_but_cannot_be_claimed(stores, tmp_path):
    path = tmp_path / "legacy.db"
    store = _seed_store(stores, path, legacy=True)
    before = _canonical_snapshot(path)

    assert len(store.all_memories(include_archived=True)) == 3
    assert store.get_embedding("active").shape == (2,)
    assert store.get_embedding("archived").shape == (3,)
    with pytest.raises(EmbeddingMismatchError):
        store.knn(np.ones(2), 1)
    with pytest.raises(EmbeddingMismatchError):
        store.bind_embeddings("model-a", 2)
    with pytest.raises(EmbeddingMismatchError):
        stores(path, embedding_identity="model-a", embedding_dim=2)
    with pytest.raises(EmbeddingMismatchError):
        store.save(store.get("active"), embedding=np.ones(2))
    assert _canonical_snapshot(path) == before


def test_first_writer_binds_dimension_for_already_open_raw_handles(stores, tmp_path):
    path = tmp_path / "shared.db"
    first = stores(path)
    stale = stores(path)
    first.add(Memory(id="first", content="First writer"), np.ones(2))

    with pytest.raises(EmbeddingMismatchError):
        stale.add(Memory(id="bad", content="Stale writer"), np.ones(3))
    assert first.get("bad") is None
    stale.add(Memory(id="compatible", content="Same raw space"), np.ones(2))
    assert first.get("compatible") is not None


def test_already_open_unbound_handle_cannot_overwrite_another_binding(stores, tmp_path):
    path = tmp_path / "shared.db"
    stale = stores(path)
    winner = stores(path)
    winner.bind_embeddings("model-b", 2)
    winner.add(Memory(id="winner", content="Named space"), np.ones(2))

    with pytest.raises(EmbeddingMismatchError):
        stale.bind_embeddings("model-a", 2)
    with pytest.raises(EmbeddingMismatchError):
        stale.add(Memory(id="bad", content="Unclaimed space"), np.ones(2))
    assert winner.get("bad") is None


@pytest.mark.parametrize("operation", ["add", "save", "knn", "get_embedding"])
@pytest.mark.parametrize("replacement_identity", ["model-a", "model-b"])
def test_stale_named_handle_rechecks_contract_on_every_vector_operation(
    stores, tmp_path, operation, replacement_identity
):
    path = tmp_path / "live.db"
    stale = stores(path, embedding_identity="model-a", embedding_dim=2)
    old = Memory(id="shared", content="Old model's data")
    stale.add(old, np.ones(2))
    replacement_path = tmp_path / "replacement.db"
    replacement = stores(replacement_path, embedding_identity=replacement_identity, embedding_dim=2)
    replacement.add(Memory(id="shared", content="Protect the new model's data"), np.ones(2))

    # An external SQLite writer can replace the contract while a manager stays alive.
    with (
        closing(sqlite3.connect(replacement_path)) as source,
        closing(sqlite3.connect(path)) as destination,
    ):
        source.backup(destination)
    before = _canonical_snapshot(path)

    for _ in range(2):
        with pytest.raises(EmbeddingMismatchError):
            if operation in {"add", "save"}:
                getattr(stale, operation)(old, embedding=np.ones(2))
            elif operation == "knn":
                stale.knn(np.ones(2), 1)
            else:
                stale.get_embedding(old.id)
    assert _canonical_snapshot(path) == before


@pytest.mark.parametrize("operation", ["add", "save", "knn", "get_embedding"])
@pytest.mark.parametrize("bind_named", [False, True], ids=["raw", "bound-named"])
def test_bound_handle_rejects_a_replaced_undeclared_database_before_writing(
    stores, tmp_path, operation, bind_named
):
    path = tmp_path / "live.db"
    stale = stores(path)
    if bind_named:
        stale.bind_embeddings("model-a", 2)
    old = Memory(id="shared", content="The handle originally owned a vector")
    stale.add(old, np.ones(2))
    replacement_path = tmp_path / "text-only.db"
    replacement = stores(replacement_path)
    replacement.add(Memory(id="shared", content="Protect unidentified legacy text"))
    with (
        closing(sqlite3.connect(replacement_path)) as source,
        closing(sqlite3.connect(path)) as destination,
    ):
        source.backup(destination)
    before = _canonical_snapshot(path)

    for _ in range(2):
        with pytest.raises(EmbeddingMismatchError):
            if operation in {"add", "save"}:
                getattr(stale, operation)(old, embedding=np.ones(2))
            elif operation == "knn":
                stale.knn(np.ones(2), 1)
            else:
                stale.get_embedding(old.id)
        assert _canonical_snapshot(path) == before


@pytest.mark.parametrize("legacy_kind", ["text-only", "archived-only"])
def test_nonempty_legacy_store_cannot_be_claimed_without_live_vectors(
    stores, tmp_path, legacy_kind
):
    path = tmp_path / "legacy.db"
    _legacy_database(path, _records())
    with closing(sqlite3.connect(path)) as conn:
        if legacy_kind == "text-only":
            conn.execute("UPDATE memories SET embedding = NULL")
        else:
            conn.execute("UPDATE memories SET archived = 1")
        conn.commit()
    inspection = stores(path)
    before = _canonical_snapshot(path)

    assert inspection.count(include_archived=True) == 3
    with pytest.raises(EmbeddingMismatchError):
        inspection.bind_embeddings("model-a", 2)
    with pytest.raises(EmbeddingMismatchError):
        inspection.add(Memory(id="invalid", content="Do not claim old rows"), np.ones(2))
    with pytest.raises(EmbeddingMismatchError):
        stores(path, embedding_identity="model-a", embedding_dim=2)
    assert _canonical_snapshot(path) == before


def test_failed_refresh_rejects_invalid_vectors_until_they_are_repaired(stores, tmp_path):
    path = tmp_path / "memory.db"
    store = stores(path, embedding_identity="model-a", embedding_dim=2)
    store.add(Memory(id="valid", content="An externally damaged vector"), np.ones(2))
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (np.ones(3, dtype=np.float32).tobytes(), "valid"),
        )
        conn.commit()

        for _ in range(2):
            with pytest.raises(EmbeddingMismatchError):
                store.knn(np.ones(2), 1)

        conn.execute(
            "UPDATE memories SET embedding = ? WHERE id = ?",
            (np.ones(2, dtype=np.float32).tobytes(), "valid"),
        )
        conn.commit()

    assert store.knn(np.ones(2), 1)[0][0] == "valid"


@pytest.mark.parametrize("legacy", [False, True], ids=["named", "mixed-legacy"])
def test_rebuild_preserves_all_records_relations_history_and_source(stores, tmp_path, legacy):
    path = tmp_path / "source.db"
    source = _seed_store(stores, path, legacy=legacy)
    original_rows = source.all_memories(include_archived=True)
    original_records = {memory.id: memory.model_dump() for memory in original_rows}
    original_history = source.maintenance_history()
    before = _canonical_snapshot(path)
    embedder = _Embedder("model-b", 4)
    destination = tmp_path / "rebuilt.db"

    assert source.rebuild_embeddings(embedder, destination, batch_size=1) == destination

    rebuilt = stores(destination, embedding_identity=embedder.identity, embedding_dim=embedder.dim)
    assert {
        memory.id: memory.model_dump() for memory in rebuilt.all_memories(include_archived=True)
    } == original_records
    assert rebuilt.maintenance_history() == original_history
    assert sorted(embedder.seen) == sorted(memory.embedding_text() for memory in original_rows)
    for memory in original_rows:
        np.testing.assert_array_equal(
            rebuilt.get_embedding(memory.id), embedder.embed(memory.embedding_text())
        )
    assert "archived" not in {mid for mid, _ in rebuilt.knn(np.ones(4), 10)}
    assert _canonical_snapshot(path) == before
    if legacy:
        with pytest.raises(EmbeddingMismatchError):
            source.knn(np.ones(2), 1)
        with pytest.raises(EmbeddingMismatchError):
            source.bind_embeddings("model-b", 4)
    else:
        reopened = stores(path, embedding_identity="model-a", embedding_dim=2)
        assert reopened.knn(source.get_embedding("active"), 1)[0][0] == "active"
    with pytest.raises(EmbeddingMismatchError):
        stores(destination, embedding_identity="model-a", embedding_dim=2)


def test_rebuild_rejects_active_transaction_without_rolling_back_the_callers_writes(
    stores, tmp_path
):
    source = _seed_store(stores, tmp_path / "source.db")
    destination = tmp_path / "rebuilt.db"
    embedder = _Embedder("model-b", 4)

    with source.transaction():
        source.add(Memory(id="pending", content="Retain the caller's pending write"), np.ones(2))
        with pytest.raises(RuntimeError, match="outside a write transaction"):
            source.rebuild_embeddings(embedder, destination)
        assert source.get("pending") is not None
        source.add(Memory(id="after", content="The transaction can still continue"), np.ones(2))

    assert source.get("pending") is not None
    assert source.get("after") is not None
    assert embedder.seen == []
    assert not destination.exists()
    reopened = stores(source.path, embedding_identity="model-a", embedding_dim=2)
    assert reopened.get("pending") is not None
    assert reopened.get("after") is not None


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_rebuild_failure_leaves_source_and_destination_untouched(
    stores, tmp_path, monkeypatch, error_type
):
    path = tmp_path / "source.db"
    source = _seed_store(stores, path)
    before = _canonical_snapshot(path)
    files_before = {entry.name for entry in tmp_path.iterdir()}
    embedder = _Embedder("model-b", 4)
    original_batch = embedder.embed_batch
    calls = 0

    def fail_after_first_batch(texts):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error_type("simulated embedding interruption")
        return original_batch(texts)

    monkeypatch.setattr(embedder, "embed_batch", fail_after_first_batch)
    destination = tmp_path / "unfinished.db"
    with pytest.raises(error_type, match="simulated embedding interruption"):
        source.rebuild_embeddings(embedder, destination, batch_size=1)

    assert calls == 2
    assert not destination.exists()
    assert {entry.name for entry in tmp_path.iterdir()} == files_before
    assert _canonical_snapshot(path) == before
    assert source.knn(source.get_embedding("active"), 1)[0][0] == "active"


@pytest.mark.parametrize("failure", ["short-batch", "wrong-dimension", "nonfinite"])
def test_rebuild_rejects_invalid_embedding_batches(stores, tmp_path, monkeypatch, failure):
    path = tmp_path / "source.db"
    source = _seed_store(stores, path)
    before = _canonical_snapshot(path)
    embedder = _Embedder("model-b", 4)

    def invalid_batch(texts):
        if failure == "short-batch":
            return [np.ones(4) for _ in texts[:-1]]
        if failure == "wrong-dimension":
            return [np.ones(3) for _ in texts]
        return [np.full(4, np.nan) for _ in texts]

    monkeypatch.setattr(embedder, "embed_batch", invalid_batch)
    destination = tmp_path / "invalid.db"
    with pytest.raises(ValueError):
        source.rebuild_embeddings(embedder, destination)

    assert not destination.exists()
    assert _canonical_snapshot(path) == before


@pytest.mark.parametrize("created_when", ["before", "during"])
def test_rebuild_never_overwrites_an_existing_destination(
    stores, tmp_path, monkeypatch, created_when
):
    path = tmp_path / "source.db"
    source = _seed_store(stores, path)
    before = _canonical_snapshot(path)
    embedder = _Embedder("model-b", 4)
    destination = tmp_path / "claimed.db"
    protected = b"Another process owns this destination."
    if created_when == "before":
        destination.write_bytes(protected)
    else:
        original_batch = embedder.embed_batch

        def claim_destination(texts):
            if not destination.exists():
                destination.write_bytes(protected)
            return original_batch(texts)

        monkeypatch.setattr(embedder, "embed_batch", claim_destination)

    with pytest.raises(FileExistsError):
        source.rebuild_embeddings(embedder, destination, batch_size=1)

    assert destination.read_bytes() == protected
    assert _canonical_snapshot(path) == before
    if created_when == "before":
        assert not embedder.seen
