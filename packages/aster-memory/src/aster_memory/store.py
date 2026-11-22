# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-centric memory store + pluggable vector index.

Canonical records and vectors live in one SQLite file selected by
``MemoryConfig.path`` or the host. Hosts choose session or workspace/project paths
explicitly:

* ``memories``      — full record as JSON + promoted columns for filtering/sort
                      + the embedding as a float32 blob.
* ``memory_edges``  — the directed, typed association graph.

The vector index is abstracted behind ``VectorIndex`` so the brute-force numpy
default can later be swapped for ``sqlite-vec`` / Chroma without touching callers.
The default ``NumpyVectorIndex`` does exact cosine KNN in memory over
L2-normalised vectors.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aster_memory.embeddings import EmbeddingMismatchError, validate_embedding
from aster_memory.schema import Edge, EdgeType, Memory, MemoryType
from aster_memory.vector_backends import (
    NumpyVectorIndex,
    VectorIndex,
    make_vector_index,
)

if TYPE_CHECKING:
    from aster_memory.config import VectorConfig
    from aster_memory.embeddings import Embedder


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


# Stamp of the DDL shape this code understands (PRAGMA user_version).
SCHEMA_VERSION = 3
_RAW_EMBEDDING_IDENTITY = "raw-vectors:v1"


class MemorySchemaError(RuntimeError):
    """The store file's schema is newer than this code understands."""


# Note: the two v2 columns (owner/status) appear in the CREATE TABLE for fresh
# files; their indexes are created in ``_migrate`` because on a v1 file the
# columns don't exist until the ALTERs run.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 5.0,
    salience      REAL NOT NULL DEFAULT 0.0,
    strength      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    archived      INTEGER NOT NULL DEFAULT 0,
    owner         TEXT NOT NULL DEFAULT '',
    status        TEXT,
    embedding     BLOB,
    data          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_archived ON memories(archived);

CREATE TABLE IF NOT EXISTS memory_edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    type       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON memory_edges(src);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON memory_edges(dst);

CREATE TABLE IF NOT EXISTS maintenance_log (
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    report TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_contract (
    id        INTEGER PRIMARY KEY CHECK (id = 1),
    identity  TEXT NOT NULL,
    dim       INTEGER NOT NULL CHECK (dim > 0),
    namespace TEXT NOT NULL
);
"""


class MemoryStore:
    """Persistent store for memories + their association graph + vectors."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        vector_index: VectorIndex | None = None,
        vector_config: VectorConfig | None = None,
        embedding_dim: int | None = None,
        embedding_identity: str | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._pending_index: set[str] | None = None
        self._embedding_identity: str | None = None
        self._embedding_dim = embedding_dim
        self._embedding_namespace: str | None = None
        self._vector_config = vector_config
        self._provided_index = vector_index
        self._index: VectorIndex = NumpyVectorIndex()
        self._index_ready = False
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            with self._lock:
                self._conn.row_factory = sqlite3.Row
                if self.path != ":memory:":
                    self._conn.execute("PRAGMA journal_mode=WAL")
                    self._conn.execute("PRAGMA busy_timeout=5000")
                self._migrate()
                self._data_version = self._current_data_version()
                contract = self._embedding_contract()
                if embedding_identity is not None:
                    self.bind_embeddings(embedding_identity, embedding_dim)
                elif contract is not None and contract["identity"] == _RAW_EMBEDDING_IDENTITY:
                    self.bind_embeddings(
                        _RAW_EMBEDDING_IDENTITY,
                        embedding_dim if embedding_dim is not None else contract["dim"],
                    )
                elif contract is None and not self.count(include_archived=True):
                    self._embedding_identity = _RAW_EMBEDDING_IDENTITY
                    if embedding_dim is not None:
                        self.bind_embeddings(_RAW_EMBEDDING_IDENTITY, embedding_dim)
                    else:
                        self._initialize_index()
                elif embedding_dim is not None and contract is not None:
                    if embedding_dim != contract["dim"]:
                        raise EmbeddingMismatchError(
                            "Embedding dimension differs from the stored contract; "
                            "use rebuild_embeddings() with a new destination."
                        )
                # Existing undeclared stores remain available for text inspection.
        except BaseException:
            self._conn.close()
            raise

    def _embedding_contract(self) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM embedding_contract WHERE id = 1").fetchone()

    def bind_embeddings(self, identity: str, dim: int | None) -> None:
        """Declare the vector space, checking existing data before opening an index.

        Only an empty, unbound database can acquire a new identity. Existing
        unidentified/raw vectors must be explicitly rebuilt, even at the same dim.
        """
        if not isinstance(identity, str) or not identity.strip():
            raise EmbeddingMismatchError("An explicit, non-empty embedding identity is required.")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise EmbeddingMismatchError("The embedding dimension must be a positive integer.")
        with self._lock:
            if self._pending_index is not None:
                raise RuntimeError("Bind embeddings outside a write transaction.")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                contract = self._embedding_contract()
                if contract is not None:
                    if contract["identity"] != identity or contract["dim"] != dim:
                        raise EmbeddingMismatchError(
                            "Embedding identity or dimension differs from the stored contract; "
                            "use rebuild_embeddings() with a new destination."
                        )
                elif self.count(include_archived=True):
                    raise EmbeddingMismatchError(
                        "Existing memories have no embedding identity. Inspect their text or "
                        "use rebuild_embeddings() with a new destination; dimensions alone "
                        "cannot establish compatibility."
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO embedding_contract (id, identity, dim, namespace) "
                        "VALUES (1, ?, ?, ?)",
                        (identity, dim, uuid.uuid4().hex),
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            contract = self._embedding_contract()
            if (
                self._index_ready
                and self._embedding_identity == identity
                and self._embedding_dim == dim
                and self._embedding_namespace == contract["namespace"]
            ):
                return
            self._embedding_identity = identity
            self._embedding_dim = dim
            self._embedding_namespace = contract["namespace"]
            self._index_ready = False
            self._initialize_index()

    def rebuild_embeddings(
        self, embedder: Embedder, destination: str | Path, *, batch_size: int = 64
    ) -> Path:
        """Re-embed a consistent snapshot into a new NumPy-backed database.

        Preserves every memory (including archived rows), edge and maintenance
        record. The source is untouched. A failed/interrupted rebuild never
        publishes a partial destination, and an existing destination is never replaced.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if str(destination) == ":memory:":
            raise ValueError("rebuild_embeddings needs a new filesystem destination")
        destination = Path(destination).expanduser().resolve()
        if os.path.lexists(destination):
            raise FileExistsError(destination)
        identity, dim = embedder.identity, embedder.dim
        if not isinstance(identity, str) or not identity.strip():
            raise EmbeddingMismatchError("The replacement embedder needs a stable identity.")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise EmbeddingMismatchError("The replacement embedder needs a positive dimension.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-rebuild-", dir=destination.parent
        ) as directory:
            staged = Path(directory) / "memory.sqlite"
            target = sqlite3.connect(staged)
            cursor: sqlite3.Cursor | None = None
            try:
                target.row_factory = sqlite3.Row
                target.executescript(_SCHEMA)
                target.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                # Copy only canonical tables: ANN tables/collections are derived,
                # and copying their extension schema would make recovery depend on it.
                with self._lock:
                    if self._conn.in_transaction:
                        raise RuntimeError("Rebuild embeddings outside a write transaction.")
                    self._conn.execute("BEGIN")
                    try:
                        for table in ("memories", "memory_edges", "maintenance_log"):
                            columns = [
                                row["name"] for row in target.execute(f"PRAGMA table_info({table})")
                            ]
                            names = ", ".join(columns)
                            placeholders = ", ".join("?" for _ in columns)
                            rows = self._conn.execute(f"SELECT {names} FROM {table}")
                            target.executemany(
                                f"INSERT INTO {table} ({names}) VALUES ({placeholders})", rows
                            )
                        target.execute(
                            "INSERT INTO embedding_contract (id, identity, dim, namespace) "
                            "VALUES (1, ?, ?, ?)",
                            (identity, dim, uuid.uuid4().hex),
                        )
                        target.commit()
                    finally:
                        self._conn.rollback()
                cursor = target.execute("SELECT id, data FROM memories ORDER BY id")
                while rows := cursor.fetchmany(batch_size):
                    texts = [
                        Memory.model_validate_json(row["data"]).embedding_text() for row in rows
                    ]
                    vectors = embedder.embed_batch(texts)
                    if len(vectors) != len(rows):
                        raise EmbeddingMismatchError(
                            "The replacement embedder returned the wrong number of vectors."
                        )
                    if embedder.identity != identity or embedder.dim != dim:
                        raise EmbeddingMismatchError("The replacement embedding contract changed.")
                    blobs = [
                        (_vec_to_blob(validate_embedding(vector, dim)), row["id"])
                        for row, vector in zip(rows, vectors, strict=True)
                    ]
                    with target:
                        target.executemany("UPDATE memories SET embedding = ? WHERE id = ?", blobs)
            finally:
                if cursor is not None:
                    cursor.close()
                target.close()
            # An atomic, non-overwriting publish on the same filesystem. Unlike
            # replace()/POSIX rename(), link() cannot clobber a concurrently created file.
            os.link(staged, destination)
        return destination

    def _check_embedding_contract(self) -> int | None:
        contract = self._embedding_contract()
        if self._embedding_identity is None:
            raise EmbeddingMismatchError(
                "This store is open for inspection without an embedding identity. "
                "Open it with the matching identity/dimension or use rebuild_embeddings()."
            )
        if contract is None:
            if (
                self._embedding_identity == _RAW_EMBEDDING_IDENTITY
                and self._embedding_namespace is None
                and not self._conn.execute(
                    "SELECT 1 FROM memories WHERE embedding IS NOT NULL LIMIT 1"
                ).fetchone()
            ):
                return self._embedding_dim
            raise EmbeddingMismatchError(
                "Stored vectors have no embedding identity; use rebuild_embeddings()."
            )
        if (
            contract["identity"] != self._embedding_identity
            or (self._embedding_dim is not None and contract["dim"] != self._embedding_dim)
            or (
                self._embedding_namespace is not None
                and contract["namespace"] != self._embedding_namespace
            )
        ):
            raise EmbeddingMismatchError(
                "The stored embedding contract changed or differs from this handle; "
                "reopen with the matching embedder or use rebuild_embeddings()."
            )
        return contract["dim"]

    def _prepare_embedding(self, vector: np.ndarray, *, write: bool = False) -> np.ndarray:
        dim = self._check_embedding_contract()
        if dim is None:
            candidate = np.asarray(vector)
            if candidate.ndim != 1 or not candidate.size:
                raise EmbeddingMismatchError("An embedding must be a non-empty 1-D vector.")
            dim = int(candidate.size)
        validated = validate_embedding(vector, dim)
        if write and self._embedding_contract() is None:
            self._conn.execute(
                "INSERT INTO embedding_contract (id, identity, dim, namespace) VALUES (1, ?, ?, ?)",
                (self._embedding_identity, dim, uuid.uuid4().hex),
            )
        return validated

    def _initialize_index(self) -> None:
        dim = self._check_embedding_contract()
        version = self._current_data_version()
        entries = self._index_entries(dim)
        contract = self._embedding_contract()
        if self._provided_index is not None:
            index = self._provided_index
        elif self._vector_config is not None:
            if self._vector_config.backend != "numpy" and dim is None:
                raise EmbeddingMismatchError("This vector backend needs an explicit embedding dim.")
            index = make_vector_index(
                self._vector_config,
                dim=dim,
                conn=self._conn,
                path=self.path,
                namespace=contract["namespace"] if contract is not None else None,
            )
        else:
            index = NumpyVectorIndex()
        for memory_id, embedding in entries:
            index.add(memory_id, embedding)
        self._index = index
        self._index_ready = True
        self._data_version = version

    def _index_entries(self, dim: int | None) -> list[tuple[str, np.ndarray]]:
        entries = []
        for row in self._conn.execute(
            "SELECT id, archived, embedding FROM memories WHERE embedding IS NOT NULL"
        ):
            embedding = validate_embedding(_blob_to_vec(row["embedding"]), dim)
            if not row["archived"]:
                entries.append((row["id"], embedding))
        return entries

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        """Bring the file to ``SCHEMA_VERSION``; refuse files from the future.

        Hybrid on purpose: the upgrades themselves are introspection-guarded
        idempotent ALTERs (a half-migrated file converges), while the version
        stamp does the one thing introspection cannot — fail loudly when the
        file was written by *newer* code (shared project stores can be opened
        by processes running different checkouts).
        """
        with self._lock:
            version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise MemorySchemaError(
                    f"memory store {self.path!r} has schema v{version}, but this code "
                    f"understands up to v{SCHEMA_VERSION} — update aster"
                )
            self._conn.executescript(_SCHEMA)
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
            if "owner" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
            if "status" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN status TEXT")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_owner ON memories(owner)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)")
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._conn.commit()

    def _current_data_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def refresh_if_changed(self) -> bool:
        """Reload the derived NumPy index after a compatible external commit."""
        with self._lock:
            version = self._current_data_version()
            if self._embedding_identity is not None:
                dim = self._check_embedding_contract()
            else:
                dim = None
            if version == self._data_version:
                return False
            if self._index_ready and isinstance(self._index, NumpyVectorIndex):
                entries = self._index_entries(dim)
                index = NumpyVectorIndex()
                for memory_id, embedding in entries:
                    index.add(memory_id, embedding)
                self._index = index
            self._data_version = version
            return True

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        data = json.loads(row["data"])
        edges = [
            Edge(
                target_id=e["dst"],
                type=EdgeType(e["type"]),
                weight=e["weight"],
                created_at=e["created_at"],
            )
            for e in self._edge_rows(row["id"])
        ]
        data["edges"] = [e.model_dump(mode="json") for e in edges]
        # Promoted columns are authoritative (archive() flips the column;
        # v1 payloads predate owner/status), so they override whatever the
        # serialised payload captured at write time.
        data["archived"] = bool(row["archived"])
        data["owner"] = row["owner"]
        data["status"] = row["status"]
        return Memory.model_validate(data)

    def _edge_rows(self, src: str) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT dst, type, weight, created_at FROM memory_edges WHERE src = ?", (src,)
            )
            return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Commit grouped writes together; nested writes join the outer transaction.

        Keep model/embedding calls outside this context. Derived vector indexes
        are updated only after commit, so rollback never exposes their writes.
        """
        with self._lock:
            if self._pending_index is not None:
                yield
                return
            self._conn.execute("BEGIN IMMEDIATE")
            self._pending_index = set()
            try:
                if self._embedding_identity is not None:
                    self._check_embedding_contract()
                yield
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            finally:
                pending = self._pending_index
                self._pending_index = None
            contract = self._embedding_contract()
            if self._embedding_identity is not None and contract is not None:
                self._check_embedding_contract()
                self._embedding_dim = contract["dim"]
                self._embedding_namespace = contract["namespace"]
            if not self._index_ready:
                return
            # Some indexes share this connection and commit their own writes.
            for id in pending:
                row = self._conn.execute(
                    "SELECT archived, embedding FROM memories WHERE id = ?", (id,)
                ).fetchone()
                if row is not None and not row["archived"] and row["embedding"] is not None:
                    self._index.add(id, _blob_to_vec(row["embedding"]))
                else:
                    self._index.remove(id)

    @contextmanager
    def edit(self, id: str, *, embedding: np.ndarray | None = None) -> Iterator[Memory | None]:
        """Read and mutate the latest record under one database write transaction.

        Keep the body to in-memory changes; the updated record is saved on exit.
        """
        with self.transaction():
            memory = self.get(id)
            before = memory.model_dump() if memory is not None else None
            yield memory
            if memory is not None and memory.model_dump() != before:
                self.save(memory, embedding=embedding)

    def add(self, memory: Memory, embedding: np.ndarray | None = None) -> Memory:
        """Insert (or replace) a memory and its edges. Persists the embedding."""
        with self.transaction():
            blob = (
                _vec_to_blob(self._prepare_embedding(embedding, write=True))
                if embedding is not None
                else None
            )
            payload = memory.model_dump(mode="json", exclude={"edges"})
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, type, content, importance, salience, strength, created_at,
                    last_accessed, access_count, archived, owner, status, embedding, data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    memory.owner,
                    memory.status,
                    blob,
                    json.dumps(payload),
                ),
            )
            # Rewrite edges for this source.
            self._conn.execute("DELETE FROM memory_edges WHERE src = ?", (memory.id,))
            for e in memory.edges:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (memory.id, e.target_id, e.type.value, e.weight, e.created_at),
                )
            self._pending_index.add(memory.id)
            return memory

    def save(self, memory: Memory, *, embedding: np.ndarray | None = None) -> None:
        """Persist mutations, keeping the stored embedding unless one is supplied."""
        with self.transaction():
            payload = memory.model_dump(mode="json", exclude={"edges"})
            self._conn.execute(
                """UPDATE memories SET type=?, content=?, importance=?, salience=?,
                   strength=?, last_accessed=?, access_count=?, archived=?, owner=?,
                   status=?, data=?, embedding=COALESCE(?, embedding) WHERE id=?""",
                (
                    memory.type.value,
                    memory.content,
                    memory.importance,
                    memory.salience,
                    memory.strength,
                    memory.last_accessed_at,
                    memory.access_count,
                    int(memory.archived),
                    memory.owner,
                    memory.status,
                    json.dumps(payload),
                    (
                        _vec_to_blob(self._prepare_embedding(embedding, write=True))
                        if embedding is not None
                        else None
                    ),
                    memory.id,
                ),
            )
            self._conn.execute("DELETE FROM memory_edges WHERE src = ?", (memory.id,))
            for e in memory.edges:
                self._conn.execute(
                    "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
                    " VALUES (?,?,?,?,?)",
                    (memory.id, e.target_id, e.type.value, e.weight, e.created_at),
                )
            if embedding is not None or memory.archived:
                self._pending_index.add(memory.id)

    def add_edge(
        self, src: str, dst: str, type: EdgeType = EdgeType.RELATED, weight: float = 1.0
    ) -> None:
        from aster_memory.schema import _now

        with self.transaction():
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
                " VALUES (?,?,?,?,?)",
                (src, dst, type.value, weight, _now()),
            )

    def archive(self, id: str) -> None:
        with self.transaction():
            self._conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (id,))
            self._pending_index.add(id)

    def delete(self, id: str) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM memories WHERE id = ?", (id,))
            self._conn.execute("DELETE FROM memory_edges WHERE src = ? OR dst = ?", (id, id))
            self._pending_index.add(id)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get(self, id: str) -> Memory | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
            return self._row_to_memory(row) if row else None

    def resolve_id(self, id_or_prefix: str) -> str | None:
        """Exact id, or a unique id prefix (>=6 chars, as rendered in recalled
        lines). Raises on an ambiguous prefix; returns None when nothing matches."""
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM memories WHERE id = ?", (id_or_prefix,)
            ).fetchone()
            if row is not None:
                return str(row["id"])
            if len(id_or_prefix) < 6:
                return None
            prefix = id_or_prefix.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            rows = self._conn.execute(
                "SELECT id FROM memories WHERE id LIKE ? ESCAPE '!' LIMIT 2", (prefix + "%",)
            ).fetchall()
            if len(rows) > 1:
                raise ValueError(f"memory id prefix {id_or_prefix!r} is ambiguous")
            return str(rows[0]["id"]) if rows else None

    def rename_owner(self, old: str, new: str) -> int:
        """Re-stamp every row owned by ``old`` to ``new``; returns rows changed.

        The promoted column is authoritative on read, so the stale ``owner``
        inside the data JSON is harmless. Used to heal legacy owner spellings
        (e.g. the TUI's module:qualname keys) into the canonical short name.
        """
        with self.transaction():
            if old == new or not old:
                return 0  # never mass-claim the unowned/shared namespace
            cur = self._conn.execute("UPDATE memories SET owner = ? WHERE owner = ?", (new, old))
            return cur.rowcount

    def owner_of(self, id: str) -> str | None:
        """The owner column alone (cheap visibility checks during graph spread)."""
        with self._lock:
            row = self._conn.execute("SELECT owner FROM memories WHERE id = ?", (id,)).fetchone()
            return None if row is None else str(row["owner"])

    def get_embedding(self, id: str) -> np.ndarray | None:
        """Read a stored vector; inspection handles never claim model compatibility."""
        with self._lock:
            if self._embedding_identity is not None:
                self._check_embedding_contract()
            row = self._conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (id,)
            ).fetchone()
            if row and row["embedding"] is not None:
                vector = _blob_to_vec(row["embedding"])
                if self._embedding_identity is not None:
                    return self._prepare_embedding(vector)
                return vector
            return None

    def neighbors(self, id: str) -> list[Edge]:
        with self._lock:
            return [
                Edge(
                    target_id=e["dst"],
                    type=EdgeType(e["type"]),
                    weight=e["weight"],
                    created_at=e["created_at"],
                )
                for e in self._edge_rows(id)
            ]

    @staticmethod
    def _owner_clause(owner: str) -> tuple[str, list[object]]:
        """SQL for one owner scope: exact instance for ``role@inst``, role
        scope (all instances + bare-role rows) for a bare role; unowned rows
        always match."""
        if "@" in owner:
            return "owner IN (?, '')", [owner]
        return "(owner = ? OR owner LIKE ? OR owner = '')", [owner, owner + "@%"]

    @classmethod
    def _filters(cls, *, include_archived: bool, owner: str | None) -> tuple[str, list[object]]:
        """WHERE clause for the shared read filters.

        ``owner=None`` means no filtering (the manager resolves policy —
        "mine" / "*" / a name — before calling the store); a concrete scope
        follows the hierarchical-owner semantics of ``_owner_clause``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if not include_archived:
            clauses.append("archived = 0")
        if owner is not None:
            clause, owner_params = cls._owner_clause(owner)
            clauses.append(clause)
            params.extend(owner_params)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def all_memories(
        self, *, include_archived: bool = False, owner: str | None = None
    ) -> list[Memory]:
        with self._lock:
            where, params = self._filters(include_archived=include_archived, owner=owner)
            q = "SELECT * FROM memories" + where
            return [self._row_to_memory(r) for r in self._conn.execute(q, params).fetchall()]

    def iter_memories(
        self, *, include_archived: bool = False, owner: str | None = None
    ) -> Iterator[Memory]:
        yield from self.all_memories(include_archived=include_archived, owner=owner)

    def count(self, *, include_archived: bool = False, owner: str | None = None) -> int:
        with self._lock:
            where, params = self._filters(include_archived=include_archived, owner=owner)
            q = "SELECT COUNT(*) AS n FROM memories" + where
            return int(self._conn.execute(q, params).fetchone()["n"])

    def _owned_ids(self, ids: list[str], owner: str) -> set[str]:
        with self._lock:
            if not ids:
                return set()
            placeholders = ",".join("?" * len(ids))
            clause, owner_params = self._owner_clause(owner)
            rows = self._conn.execute(
                f"SELECT id FROM memories WHERE id IN ({placeholders}) AND {clause}",
                [*ids, *owner_params],
            ).fetchall()
            return {r["id"] for r in rows}

    def knn(
        self, query_vec: np.ndarray, k: int, *, owner: str | None = None
    ) -> list[tuple[str, float]]:
        """Top-k by cosine similarity (excludes archived).

        With ``owner``, results are post-filtered to that owner + unowned rows.
        The ``VectorIndex`` protocol has no metadata filter, so the index is
        oversampled (4x, growing) until ``k`` survive or it is exhausted.
        """
        with self._lock:
            query_vec = self._prepare_embedding(query_vec)
            if not self._index_ready:
                self._initialize_index()
            self.refresh_if_changed()
            if owner is None:
                return self._index.query(query_vec, k)
            total = len(self._index)
            n = min(total, max(k * 4, k))
            while True:
                ranked = self._index.query(query_vec, n)
                allowed = self._owned_ids([mid for mid, _ in ranked], owner)
                out = [(mid, score) for mid, score in ranked if mid in allowed]
                if len(out) >= k or n >= total:
                    return out[:k]
                n = min(total, n * 4)

    def keyword_search(
        self,
        text: str,
        k: int,
        *,
        mem_type: MemoryType | None = None,
        owner: str | None = None,
    ) -> list[str]:
        """Cheap sparse fallback: rank by overlapping-token LIKE matches."""
        from aster_memory.embeddings import _TOKEN_RE

        with self._lock:
            tokens = list(dict.fromkeys(_TOKEN_RE.findall(text.lower())))[:12]
            if not tokens:
                return []
            scored: dict[str, int] = {}
            for tok in tokens:
                clause = "SELECT id FROM memories WHERE archived = 0 AND lower(content) LIKE ?"
                params: list[object] = [f"%{tok}%"]
                if mem_type is not None:
                    clause += " AND type = ?"
                    params.append(mem_type.value)
                if owner is not None:
                    owner_sql, owner_params = self._owner_clause(owner)
                    clause += f" AND {owner_sql}"
                    params.extend(owner_params)
                for row in self._conn.execute(clause, params).fetchall():
                    scored[row["id"]] = scored.get(row["id"], 0) + 1
            ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
            return [mid for mid, _ in ranked[:k]]

    def log_maintenance(self, kind: str, report: dict) -> None:
        """Append one reflect()/prune() run to the store-level maintenance history.

        The one deliberate exception to record-level self-containment: this
        history describes the *store*, not any single memory — it still lives
        in the same SQLite file (nothing external to track).
        """
        from aster_memory.schema import _now

        with self.transaction():
            self._conn.execute(
                "INSERT INTO maintenance_log (ts, kind, report) VALUES (?,?,?)",
                (_now(), kind, json.dumps(report)),
            )

    def maintenance_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, kind, report FROM maintenance_log ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                {"ts": r["ts"], "kind": r["kind"], "report": json.loads(r["report"])} for r in rows
            ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
