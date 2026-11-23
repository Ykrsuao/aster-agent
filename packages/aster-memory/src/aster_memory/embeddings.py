# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedding backends for memory.

Two implementations behind a tiny ``Embedder`` protocol:

* ``HashingEmbedder`` — deterministic, offline, dependency-free. Hashes token
  n-grams into a fixed-dim L2-normalised vector. The default, so the memory
  system works out-of-the-box and tests run without a network.
* ``LiteLLMEmbedder`` — calls an OpenAI-compatible ``/embeddings`` endpoint
  (e.g. the NVIDIA gateway) synchronously via ``litellm.embedding``.

Embeddings are always **L2-normalised**, so cosine similarity == dot product.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol, runtime_checkable

import numpy as np

from aster_memory.config import EmbeddingConfig

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingMismatchError(ValueError):
    """An embedding does not match its declared vector space."""


def validate_embedding(vector: np.ndarray, dim: int) -> np.ndarray:
    """Return a finite float32 vector of exactly the expected dimension."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            vector = np.asarray(vector, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EmbeddingMismatchError("Embedding must contain numeric float32 values") from exc
    if dim <= 0 or vector.ndim != 1 or vector.size != dim:
        raise EmbeddingMismatchError(
            f"Embedding shape {vector.shape} does not match dimension {dim}"
        )
    if not np.isfinite(vector).all():
        raise EmbeddingMismatchError("Embedding must contain only finite values")
    return vector


@runtime_checkable
class Embedder(Protocol):
    """Turns text into an L2-normalised float32 vector in a stable vector space."""

    @property
    def identity(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, text: str) -> np.ndarray: ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]: ...


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = vec.astype(np.float32)
    precise = vec.astype(np.float64)
    norm = float(np.linalg.norm(precise))
    if norm == 0.0:
        return vec
    return (precise / norm).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-(1,2)-grams hashing embedder.

    Not semantically powerful, but stable, fast, free and offline — ideal as a
    zero-config default and for tests. Identical text always maps to the same
    vector; texts that share tokens have positive cosine similarity.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def identity(self) -> str:
        return "hashing:blake2b-word-bigram-v1"

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        grams = list(tokens)
        grams += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        return grams

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for feat in self._features(text):
            h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self._dim
            sign = 1.0 if h[4] & 1 else -1.0  # signed hashing reduces collisions
            vec[idx] += sign
        return _normalize(vec)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


class LiteLLMEmbedder:
    """Synchronous embeddings via ``litellm.embedding`` (OpenAI-compatible)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config.model_copy(
            update={
                "model": config.model.strip(),
                "endpoint": (config.endpoint or "").strip().rstrip("/") or None,
            }
        )
        descriptor = {
            "model": self._config.model,
            "endpoint": self._config.endpoint,
            "dimensions": self._config.dimensions,
        }
        digest = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._identity = f"litellm:v1:{digest}"
        # Dimension comes from the API: the requested `dimensions` (if set) or,
        # left None, probed from the first response. (`config.dim` is the *hashing*
        # backend's size and must NOT be used here.)
        self._dim: int | None = config.dimensions

    @property
    def identity(self) -> str:
        return self._identity

    @property
    def dim(self) -> int:
        if self._dim is None:
            # Probe once to learn the dimension.
            self.embed(" ")
        assert self._dim is not None
        return self._dim

    def _call(self, texts: list[str], dim: int | None) -> list[np.ndarray]:
        import litellm

        kwargs: dict[str, object] = {"model": self._config.model, "input": texts}
        if self._config.endpoint:
            kwargs["api_base"] = self._config.endpoint
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        if self._config.dimensions is not None:
            kwargs["dimensions"] = self._config.dimensions
        # Hard timeout + retries so a hung gateway embedding request can't stall the caller
        # forever (litellm aborts and retries instead of blocking indefinitely).
        kwargs["timeout"] = self._config.timeout
        kwargs["num_retries"] = self._config.num_retries
        resp = litellm.embedding(**kwargs)
        try:
            data = resp["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                raise EmbeddingMismatchError(
                    f"Embedding response must contain exactly {len(texts)} vectors"
                )
            if any("index" in item for item in data):
                indices = [item.get("index") for item in data]
                if any(type(index) is not int for index in indices) or sorted(indices) != list(
                    range(len(texts))
                ):
                    raise EmbeddingMismatchError("Embedding response indices must match the inputs")
                data = sorted(data, key=lambda item: item["index"])
            expected_dim = dim if dim is not None else len(data[0]["embedding"])
            return [
                _normalize(validate_embedding(item["embedding"], expected_dim)) for item in data
            ]
        except (KeyError, TypeError, AttributeError) as exc:
            raise EmbeddingMismatchError("Embedding response contains malformed vectors") from exc

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        bs = max(1, self._config.batch_size)
        out: list[np.ndarray] = []
        dim = self._dim
        for i in range(0, len(texts), bs):
            batch = self._call(texts[i : i + bs], dim)
            if dim is None:
                dim = int(batch[0].size)
            out.extend(batch)
        self._dim = dim
        return out


def get_embedder(config: EmbeddingConfig) -> Embedder:
    """Instantiate the embedder named by ``config.backend``."""
    if config.backend == "hashing":
        return HashingEmbedder(dim=config.dim)
    if config.backend == "litellm":
        return LiteLLMEmbedder(config)
    raise ValueError(f"Unknown embedding backend: {config.backend!r}")
