# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the embedding backends."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from aster_memory.config import EmbeddingConfig
from aster_memory.embeddings import (
    EmbeddingMismatchError,
    HashingEmbedder,
    LiteLLMEmbedder,
    get_embedder,
    validate_embedding,
)


def _cos(a, b):
    return float(np.dot(a, b))


def _response(*vectors):
    return {"data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]}


@pytest.fixture
def embedding_api(monkeypatch):
    api = Mock()
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(embedding=api))
    return api


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    assert np.array_equal(e.embed("deploy the app"), e.embed("deploy the app"))


def test_hashing_embedder_dim_and_normalization():
    e = HashingEmbedder(dim=128)
    v = e.embed("some non empty text")
    assert v.shape == (128,)
    assert e.dim == 128
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(v))


def test_empty_text_is_zero_vector():
    e = HashingEmbedder(dim=32)
    v = e.embed("")
    assert float(np.linalg.norm(v)) == 0.0


def test_similar_text_has_higher_cosine_than_dissimilar():
    e = HashingEmbedder(dim=512)
    base = e.embed("deploy ship the application to production")
    similar = e.embed("ship the application deploy to production")
    dissimilar = e.embed("banana fruit salad recipe with mango")
    assert _cos(base, similar) > _cos(base, dissimilar)


def test_embed_batch_matches_single():
    e = HashingEmbedder(dim=64)
    texts = ["alpha", "beta gamma", "delta"]
    batch = e.embed_batch(texts)
    assert len(batch) == 3
    assert np.array_equal(batch[1], e.embed("beta gamma"))


def test_get_embedder_selects_backend():
    e = get_embedder(EmbeddingConfig(backend="hashing", dim=77))
    assert isinstance(e, HashingEmbedder)
    assert e.dim == 77


def test_get_embedder_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_embedder(EmbeddingConfig(backend="nope"))  # type: ignore[arg-type]


def test_litellm_embedder_dim_uses_dimensions_not_hashing_dim():
    """Regression: LiteLLMEmbedder.dim must come from `dimensions`, not the hashing `dim`.

    (A sqlite-vec/Chroma table is created from this dim; using the 256 hashing
    default against 1024-d text-embedding-3-large caused a live dim-mismatch.)
    """
    e = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dim=256, dimensions=1024))
    assert e._dim == 1024  # set from dimensions, no network probe
    assert e.dim == 1024

    # dimensions unset -> probe lazily (stays None until first call)
    e2 = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dim=256, dimensions=None))
    assert e2._dim is None


def test_hashing_identity_is_stable_and_dimension_is_separate():
    assert HashingEmbedder(64).identity == HashingEmbedder(128).identity
    assert HashingEmbedder(64).identity.startswith("hashing:")


def test_litellm_identity_uses_only_normalized_vector_space_settings(embedding_api):
    config = EmbeddingConfig(
        backend="litellm",
        model="openai/test-embedding",
        endpoint="https://example.test/v1",
        api_key="first-secret",
        dimensions=2,
    )
    original = LiteLLMEmbedder(config)
    equivalent = LiteLLMEmbedder(
        config.model_copy(
            update={
                "model": " openai/test-embedding ",
                "endpoint": " https://example.test/v1/ ",
                "api_key": "second-secret",
                "dim": 128,
                "timeout": 7,
                "num_retries": 0,
                "batch_size": 1,
            }
        )
    )
    assert original.identity == equivalent.identity
    assert original.identity.startswith("litellm:v1:")
    assert "first-secret" not in original.identity
    embedding_api.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"model": "openai/other-model"},
        {"endpoint": "https://other.test/v1"},
        {"dimensions": 3},
    ],
)
def test_litellm_identity_changes_with_vector_space(change):
    config = EmbeddingConfig(backend="litellm", dimensions=2)
    assert (
        LiteLLMEmbedder(config).identity
        != LiteLLMEmbedder(config.model_copy(update=change)).identity
    )


def test_validate_embedding_converts_to_float32():
    vector = validate_embedding(np.array([3, 4], dtype=np.float64), 2)
    assert vector.dtype == np.float32
    np.testing.assert_array_equal(vector, [3, 4])


@pytest.mark.parametrize(
    "vector",
    [
        1,
        [[1, 2]],
        [1],
        [],
        [1, np.nan],
        [1, np.inf],
        [1, -np.inf],
        [1, 1e100],
        ["not", "numbers"],
    ],
)
def test_validate_embedding_rejects_invalid_vectors(vector):
    with pytest.raises(EmbeddingMismatchError):
        validate_embedding(vector, 2)


def test_litellm_normalizes_vectors_and_preserves_input_order(embedding_api):
    embedding_api.return_value = {
        "data": [
            {"index": 1, "embedding": [0, 5]},
            {"index": 0, "embedding": [3, 4]},
        ]
    }
    embedder = LiteLLMEmbedder(
        EmbeddingConfig(
            backend="litellm",
            model=" openai/test ",
            endpoint=" https://example.test/v1/ ",
            dimensions=2,
        )
    )
    vectors = embedder.embed_batch(["first", "second"])
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [0, 1]])
    assert all(vector.dtype == np.float32 for vector in vectors)
    assert embedding_api.call_args.kwargs["input"] == ["first", "second"]
    assert embedding_api.call_args.kwargs["model"] == "openai/test"
    assert embedding_api.call_args.kwargs["api_base"] == "https://example.test/v1"
    assert embedding_api.call_args.kwargs["dimensions"] == 2


def test_litellm_infers_dimension_once_without_changing_identity(embedding_api):
    embedding_api.return_value = _response([3, 4])
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    identity = embedder.identity
    assert embedder.dim == 2
    assert embedder.dim == 2
    embedding_api.assert_called_once()
    assert embedding_api.call_args.kwargs["input"] == [" "]
    assert "dimensions" not in embedding_api.call_args.kwargs
    assert embedder.identity == identity


def test_litellm_empty_batch_does_not_probe(embedding_api):
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    assert embedder.embed_batch([]) == []
    assert embedder._dim is None
    embedding_api.assert_not_called()


@pytest.mark.parametrize("count", [0, 1, 3])
def test_litellm_rejects_wrong_response_count_without_learning_dimension(embedding_api, count):
    embedding_api.return_value = _response(*([[1, 2]] * count))
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    with pytest.raises(EmbeddingMismatchError, match="exactly 2"):
        embedder.embed_batch(["first", "second"])
    assert embedder._dim is None


@pytest.mark.parametrize("dimensions", [None, 2])
@pytest.mark.parametrize("invalid", [[1, 2, 3], [[1, 2]], [np.nan, 1], [np.inf, 1]])
def test_litellm_rejects_invalid_vectors_without_learning_dimension(
    embedding_api, dimensions, invalid
):
    embedding_api.return_value = _response([1, 2], invalid)
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dimensions=dimensions))
    with pytest.raises(EmbeddingMismatchError):
        embedder.embed_batch(["first", "second"])
    assert embedder._dim == dimensions


@pytest.mark.parametrize("dimensions", [None, 2])
def test_litellm_rejects_a_changed_dimension(embedding_api, dimensions):
    embedding_api.return_value = _response([3, 4])
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dimensions=dimensions))
    embedder.embed("valid")
    embedding_api.return_value = _response([1, 2, 3])
    with pytest.raises(EmbeddingMismatchError, match="dimension 2"):
        embedder.embed("changed")
    assert embedder.dim == 2


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (_response([1, 2, 3]), EmbeddingMismatchError),
        (RuntimeError("gateway failed"), RuntimeError),
    ],
)
def test_litellm_does_not_learn_dimension_from_a_partially_failed_batch(
    embedding_api, failure, error
):
    embedding_api.side_effect = [_response([1, 2]), failure]
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", batch_size=1))
    with pytest.raises(error):
        embedder.embed_batch(["first", "second"])
    assert embedder._dim is None

    embedding_api.side_effect = None
    embedding_api.return_value = _response([1, 2, 3])
    assert embedder.embed("retry").shape == (3,)
    assert embedder.dim == 3


@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [0, None], [0, 1.0], [0, True]])
def test_litellm_rejects_indices_that_do_not_match_inputs(embedding_api, indices):
    embedding_api.return_value = {
        "data": [{"index": index, "embedding": [1, 2]} for index in indices]
    }
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    with pytest.raises(EmbeddingMismatchError, match="indices"):
        embedder.embed_batch(["first", "second"])
    assert embedder._dim is None


def test_litellm_accepts_ordered_responses_without_indices(embedding_api):
    embedding_api.return_value = {"data": [{"embedding": [3, 4]}]}
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    np.testing.assert_allclose(embedder.embed("text"), [0.6, 0.8])


@pytest.mark.parametrize("value", [np.finfo(np.float32).max, np.finfo(np.float32).tiny])
def test_litellm_normalizes_finite_vectors_without_overflow_or_underflow(embedding_api, value):
    embedding_api.return_value = _response([value, value])
    embedder = LiteLLMEmbedder(EmbeddingConfig(backend="litellm"))
    np.testing.assert_allclose(embedder.embed("text"), [np.sqrt(0.5), np.sqrt(0.5)])
