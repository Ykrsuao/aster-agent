# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for reflection / offline consolidation."""

import time
from contextlib import contextmanager

import pytest
from aster_memory.config import ForgetPolicy, ReflectionPolicy
from aster_memory.embeddings import HashingEmbedder
from aster_memory.reflection import ReflectionEngine
from aster_memory.schema import EdgeType, Memory, MemoryType
from aster_memory.store import MemoryStore

DAY = 24 * 3600.0


@pytest.fixture
def emb():
    return HashingEmbedder(dim=256)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


def _add(store, emb, content, **kw):
    m = Memory(content=content, **kw)
    return store.add(m, emb.embed(m.embedding_text()))


def _engine(store, emb, **reflect_kw):
    return ReflectionEngine(store, emb, ReflectionPolicy(**reflect_kw), ForgetPolicy())


def test_merge_folds_duplicates(store, emb):
    a = _add(store, emb, "deploy with make ship", type=MemoryType.INFO)
    b = _add(store, emb, "deploy with make ship", type=MemoryType.INFO)  # identical -> cos 1.0
    eng = _engine(store, emb, merge_threshold=0.95)
    report = eng.consolidate()
    assert report.merged >= 1
    # exactly one survives in the active set
    active = store.all_memories()
    assert len(active) == 1
    survivor = active[0]
    assert survivor.reinforcement_count >= 1
    # the merged-away one is archived with a provenance edge from the survivor
    assert store.get(b.id).archived or store.get(a.id).archived


@pytest.mark.parametrize("change", ["content", "delete", "archive", "touch"])
def test_merge_rereads_sources_before_archiving(store, emb, monkeypatch, change):
    original = _add(store, emb, "deploy with make ship", created_at=1)
    duplicate = _add(store, emb, "deploy with make ship", created_at=2)
    transaction = store.transaction
    changed = False

    @contextmanager
    def change_before_transaction():
        nonlocal changed
        if not changed:
            changed = True
            if change == "delete":
                store.delete(duplicate.id)
            elif change == "archive":
                store.archive(duplicate.id)
            elif change == "content":
                duplicate.content = "completely unrelated bananas"
                store.add(duplicate, emb.embed(duplicate.content))
            else:
                with store.edit(duplicate.id) as current:
                    current.touch()
        with transaction():
            yield

    monkeypatch.setattr(store, "transaction", change_before_transaction)
    merged = _engine(store, emb, merge_threshold=0.95)._merge_duplicates()
    if change == "touch":
        assert merged == 1
        assert store.get(original.id).strength == 3
        assert store.get(duplicate.id).archived
    else:
        assert merged == 0
        assert store.get(original.id).strength == 1
        if change == "content":
            assert not store.get(duplicate.id).archived


def test_merge_rolls_back_archives_when_survivor_save_fails(store, emb, monkeypatch):
    originals = [_add(store, emb, "deploy with make ship") for _ in range(2)]

    def fail(*args, **kwargs):
        raise RuntimeError("save failed")

    monkeypatch.setattr(store, "save", fail)
    with pytest.raises(RuntimeError, match="save failed"):
        _engine(store, emb, merge_threshold=0.95)._merge_duplicates()
    assert {m.id for m in store.all_memories()} == {m.id for m in originals}
    assert len(store.knn(emb.embed(originals[0].content), 2)) == 2


def test_distinct_memories_are_not_merged(store, emb):
    _add(store, emb, "completely different topic alpha beta")
    _add(store, emb, "unrelated subject gamma delta epsilon")
    eng = _engine(store, emb, merge_threshold=0.95)
    report = eng.consolidate()
    assert report.merged == 0
    assert len(store.all_memories()) == 2


def test_edge_formation_links_related_memories(store, emb):
    # Share some tokens -> cosine in the [edge, merge) band.
    _add(store, emb, "deploy ship release alpha")
    _add(store, emb, "deploy ship rollback beta")
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.2, max_edges_per_node=3)
    report = eng.consolidate()
    assert report.edges_added >= 1
    assert any(m.edges for m in store.all_memories())


def test_rescore_importance_clamped_and_salience_boosts(store, emb):
    m = _add(store, emb, "a salient lesson", importance=3.0, salience=1.0)
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    eng.consolidate()
    out = store.get(m.id)
    assert 0.0 <= out.importance <= 10.0
    assert out.importance > 3.0  # salience pushed importance up


@pytest.mark.parametrize("operation", ["_form_edges", "_rescore_importance"])
def test_reflection_rechecks_owner_before_mutating(store, emb, monkeypatch, operation):
    original = _add(store, emb, "deploy ship release alpha", owner="mine", importance=3)
    _add(store, emb, "deploy ship rollback beta", owner="mine")
    engine = ReflectionEngine(
        store,
        emb,
        ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.2),
        ForgetPolicy(),
        owner="mine",
    )
    edit = store.edit

    @contextmanager
    def change_owner_before_edit(id, **kwargs):
        if id == original.id:
            with edit(id) as current:
                current.owner = "another-agent"
        with edit(id, **kwargs) as current:
            yield current

    monkeypatch.setattr(store, "edit", change_owner_before_edit)
    getattr(engine, operation)()
    latest = store.get(original.id)
    assert latest.owner == "another-agent"
    assert latest.importance == original.importance
    assert latest.edges == []


def test_reflection_prunes_decayed_memory(store, emb):
    now = time.time()
    keep = _add(store, emb, "fresh relevant note", importance=5.0)
    _add(
        store,
        emb,
        "ancient trivia nobody needs",
        importance=1.0,
        created_at=now - 200 * DAY,
        last_accessed_at=now - 200 * DAY,
    )
    eng = ReflectionEngine(
        store, emb, ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.999), ForgetPolicy()
    )
    report = eng.consolidate()
    assert report.pruned >= 1
    assert keep.id in {m.id for m in store.all_memories()}


def test_abstraction_with_reasoner_creates_skill_linked_to_episodes(store, emb):
    e1 = _add(store, emb, "episode: tried X, it worked", type=MemoryType.EPISODE)
    e2 = _add(store, emb, "episode: tried X again, worked", type=MemoryType.EPISODE)

    def reasoner(episodes):
        assert len(episodes) >= 2
        return [Memory(content="skill: X reliably works", type=MemoryType.SKILL)]

    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    report = eng.consolidate(reasoner=reasoner)
    assert report.created == 1
    skills = [m for m in store.all_memories() if m.type == MemoryType.SKILL]
    assert len(skills) == 1
    targets = {e.target_id for e in skills[0].edges if e.type == EdgeType.DERIVED_FROM}
    assert {e1.id, e2.id} <= targets


def test_reconsolidate_archives_stale_and_keeps_current(store, emb):
    older = _add(store, emb, "my 5k personal best is 27:00", type=MemoryType.INFO)
    _add(store, emb, "my 5k personal best is now 25:50", type=MemoryType.INFO)

    def reconciler(cluster):
        # cluster is oldest -> newest; supersede everything but the latest
        outdated = [c.id for c in cluster[:-1]]
        keep = Memory(
            content="current 5k personal best: 25:50", type=MemoryType.INFO, importance=8.0
        )
        return keep, outdated

    eng = ReflectionEngine(
        store,
        emb,
        ReflectionPolicy(merge_threshold=0.999, edge_threshold=0.999, recon_threshold=0.3),
        ForgetPolicy(),
    )
    report = eng.consolidate(reconciler=reconciler)
    assert report.reconciled >= 1 and report.superseded >= 1
    active = [m.content for m in store.all_memories()]
    assert any("25:50" in c for c in active)  # current value retained
    assert not any("27:00" in c for c in active)  # stale value archived
    assert store.get(older.id).archived is True


@pytest.mark.parametrize("failure", ["embedding", "save"])
def test_reconsolidation_keeps_originals_if_replacement_fails(store, emb, monkeypatch, failure):
    originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]
    engine = _engine(store, emb, recon_threshold=0)

    def fail(*args):
        raise TimeoutError("synthetic failure")

    monkeypatch.setattr(
        emb if failure == "embedding" else store, "embed" if failure == "embedding" else "add", fail
    )
    with pytest.raises(TimeoutError, match="synthetic"):
        engine._reconsolidate(
            lambda cluster: (Memory(content="deployment host gamma"), [m.id for m in cluster])
        )
    assert {m.id for m in store.all_memories()} == {m.id for m in originals}


def test_reconsolidation_can_reuse_an_original_id(store, emb):
    originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]
    engine = _engine(store, emb, recon_threshold=0)
    replacement = Memory(id=originals[0].id, content="deployment host gamma")
    engine._reconsolidate(lambda cluster: (replacement, [m.id for m in cluster]))
    active = store.all_memories()
    assert len(active) == 1
    assert active[0].content == replacement.content
    assert all(edge.target_id != replacement.id for edge in active[0].edges)


@pytest.mark.parametrize("phase", ["reconciler", "embedding"])
@pytest.mark.parametrize("change", ["content", "delete", "archive", "owner"])
def test_reconsolidation_discards_result_after_source_changes(
    tmp_path, emb, monkeypatch, phase, change
):
    store = MemoryStore(tmp_path / "shared.sqlite")
    other = MemoryStore(tmp_path / "shared.sqlite")
    try:
        originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]
        replacement = Memory(content="deployment host gamma")
        embed = emb.embed

        def change_source():
            if change == "delete":
                other.delete(originals[0].id)
            elif change == "archive":
                other.archive(originals[0].id)
            else:
                with other.edit(originals[0].id) as current:
                    if change == "content":
                        current.content = "deployment host delta"
                    else:
                        current.owner = "another-agent"

        def embed_after_change(text):
            change_source()
            return embed(text)

        def reconcile(cluster):
            if phase == "reconciler":
                change_source()
            return replacement, [m.id for m in cluster]

        if phase == "embedding":
            monkeypatch.setattr(emb, "embed", embed_after_change)
        assert _engine(store, emb, recon_threshold=0)._reconsolidate(reconcile) == (0, 0)
        assert store.get(replacement.id) is None
        assert not store.get(originals[1].id).archived
        if change == "delete":
            assert store.get(originals[0].id) is None
        elif change == "archive":
            assert store.get(originals[0].id).archived
        else:
            assert not store.get(originals[0].id).archived
            assert getattr(store.get(originals[0].id), change) == (
                "deployment host delta" if change == "content" else "another-agent"
            )
    finally:
        other.close()
        store.close()


def test_reconsolidation_allows_reconciler_to_mutate_input_copy(store, emb):
    originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]

    def reconcile(cluster):
        cluster[0].content = "deployment host gamma"
        return cluster[0], [m.id for m in cluster]

    assert _engine(store, emb, recon_threshold=0)._reconsolidate(reconcile) == (1, 1)
    assert store.get(originals[0].id).content == "deployment host gamma"
    assert not store.get(originals[0].id).archived


def test_reconsolidation_rolls_back_replacement_and_archives(store, emb, monkeypatch):
    originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]
    replacement = Memory(content="deployment host gamma")
    archive = store.archive

    def fail_after_archive(id):
        archive(id)
        raise RuntimeError("archive failed")

    monkeypatch.setattr(store, "archive", fail_after_archive)
    with pytest.raises(RuntimeError, match="archive failed"):
        _engine(store, emb, recon_threshold=0)._reconsolidate(
            lambda cluster: (replacement, [m.id for m in cluster])
        )
    assert {m.id for m in store.all_memories()} == {m.id for m in originals}
    assert store.get(replacement.id) is None
    assert {mid for mid, _ in store.knn(emb.embed(replacement.content), 3)} == {
        m.id for m in originals
    }


def test_reconsolidation_does_not_replace_a_record_outside_its_cluster(store, emb):
    originals = [_add(store, emb, f"deployment host {name}") for name in ("alpha", "beta")]
    unrelated = store.add(Memory(content="unrelated record"))
    replacement = Memory(id=unrelated.id, content="deployment host gamma")
    result = _engine(store, emb, recon_threshold=0)._reconsolidate(
        lambda cluster: (replacement, [m.id for m in cluster])
    )
    assert result == (0, 0)
    assert store.get(unrelated.id).content == unrelated.content
    assert {m.id for m in store.all_memories()} == {m.id for m in [*originals, unrelated]}


def test_reconciler_noop_when_nothing_archived(store, emb):
    _add(store, emb, "fact one alpha", type=MemoryType.INFO)
    _add(store, emb, "fact two beta", type=MemoryType.INFO)
    eng = _engine(store, emb, merge_threshold=0.999, edge_threshold=0.999)
    report = eng.consolidate(reconciler=lambda cluster: (None, []))
    assert report.reconciled == 0 and report.superseded == 0
    assert len(store.all_memories()) == 2


def test_consolidate_empty_store(store, emb):
    eng = _engine(store, emb)
    report = eng.consolidate()
    assert report.merged == 0 and report.pruned == 0
