# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline project-memory lifecycle checks, shared by native Windows and Linux."""

import builtins
import json
import sqlite3
import sys
from contextlib import closing
from types import ModuleType

import pytest
import yaml
from aster_cli.coding.agent import CodingAgent
from aster_cli.coding.project_memory import ProjectMemory, _remote_budget_guard
from aster_cli.coding.settings import load_project_memory_settings
from aster_cli.coding.slash_commands import CodingSlashCommandRegistry
from aster_cli.sessions import SessionStore
from aster_memory import HashingEmbedder, MemoryStore

from aster import workspace_trust
from aster.unifiedllm import FakeLLMClient


def _configure(root, *, enabled=True, database="main", embedding=None):
    path = root / ".aster" / "settings.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "coding": {
            "project_memory": {
                "enabled": enabled,
                "database": database,
                "embedding": embedding or {"backend": "hashing", "dim": 64},
            }
        },
        "unrelated": {"keep": "保留"},
    }
    path.write_text(yaml.safe_dump(settings, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("ASTER_USER_DIR", str(home / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "unrelated-project"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.delenv("ASTER_LLM_CONFIG", raising=False)


@pytest.fixture
async def make_agent():
    agents = []

    def make(root, **kwargs):
        root.mkdir(parents=True, exist_ok=True)
        agent = CodingAgent(
            llm=FakeLLMClient(strict_exhaustion=True),
            cwd=root,
            libs_dir=root / ".aster" / "libs",
            **kwargs,
        )
        agents.append(agent)
        return agent

    yield make
    for agent in reversed(agents):
        if hasattr(agent, "shell"):
            await agent.close()


async def test_default_off_ignores_user_and_environment_opt_in(tmp_path, monkeypatch, make_agent):
    settings = _configure(tmp_path / "user-config")
    monkeypatch.setenv("ASTER_SETTINGS", str(settings))
    root = tmp_path / "workspace"
    agent = make_agent(root)
    memory = ProjectMemory(agent, root)
    assert memory.state == "disabled"
    assert not hasattr(agent, "memory")
    assert not hasattr(agent, "_memory")
    assert not memory.directory.exists()
    assert "memory" in {command.name for command in CodingSlashCommandRegistry(agent).commands()}


async def test_untrusted_project_never_opens_its_memory_settings(tmp_path, monkeypatch, make_agent):
    root = tmp_path / "workspace"
    _configure(root)
    agent = make_agent(root)
    monkeypatch.delenv("ASTER_TRUST_ALL_WORKSPACES", raising=False)
    with workspace_trust.enforced(True):
        memory = ProjectMemory(agent, root)
        assert memory.state == "disabled"
        assert not memory.directory.exists()
        with pytest.raises(ValueError, match="trusted"):
            memory.command("off")
        assert load_project_memory_settings(root) == {}


async def test_save_close_reopen_recall_delete_and_persistently_disable(tmp_path, make_agent):
    root = tmp_path / "workspace"
    settings = _configure(root)
    store = SessionStore(tmp_path / "sessions")
    with store.create(session_id="first") as handle:
        agent = make_agent(root, storage=handle.storage)
        memory = ProjectMemory(agent, root, session_ref=handle.id)
        assert memory.state == "ready", memory.message
        memory_id = agent.memory.remember("Deploy this project with the blue release target.")
        manager = agent._memory
        agent.context_manager.set_dynamic("project_memory_recalled", value="obsolete cache")
        handle.storage.save_current_snapshot(agent)
        await agent.close()
        with pytest.raises(sqlite3.ProgrammingError):
            manager.store.count()
    with store.open("first") as handle:
        agent = make_agent(root, storage=handle.storage)
        assert handle.storage.restore_current_snapshot(agent)
        memory = ProjectMemory(agent, root, session_ref=handle.id)
        assert memory.state == "ready", memory.message
        assert "project_memory_recalled" not in agent.context_manager
        assert [item.id for item in agent.memory.recall("blue release")] == [memory_id]
        commands = CodingSlashCommandRegistry(agent)
        listed = await commands.invoke("memory", "list")
        assert not listed.output_to_agent
        assert memory_id in listed.text
        await commands.invoke("memory", f"delete {memory_id}")
        assert agent.memory.search("blue release") == []
        await commands.invoke("memory", "off")
        assert not hasattr(agent, "memory")
        assert not hasattr(agent, "_memory")
        assert "project_memory_instructions" not in agent.context_manager
        saved = yaml.safe_load(settings.read_text(encoding="utf-8"))
        assert saved["coding"]["project_memory"]["enabled"] is False
        assert saved["unrelated"] == {"keep": "保留"}
    reopened = make_agent(root)
    memory = ProjectMemory(reopened, root)
    assert memory.state == "disabled"
    assert not hasattr(reopened, "memory")
    assert memory.inspect() == []


async def test_memory_close_failure_still_closes_other_agent_resources(
    tmp_path, monkeypatch, make_agent
):
    from unittest.mock import AsyncMock

    class FailingMemory:
        def close(self):
            raise RuntimeError("offline memory close failure")

    agent = make_agent(tmp_path / "workspace")
    agent._project_memory = FailingMemory()
    shutdown = AsyncMock(wraps=agent.queue_manager.shutdown)
    close_client = AsyncMock(wraps=agent.llm.aclose)
    monkeypatch.setattr(agent.queue_manager, "shutdown", shutdown)
    monkeypatch.setattr(agent.llm, "aclose", close_client)
    with pytest.raises(RuntimeError, match="offline memory close failure"):
        await agent.close()
    shutdown.assert_awaited_once()
    close_client.assert_awaited_once()


async def test_cross_session_project_isolation_and_external_delete(
    tmp_path, monkeypatch, make_agent
):
    roots = [tmp_path / name for name in ("one", "two")]
    for root in roots:
        _configure(root)
    first, other, same = (make_agent(roots[index]) for index in (0, 1, 0))
    owners = [
        ProjectMemory(agent, root)
        for agent, root in zip((first, other, same), (roots[0], roots[1], roots[0]), strict=True)
    ]
    memory_id = first.memory.remember("The hummingbird project uses its cobalt backend.")
    assert other.memory.recall("hummingbird cobalt") == []
    assert [item.id for item in same.memory.recall("hummingbird cobalt")] == [memory_id]
    monkeypatch.chdir(roots[1])
    first.cwd = roots[1]
    assert owners[0].root == roots[0].resolve()
    owners[0].delete(memory_id)
    assert same.memory.recall("hummingbird cobalt") == []
    assert owners[0].directory == owners[2].directory != owners[1].directory


async def test_off_stops_existing_handles_and_refresh_clears_tools(tmp_path, make_agent):
    root = tmp_path / "workspace"
    _configure(root)
    first, second = make_agent(root), make_agent(root)
    owner, other = ProjectMemory(first, root), ProjectMemory(second, root)
    held_skill = second.memory
    memory_id = first.memory.remember("A durable note before shutdown.")
    second_id = first.memory.remember("The separate second note keeps its relationships.")
    held_manager = second._memory
    original = other.inspect()
    owner.command("off")
    for action in (
        lambda: held_skill.remember("must not be written"),
        lambda: held_manager.update(memory_id, importance=9.0),
        lambda: held_manager.forget(memory_id),
        lambda: held_manager.associate(memory_id, second_id),
        lambda: held_manager.deref(f"memory:{memory_id}"),
    ):
        with pytest.raises(ValueError, match="settings changed"):
            action()
    assert other.inspect() == original
    assert held_manager.store.neighbors(memory_id) == []
    other.refresh()
    assert not hasattr(second, "memory")
    assert len(other.inspect()) == 2
    assert "must not" not in json.dumps(other.inspect())


@pytest.mark.parametrize(
    "database", ["../outside", "C:\\outside", "other/name", "con", "COM1", ".", ""]
)
async def test_database_names_cannot_change_the_project_boundary(tmp_path, make_agent, database):
    root = tmp_path / "workspace"
    _configure(root, database=database)
    owner = ProjectMemory(make_agent(root), root)
    assert owner.state == "blocked"
    assert not owner.directory.exists()


@pytest.mark.parametrize("link_kind", ["directory", "database", "wal"])
async def test_linked_memory_paths_cannot_expose_another_project(tmp_path, make_agent, link_kind):
    source, target = tmp_path / "source", tmp_path / "target"
    _configure(source, enabled=False)
    _configure(target)
    source_agent, target_agent = make_agent(source), make_agent(target)
    owner, other = ProjectMemory(source_agent, source), ProjectMemory(target_agent, target)
    memory_id = target_agent.memory.remember("Only the target project may retrieve this note.")
    owner.directory.parent.mkdir(parents=True, exist_ok=True)
    if link_kind == "directory":
        link, destination = owner.directory, other.directory
    else:
        owner.directory.mkdir()
        link = owner.path if link_kind == "database" else owner.path.with_suffix(".sqlite-wal")
        destination = other.path
    try:
        link.symlink_to(destination, target_is_directory=link_kind == "directory")
    except OSError as exc:
        pytest.skip(f"Host does not permit test symlinks: {exc}")
    _configure(source)
    owner.reload()
    assert owner.state == "blocked"
    assert "own project directory" in owner.message
    assert not hasattr(source_agent, "memory")
    with pytest.raises(ValueError, match="own project directory"):
        owner.inspect()
    assert [item["id"] for item in other.inspect()] == [memory_id]


async def test_model_change_requires_explicit_rebuild_and_preserves_source(
    tmp_path, monkeypatch, make_agent
):
    root = tmp_path / "workspace"
    calls = []

    class Embedder(HashingEmbedder):
        def __init__(self, config):
            super().__init__(dim=config.dimensions)
            self.model = config.model

        @property
        def identity(self):
            return f"offline-model:{self.model}"

        def embed_batch(self, texts):
            calls.append((self.model, len(texts)))
            return super().embed_batch(texts)

    monkeypatch.setattr("aster_memory.embeddings.get_embedder", Embedder)
    _configure(root, embedding={"backend": "litellm", "model": "offline/one", "dimensions": 64})
    agent = make_agent(root)
    owner = ProjectMemory(agent, root)
    first = agent.memory.remember("Orchid deploys use the release job.")
    second = agent.memory.remember("Cobalt service stores its backups externally.")
    agent.memory.associate(first, second, "related")
    old_path = owner.path
    _configure(root, embedding={"backend": "litellm", "model": "offline/two", "dimensions": 64})
    owner.reload()
    assert owner.state == "blocked"
    assert {item["id"] for item in owner.inspect()} == {first, second}
    owner.rebuild("v2")
    assert owner.path == old_path
    with (
        closing(MemoryStore(old_path)) as old,
        closing(MemoryStore(owner.directory / "v2.sqlite")) as rebuilt,
    ):
        assert {row.id for row in old.all_memories()} == {first, second}
        assert {row.id for row in rebuilt.all_memories()} == {first, second}
        assert old.neighbors(first) == rebuilt.neighbors(first)
    _configure(
        root,
        database="v2",
        embedding={"backend": "litellm", "model": "offline/two", "dimensions": 64},
    )
    owner.reload()
    assert owner.state == "ready", owner.message
    assert first in {item.id for item in agent.memory.search("Orchid deploys")}
    assert calls


async def test_rebuild_refuses_a_destination_link_to_another_project(tmp_path, make_agent):
    root = tmp_path / "workspace"
    _configure(root)
    agent = make_agent(root)
    owner = ProjectMemory(agent, root)
    memory_id = agent.memory.remember("Keep the source project note.")
    external = tmp_path / "another-project.sqlite"
    external.write_bytes(b"external destination must not change")
    destination = owner.directory / "rebuilt.sqlite"
    try:
        destination.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Host does not permit test symlinks: {exc}")
    with pytest.raises(ValueError, match="own project directory"):
        owner.rebuild("rebuilt")
    assert [item["id"] for item in owner.inspect()] == [memory_id]
    assert external.read_bytes() == b"external destination must not change"


async def test_interrupted_rebuild_keeps_source_and_never_publishes_destination(
    tmp_path, monkeypatch, make_agent
):
    root = tmp_path / "workspace"
    _configure(root)
    agent = make_agent(root)
    owner = ProjectMemory(agent, root)
    memory_id = agent.memory.remember("The original durable note must survive migration.")
    original = owner.inspect()

    class Interrupted(HashingEmbedder):
        def embed_batch(self, texts):
            raise KeyboardInterrupt("offline interruption")

    monkeypatch.setattr("aster_memory.embeddings.get_embedder", lambda config: Interrupted(dim=128))
    with pytest.raises(KeyboardInterrupt, match="offline interruption"):
        owner.rebuild("v2")
    assert not (owner.directory / "v2.sqlite").exists()
    assert not list(owner.directory.glob(".*-rebuild-*"))
    assert owner.inspect() == original
    assert memory_id in {item.id for item in agent.memory.recall("original durable note")}


async def test_explicit_dimension_change_is_blocked_without_mixing_vectors(tmp_path, make_agent):
    root = tmp_path / "workspace"
    _configure(root)
    agent = make_agent(root)
    owner = ProjectMemory(agent, root)
    memory_id = agent.memory.remember("The stored vector has sixty-four dimensions.")
    _configure(root, embedding={"backend": "hashing", "dim": 128})
    owner.reload()
    assert owner.state == "blocked"
    assert not hasattr(agent, "memory")
    assert [item["id"] for item in owner.inspect()] == [memory_id]
    owner.delete(memory_id)
    assert owner.inspect() == []


async def test_saved_embedding_alias_uses_existing_registry_and_does_not_expose_credentials(
    tmp_path, monkeypatch, make_agent
):
    root = tmp_path / "workspace"
    _configure(root, embedding={"backend": "litellm", "alias": "notes", "dimensions": 64})
    (root / ".aster" / "llm_config.yaml").write_text(
        yaml.safe_dump(
            {
                "models": {
                    "notes": {
                        "client_type": "embedding",
                        "model": "offline/embedding",
                        "endpoint": "https://invalid.example/v1",
                        "api_key_env": "ASTER_TEST_EMBEDDING_KEY",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ASTER_TEST_EMBEDDING_KEY", "synthetic-test-key")
    captured = []

    def embedder(config):
        captured.append(config)
        return HashingEmbedder(dim=64)

    monkeypatch.setattr("aster_memory.embeddings.get_embedder", embedder)
    owner = ProjectMemory(make_agent(root), root)
    assert owner.state == "ready", owner.message
    assert captured[0].model == "offline/embedding"
    assert captured[0].api_key == "synthetic-test-key"
    assert "synthetic-test-key" not in json.dumps(owner.snapshot())
    owner.command("off")
    assert "synthetic-test-key" not in (root / ".aster" / "settings.yaml").read_text()


@pytest.mark.parametrize("missing", ["aster.task_budget", "budget_dependency"])
def test_budget_compatibility_only_allows_an_absent_host_module(monkeypatch, missing):
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "aster.task_budget":
            raise ModuleNotFoundError(f"No module named {missing}", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    if missing == "aster.task_budget":
        _remote_budget_guard()
    else:
        with pytest.raises(ModuleNotFoundError, match=missing):
            _remote_budget_guard()


@pytest.mark.parametrize("use_alias", [False, True])
@pytest.mark.parametrize("denied_lookup", [False, True])
async def test_remote_embeddings_refuse_unaccounted_task_budget_and_hashing_still_works(
    tmp_path, monkeypatch, make_agent, use_alias, denied_lookup
):
    budget = ModuleType("aster.task_budget")

    class BudgetError(RuntimeError):
        pass

    budget.TaskBudgetError = BudgetError
    budget.current_task_budget = lambda: None
    monkeypatch.setitem(sys.modules, "aster.task_budget", budget)
    called = []

    class Remote(HashingEmbedder):
        def embed_batch(self, texts):
            called.append(texts)
            return super().embed_batch(texts)

    monkeypatch.setattr("aster_memory.embeddings.get_embedder", lambda config: Remote(dim=64))
    root = tmp_path / "workspace"
    if use_alias:
        _configure(root, embedding={"alias": "notes", "dimensions": 64})
        (root / ".aster" / "llm_config.yaml").write_text(
            yaml.safe_dump(
                {"models": {"notes": {"client_type": "embedding", "model": "offline/model"}}}
            ),
            encoding="utf-8",
        )
    else:
        _configure(
            root, embedding={"backend": "litellm", "model": "offline/model", "dimensions": 64}
        )
    agent = make_agent(root)
    owner = ProjectMemory(agent, root)
    assert owner.state == "ready", owner.message
    memory_id = agent.memory.remember("A saved note before the budget was enabled.")
    called.clear()

    def active_budget():
        if denied_lookup:
            raise BudgetError("budget context unavailable")
        return object()

    budget.current_task_budget = active_budget
    monkeypatch.setattr("aster_memory.manager.derive_queries", lambda *args: ["saved note"])
    for action in (
        lambda: agent.memory.remember("must not dispatch"),
        lambda: agent.memory.recall("saved note"),
        lambda: agent._memory._on_before_turn(object()),
        lambda: owner.rebuild("rebuilt"),
    ):
        with pytest.raises(BudgetError, match="do not yet account|budget context unavailable"):
            action()
    assert not called
    assert [item["id"] for item in owner.inspect()] == [memory_id]
    assert not (owner.directory / "rebuilt.sqlite").exists()
    _configure(root)
    owner.reload()
    assert agent.memory.remember("Local hashing remains available.")
    assert called
