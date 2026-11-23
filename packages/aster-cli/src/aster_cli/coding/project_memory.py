# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional, host-owned memory with a fixed project boundary."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from aster import workspace_trust
from aster.paths import get_user_dir
from aster_cli.coding.settings import disable_project_memory, load_project_memory_settings

_DATABASE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_DEVICE_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}
_INSTRUCTIONS = "project_memory_instructions"
_RECALLED = "project_memory_recalled"


def _database_name(value: object) -> str:
    if not isinstance(value, str) or not _DATABASE_NAME.fullmatch(value) or value in _DEVICE_NAMES:
        raise ValueError("Memory database names use 1–64 lowercase letters, digits, '_' or '-'.")
    return value


def _remote_budget_guard(agent: Any = None) -> None:
    # A5 is optional in older hosts; do not hide import errors inside that module.
    try:
        from aster.task_budget import TaskBudgetError, current_task_budget
    except ModuleNotFoundError as exc:
        if exc.name != "aster.task_budget":
            raise
        return
    if current_task_budget() is not None or getattr(agent, "_task_budget", None) is not None:
        raise TaskBudgetError(
            "Task budgets do not yet account for remote embeddings; use the hashing embedder."
        )


class _GuardedEmbedder:
    def __init__(
        self, delegate: Any, check: Callable[[], None], *, remote: bool, agent: Any
    ) -> None:
        self._delegate, self._check, self._remote = delegate, check, remote
        self._agent = agent

    @property
    def identity(self) -> str:
        return self._delegate.identity

    @property
    def dim(self) -> int:
        return self._delegate.dim

    def embed(self, text: str) -> Any:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> Any:
        self._check()
        if self._remote:
            _remote_budget_guard(self._agent)
        return self._delegate.embed_batch(texts)


class ProjectMemory:
    """Rebuilt by the host after restoring a conversation, never snapshotted."""

    __nosnapshot__ = True

    def __init__(self, agent: Any, workspace: str | Path, *, session_ref: str | None = None):
        self.agent = agent
        self.root = Path(workspace).expanduser().resolve()
        key = hashlib.sha256(os.path.normcase(str(self.root)).encode("utf-8")).hexdigest()
        self.directory = get_user_dir("project-memory").resolve() / key
        self.session_ref = session_ref
        self._settings: dict[str, Any] = {}
        self._skill: Any = None
        self._closed = False
        self.state = "disabled"
        self.message = (
            "Project memory is off. Enable coding.project_memory in this project's settings."
        )
        agent._project_memory = self
        self.reload()

    def _database_path(self, database: object) -> Path:
        path = self.directory / f"{_database_name(database)}.sqlite"
        if self.directory.resolve() != self.directory or any(
            candidate.resolve() != candidate
            for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        ):
            raise ValueError("Project memory paths must stay inside their own project directory.")
        return path

    @property
    def path(self) -> Path:
        return self._database_path(self._settings.get("database", "main"))

    def _trusted(self) -> None:
        if self._closed or not workspace_trust.is_trusted(self.root):
            raise ValueError("Project memory requires an open session in a trusted workspace.")

    def _require_enabled(self) -> None:
        self._trusted()
        if not self._settings.get("enabled"):
            raise ValueError("Project memory is disabled.")
        if load_project_memory_settings(self.root) != self._settings:
            raise ValueError("Project memory settings changed; use /memory reload.")

    def _embedding_config(self) -> Any:
        from aster_memory import EmbeddingConfig

        raw = self._settings.get("embedding", {"backend": "hashing"})
        if not isinstance(raw, dict):
            raise ValueError("project_memory.embedding must be a mapping.")
        values = dict(raw)
        alias = values.pop("alias", None)
        if "api_key" in values:
            raise ValueError(
                "Use a saved embedding alias or api_key_env; do not put keys in settings."
            )
        if alias is not None:
            from aster.layered_config import layered_paths
            from aster.unifiedllm.registry import load_model_entries

            entries = load_model_entries(
                *layered_paths(
                    "llm_config.yaml", "ASTER_LLM_CONFIG", project_dir=self.root / ".aster"
                )
            )
            entry = entries.get(alias) if isinstance(alias, str) else None
            if not entry or entry.get("client_type") != "embedding":
                raise ValueError("Choose an explicitly saved alias with client_type: embedding.")
            if {"model", "endpoint", "api_key_env"}.intersection(values):
                raise ValueError(
                    "Set model, endpoint and api_key_env on the saved embedding alias."
                )
            values.update(
                model=entry.get("model"),
                endpoint=entry.get("endpoint") or entry.get("api_base"),
                api_key_env=entry.get("api_key_env"),
            )
            values.setdefault("backend", "litellm")
        env_name = values.pop("api_key_env", None)
        if set(values) - set(EmbeddingConfig.model_fields):
            raise ValueError("Unknown project memory embedding setting.")
        if values.get("backend", "hashing") == "litellm":
            if not isinstance(values.get("model"), str) or not values["model"].strip():
                raise ValueError("Remote embeddings require an explicit saved alias or model.")
            if type(values.get("dimensions")) is not int or values["dimensions"] < 1:
                raise ValueError(
                    "Set embedding.dimensions explicitly; startup does not probe a model."
                )
            if env_name is not None:
                from aster.unifiedllm.registry import resolve_api_key_from_config

                key = resolve_api_key_from_config("project memory", {"api_key_env": env_name})
                if key is None:
                    raise ValueError(
                        "The configured embedding credential environment variable is missing."
                    )
                values["api_key"] = key
        try:
            return EmbeddingConfig.model_validate(values)
        except ValueError:
            raise ValueError("Invalid project memory embedding configuration.") from None

    def _embedder(self, *, explicit: bool = False) -> Any:
        from aster_memory.embeddings import get_embedder

        config = self._embedding_config()
        return _GuardedEmbedder(
            get_embedder(config),
            self._trusted if explicit else self._require_enabled,
            remote=config.backend == "litellm",
            agent=self.agent,
        )

    def _mount(self) -> None:
        from aster_memory import MemoryConfig, MemoryManager, MemoryStore
        from aster_memory.memory_skill import MemorySkill

        from aster.skill import Skill

        owner = self
        embedder = self._embedder()

        class ProjectMemoryStore(MemoryStore):
            @contextmanager
            def transaction(self) -> Iterator[None]:
                # Held managers and cached engines must also respect an external opt-out.
                owner._require_enabled()
                with super().transaction():
                    yield
                    owner._require_enabled()

        class ProjectMemoryManager(MemoryManager):
            def _make_store(self, agent: Any) -> MemoryStore:
                return ProjectMemoryStore(
                    owner.path,
                    vector_config=self.config.vector,
                    embedding_dim=self.embedder.dim,
                    embedding_identity=self.embedder.identity,
                )

        class ProjectMemorySkill(MemorySkill):
            def attach(self, agent: Any) -> None:
                Skill.attach(self, agent)
                self._mgr = ProjectMemoryManager.install(
                    agent, config=self._config, embedder=embedder
                )
                self._mgr.session_ref = owner.session_ref

            def _memory_or_raise(self) -> Any:
                owner._require_enabled()
                return super()._memory_or_raise()

        config = MemoryConfig(
            enabled=True,
            path=str(self.path),
            owner="Project",
            api_prefix="self.memory.",
            instruct_block_key=_INSTRUCTIONS,
            spontaneous={"context_block_key": _RECALLED, "inject_cadence": "every_turn"},
            write={"on_events": (), "write_episodic": False},
            reflection={"enabled": False},
        )
        self._skill = ProjectMemorySkill(config)
        self.agent.skills.register("aster.memory", self._skill)
        self.agent.skills.activate(["aster.memory"])

    def _clear_recall(self) -> None:
        self.agent.context_manager.pop(_RECALLED, None)

    def _unmount(self) -> None:
        skill, self._skill = self._skill, None
        if skill is not None:
            self.agent.skills.unregister("aster.memory", instance=skill)
            # A failed attach can leave the attribute before registration completed.
            if getattr(self.agent, "memory", None) is skill:
                del self.agent.memory
            skill.detach()
        self.agent.context_manager.pop(_INSTRUCTIONS, None)
        self._clear_recall()

    def reload(self) -> dict[str, Any]:
        if self._closed:
            raise ValueError("Project memory is closed.")
        self._unmount()
        try:
            self._settings = load_project_memory_settings(self.root)
            if not self._settings.get("enabled"):
                self.state = "disabled"
                self.message = "Project memory is off. Set coding.project_memory.enabled: true, then /memory reload."
                return self.snapshot()
            if set(self._settings) - {"enabled", "database", "embedding"}:
                raise ValueError("Unknown coding.project_memory setting.")
            _ = self.path
            self._mount()
            self.state, self.message = (
                "ready",
                "Project memory is on. Notes are shared only with this project.",
            )
        except ImportError:
            self._unmount()
            self.state, self.message = (
                "unavailable",
                "Install the optional aster[memory] package to use project memory.",
            )
        except Exception as exc:
            self._unmount()
            self.state = "blocked"
            # Provider exceptions can contain request text, endpoints or credentials.
            if isinstance(exc, ValueError):
                self.message = str(exc)
            else:
                self.message = "Project memory could not open. Check its settings or rebuild into a new database."
        return self.snapshot()

    def refresh(self) -> None:
        if not self._closed:
            try:
                changed = load_project_memory_settings(self.root) != self._settings
            except ValueError:
                changed = True
            if changed:
                self.reload()

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state, "message": self.message, "workspace": str(self.root)}

    def inspect(self, memory_id: str = "") -> list[dict[str, Any]]:
        from aster_memory import MemoryStore

        self._trusted()
        if not self.path.is_file():
            return []
        with closing(MemoryStore(self.path)) as store:
            rows = store.all_memories(include_archived=True)
            if memory_id:
                rows = [row for row in rows if row.id == memory_id]
            return [row.model_dump(mode="json") for row in rows[:100]]

    def delete(self, memory_id: str) -> None:
        from aster_memory import MemoryStore

        self._trusted()
        if not memory_id or not self.path.is_file():
            raise ValueError("Choose an existing memory ID from /memory list.")
        with closing(MemoryStore(self.path)) as store:
            if store.get(memory_id) is None:
                raise ValueError("Choose an existing memory ID from /memory list.")
            store.delete(memory_id)
        self._clear_recall()

    def rebuild(self, database: str) -> Path:
        from aster_memory import MemoryStore

        self._trusted()
        destination = self._database_path(database)
        if not self.path.is_file():
            raise ValueError("There is no project memory database to rebuild.")
        with closing(MemoryStore(self.path)) as store:
            return store.rebuild_embeddings(self._embedder(explicit=True), destination)

    def command(self, action: str = "status", argument: str = "") -> str:
        """Inspect project memory, delete a note, persistently turn it off, or rebuild it."""
        if (
            action in {"delete", "off", "reload", "rebuild"}
            and self.agent.queue_manager.running_handles()
        ):
            raise ValueError("Stop or finish background jobs before changing project memory.")
        if action == "status" and not argument:
            return (
                self.message
                + "\nUse /memory list, view ID, delete ID, off, reload, or rebuild NEW_DATABASE."
            )
        if action in {"list", "view"}:
            if (action == "list" and argument) or (action == "view" and not argument):
                raise ValueError("Use /memory list or /memory view ID.")
            rows = self.inspect(argument)
            return (
                json.dumps(rows, ensure_ascii=False, indent=2)
                if rows
                else "No saved project memories."
            )
        if action == "delete" and argument:
            self.delete(argument)
            return "Deleted the project memory. It will no longer be retrieved."
        if action == "off" and not argument:
            try:
                disable_project_memory(self.root)
            finally:
                self._unmount()
                self.state = "disabled"
                self.message = "Project memory is off for this session."
            self._settings["enabled"] = False
            self.message = "Project memory is off for this project, including new sessions. Saved notes are kept."
            return self.message
        if action == "reload" and not argument:
            self.reload()
            return self.message
        if action == "rebuild" and argument:
            self.rebuild(argument)
            return (
                f"Created memory database {argument}. The source is unchanged. After stopping other writers, "
                f"set coding.project_memory.database: {argument} and use /memory reload to switch."
            )
        raise ValueError(
            "Use /memory [status|list|view ID|delete ID|off|reload|rebuild NEW_DATABASE]."
        )

    def close(self) -> None:
        self._closed = True
        self._unmount()
        self.state, self.message = "closed", "Project memory is closed."
