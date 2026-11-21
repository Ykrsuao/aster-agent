# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remember successful model selections through the ACP and startup boundaries."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from acp import RequestError
from aster_acp.server import CodingACPAdapter
from aster_cli._tui_model import resolve_tui_model

from aster.skill_registry import SkillRegistry
from aster.unifiedllm import FakeLLMClient, registry
from aster.unifiedllm.reasoning import ReasoningConfig


class _Client:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


class _Model(FakeLLMClient):
    def __init__(self, alias: str) -> None:
        super().__init__(strict_exhaustion=True)
        self.model = f"wire-{alias}"
        self._context_window = 131_072
        self.config["max_tokens"] = 4096
        self._reasoning_config = ReasoningConfig(
            levels={
                "low": {"reasoning_effort": "low", "max_tokens": 8192},
                "high": {"reasoning_effort": "high", "max_tokens": 16_384},
            },
            default="low",
        )
        self.closed = False

    async def refresh_model_info(self) -> None:
        pass

    async def aclose(self) -> None:
        self.closed = True
        await super().aclose()


class _Models:
    def __init__(self) -> None:
        self.created: list[_Model] = []
        self.factory = _Model

    def __call__(self, alias: str) -> _Model:
        llm = self.factory(alias)
        self.created.append(llm)
        return llm


@pytest.fixture
def environment(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    user = tmp_path / "user"
    user.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("ASTER_USER_DIR", str(user))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(project))
    monkeypatch.setenv("ASTER_SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    for name in ("ASTER_MODEL", "ASTER_SETTINGS", "ASTER_LLM_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(SkillRegistry, "discovered", lambda self: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(
        registry,
        "MODELS",
        {"alpha": {"model_name": "wire-alpha"}, "beta": {"model_name": "wire-beta"}},
    )
    (project / "settings.yaml").write_text("tui:\n  default_model: alpha\n", encoding="utf-8")
    (project / "llm_config.yaml").write_text(
        "models:\n  alpha:\n    model_name: wire-alpha\n  beta:\n    model_name: wire-beta\n",
        encoding="utf-8",
    )
    models = _Models()
    return workspace, project / "settings.yaml", models


@asynccontextmanager
async def _adapter(models: _Models, *, model: str | None = "alpha"):
    def initial_model():
        if model is None:
            raise ValueError("Choose a model before starting a conversation")
        return models(model)

    adapter = CodingACPAdapter(initial_model, model_factory=models, model=model)
    client = _Client()
    adapter.on_connect(client)  # type: ignore[arg-type]
    await adapter.initialize(1, aster={"runtimeControls": True})
    try:
        async with asyncio.timeout(15):
            yield adapter, client
    finally:
        await adapter.close()


async def _runtime(adapter: CodingACPAdapter, session_id: str):
    return await adapter.ext_method("aster/session/runtime", {"sessionId": session_id})


async def _configure(adapter: CodingACPAdapter, session_id: str, **changes):
    return await adapter.ext_method("aster/session/configure", {"sessionId": session_id, **changes})


def _reject_settings_replace(monkeypatch, settings: Path) -> None:
    replace = os.replace

    def reject(source, target, *args, **kwargs):
        if Path(target).resolve() == settings.resolve():
            raise PermissionError("test settings destination is read-only")
        return replace(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", reject)


async def test_switching_model_remembers_the_alias_for_the_next_start(environment):
    workspace, _, models = environment
    adapter = CodingACPAdapter(lambda: models("alpha"), model_factory=models)
    adapter.on_connect(_Client())  # type: ignore[arg-type]
    await adapter.initialize(1, aster={"runtimeControls": True})
    try:
        session = await adapter.new_session(str(workspace))
        changed = await adapter.ext_method(
            "aster/session/configure", {"sessionId": session.session_id, "model": "beta"}
        )
        assert changed["model"] == "beta"
        assert changed["modelName"] == "wire-beta"
    finally:
        await adapter.close()

    assert resolve_tui_model(workspace) == "beta"
    assert all(llm.call_count == 0 for llm in models.created)


@pytest.mark.parametrize("with_effort", [False, True])
async def test_unknown_startup_wire_id_is_not_saved_as_a_model_alias(environment, with_effort):
    workspace, settings, models = environment
    before_file = settings.read_bytes()
    adapter = CodingACPAdapter(lambda: models("alpha"), model_factory=models)
    adapter.on_connect(_Client())  # type: ignore[arg-type]
    await adapter.initialize(1, aster={"runtimeControls": True})
    try:
        session = await adapter.new_session(str(workspace))
        initial = await _runtime(adapter, session.session_id)
        assert initial["model"] == "wire-alpha"
        assert initial["modelName"] == "wire-alpha"
        changes = {"model": initial["model"]}
        if with_effort:
            changes["reasoningLevel"] = "high"

        confirmed = await _configure(adapter, session.session_id, **changes)

        assert confirmed["model"] == "wire-alpha"
        assert confirmed["reasoningSelection"] == ("high" if with_effort else None)
        assert settings.read_bytes() == before_file
        assert resolve_tui_model(workspace) == "alpha"
        created = await adapter.new_session(str(workspace))
        next_runtime = await _runtime(adapter, created.session_id)
        assert next_runtime["model"] == "wire-alpha"
        assert next_runtime["modelName"] == "wire-alpha"
        assert [llm.model for llm in models.created] == ["wire-alpha", "wire-alpha"]

        changed = await _configure(adapter, session.session_id, model="beta")
        assert changed["model"] == "beta"
        assert changed["modelName"] == "wire-beta"
        assert resolve_tui_model(workspace) == "beta"
        remembered = await adapter.new_session(str(workspace))
        next_runtime = await _runtime(adapter, remembered.session_id)
        assert next_runtime["model"] == "beta"
        assert next_runtime["modelName"] == "wire-beta"
    finally:
        await adapter.close()

    assert all(llm.call_count == 0 for llm in models.created)


async def test_remembered_model_is_used_for_new_loaded_and_restarted_sessions(environment):
    workspace, _, models = environment
    async with _adapter(models) as (adapter, _):
        original = await adapter.new_session(str(workspace))
        assert (await _runtime(adapter, original.session_id))["model"] == "alpha"
        await _configure(adapter, original.session_id, model="beta")

        created = await adapter.new_session(str(workspace))
        assert (await _runtime(adapter, created.session_id))["model"] == "beta"
        await adapter.close_session(original.session_id)
        await adapter.load_session(cwd=str(workspace), session_id=original.session_id)
        assert (await _runtime(adapter, original.session_id))["model"] == "beta"

    selected = resolve_tui_model(workspace)
    assert selected == "beta"
    async with _adapter(models, model=selected) as (restarted, _):
        await restarted.load_session(cwd=str(workspace), session_id=original.session_id)
        state = await _runtime(restarted, original.session_id)
        assert state["model"] == "beta"
        assert state["modelName"] == "wire-beta"
    assert all(llm.call_count == 0 for llm in models.created)


async def test_model_change_does_not_replace_another_workspaces_startup_factory(environment):
    workspace, _, models = environment
    other = workspace.parent / "other-workspace"
    other.mkdir()
    async with _adapter(models) as (adapter, _):
        first = await adapter.new_session(str(workspace))
        await _configure(adapter, first.session_id, model="beta")
        second = await adapter.new_session(str(other))
        assert (await _runtime(adapter, second.session_id))["model"] == "alpha"
        third = await adapter.new_session(str(workspace))
        assert (await _runtime(adapter, third.session_id))["model"] == "beta"


async def test_explicitly_selecting_current_alias_remembers_it_and_keeps_its_effort(environment):
    workspace, _, models = environment
    async with _adapter(models, model="beta") as (adapter, _):
        session = await adapter.new_session(str(workspace))
        assert (await _runtime(adapter, session.session_id))["model"] == "beta"
        await _configure(adapter, session.session_id, reasoningLevel="high")
        assert resolve_tui_model(workspace) == "alpha"

        configured = await _configure(adapter, session.session_id, model="beta")

        assert configured["model"] == "beta"
        assert configured["modelName"] == "wire-beta"
        assert configured["reasoningSelection"] == "high"
        assert resolve_tui_model(workspace) == "beta"
        assert len(models.created) == 1


async def test_effort_mode_and_approval_changes_do_not_rewrite_the_default(environment):
    workspace, settings, models = environment
    before = settings.read_bytes()
    async with _adapter(models, model="beta") as (adapter, _):
        session = await adapter.new_session(str(workspace))
        await _configure(adapter, session.session_id, reasoningLevel="high")
        await _configure(adapter, session.session_id, approval="session")
        await _configure(adapter, session.session_id, mode="plan")
        state = await _runtime(adapter, session.session_id)
        assert state["model"] == "beta"
        assert state["reasoningSelection"] == "high"
        assert state["mode"] == "plan"
        assert settings.read_bytes() == before
        assert resolve_tui_model(workspace) == "alpha"


@pytest.mark.parametrize(
    "change", [{"model": "missing"}, {"model": "beta", "reasoningLevel": "max"}]
)
async def test_invalid_model_or_effort_does_not_change_saved_or_running_settings(
    environment, change
):
    workspace, settings, models = environment
    before_file = settings.read_bytes()
    async with _adapter(models) as (adapter, _):
        session = await adapter.new_session(str(workspace))
        before_runtime = await _runtime(adapter, session.session_id)
        with pytest.raises(RequestError):
            await _configure(adapter, session.session_id, mode="plan", **change)
        assert await _runtime(adapter, session.session_id) == before_runtime
        assert settings.read_bytes() == before_file
        assert resolve_tui_model(workspace) == "alpha"
        assert not models.created[0].closed
        assert all(candidate.closed for candidate in models.created[1:])


@pytest.mark.parametrize("selected", ["alpha", "beta"])
async def test_save_failure_rolls_back_combined_model_effort_and_mode_changes(
    environment, monkeypatch, selected
):
    workspace, settings, models = environment
    if selected == "alpha":
        settings.write_text("tui:\n  default_model: beta\n", encoding="utf-8")
    before_file = settings.read_bytes()
    async with _adapter(models) as (adapter, _):
        session = await adapter.new_session(str(workspace))
        before_runtime = await _runtime(adapter, session.session_id)
        _reject_settings_replace(monkeypatch, settings)
        with pytest.raises(RequestError):
            await _configure(
                adapter, session.session_id, model=selected, reasoningLevel="high", mode="plan"
            )
        assert await _runtime(adapter, session.session_id) == before_runtime
        assert settings.read_bytes() == before_file
        assert not models.created[0].closed
        assert all(candidate.closed for candidate in models.created[1:])
        created = await adapter.new_session(str(workspace))
        assert (await _runtime(adapter, created.session_id))["model"] == "alpha"


async def test_invalid_candidate_reply_budget_does_not_save_or_commit_the_model(environment):
    workspace, settings, models = environment
    before_file = settings.read_bytes()

    class InvalidReplyCapModel(_Model):
        def __init__(self, alias: str) -> None:
            super().__init__(alias)
            self.config["max_tokens"] = -1

    async with _adapter(models) as (adapter, _):
        session = await adapter.new_session(str(workspace))
        before_runtime = await _runtime(adapter, session.session_id)
        models.factory = InvalidReplyCapModel
        with pytest.raises(RequestError):
            await _configure(adapter, session.session_id, model="beta")
        assert await _runtime(adapter, session.session_id) == before_runtime
        assert settings.read_bytes() == before_file
        assert not models.created[0].closed
        assert models.created[-1].closed


@pytest.mark.parametrize("operation", ["new", "load"])
async def test_initial_picker_remembers_only_after_session_preparation(environment, operation):
    workspace, settings, models = environment
    previous_id = None
    if operation == "load":
        async with _adapter(models) as (original, _):
            previous_id = (await original.new_session(str(workspace))).session_id
    before_file = settings.read_bytes()
    async with _adapter(models, model=None) as (adapter, _):
        count = len(models.created)
        selected = await adapter.ext_method("aster/models/select", {"model": "beta"})
        assert selected["model"] == "beta"
        assert len(models.created) == count
        assert settings.read_bytes() == before_file
        if previous_id is None:
            session_id = (await adapter.new_session(str(workspace))).session_id
        else:
            await adapter.load_session(cwd=str(workspace), session_id=previous_id)
            session_id = previous_id
        assert (await _runtime(adapter, session_id))["model"] == "beta"
        assert resolve_tui_model(workspace) == "beta"


async def test_failed_initial_model_preparation_does_not_save_the_pending_alias(environment):
    workspace, settings, models = environment
    before_file = settings.read_bytes()

    class UnavailableModel(_Model):
        async def refresh_model_info(self) -> None:
            raise ValueError("metadata unavailable")

    models.factory = UnavailableModel
    async with _adapter(models, model=None) as (adapter, _):
        await adapter.ext_method("aster/models/select", {"model": "beta"})
        with pytest.raises((ValueError, RequestError)):
            await adapter.new_session(str(workspace))
        assert settings.read_bytes() == before_file
        assert resolve_tui_model(workspace) == "alpha"
        assert models.created and all(llm.closed for llm in models.created)
        assert (await adapter.list_sessions(cwd=str(workspace))).sessions == []


@pytest.mark.parametrize("operation", ["new", "load"])
async def test_failed_initial_save_closes_the_prepared_runtime(environment, monkeypatch, operation):
    workspace, settings, models = environment
    previous_id = None
    if operation == "load":
        async with _adapter(models) as (original, _):
            previous_id = (await original.new_session(str(workspace))).session_id
    before_file = settings.read_bytes()
    async with _adapter(models, model=None) as (adapter, _):
        await adapter.ext_method("aster/models/select", {"model": "beta"})
        _reject_settings_replace(monkeypatch, settings)
        with pytest.raises((ValueError, RequestError)):
            if previous_id is None:
                await adapter.new_session(str(workspace))
            else:
                await adapter.load_session(cwd=str(workspace), session_id=previous_id)
        assert settings.read_bytes() == before_file
        assert resolve_tui_model(workspace) == "alpha"
        assert models.created[-1].model == "wire-beta"
        assert models.created[-1].closed
        if previous_id is None:
            assert (await adapter.list_sessions(cwd=str(workspace))).sessions == []
        else:
            with pytest.raises(RequestError):
                await _runtime(adapter, previous_id)


async def test_cancelling_candidate_preparation_preserves_running_and_saved_model(environment):
    workspace, settings, models = environment
    before_file = settings.read_bytes()
    refreshing = asyncio.Event()

    class WaitingModel(_Model):
        async def refresh_model_info(self) -> None:
            refreshing.set()
            await asyncio.Event().wait()

    async with _adapter(models) as (adapter, _):
        session = await adapter.new_session(str(workspace))
        before_runtime = await _runtime(adapter, session.session_id)
        models.factory = WaitingModel
        change = asyncio.create_task(_configure(adapter, session.session_id, model="beta"))
        try:
            await asyncio.wait_for(refreshing.wait(), 2)
            change.cancel()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert await _runtime(adapter, session.session_id) == before_runtime
            assert settings.read_bytes() == before_file
            assert not models.created[0].closed
            assert models.created[-1].closed
        finally:
            change.cancel()
            await asyncio.gather(change, return_exceptions=True)


async def test_cancelling_old_client_cleanup_keeps_the_committed_model_saved(environment):
    workspace, _, models = environment
    closing = asyncio.Event()
    release = asyncio.Event()

    class WaitingCloseModel(_Model):
        async def aclose(self) -> None:
            if self.model == "wire-alpha":
                closing.set()
                try:
                    await release.wait()
                finally:
                    await super().aclose()
            else:
                await super().aclose()

    models.factory = WaitingCloseModel
    async with _adapter(models) as (adapter, client):
        session = await adapter.new_session(str(workspace))
        change = asyncio.create_task(_configure(adapter, session.session_id, model="beta"))
        try:
            await asyncio.wait_for(closing.wait(), 2)
            assert resolve_tui_model(workspace) == "beta"
            assert (await _runtime(adapter, session.session_id))["model"] == "beta"
            change.cancel()
            with pytest.raises(asyncio.CancelledError):
                await change
            assert resolve_tui_model(workspace) == "beta"
            created = await adapter.new_session(str(workspace))
            assert (await _runtime(adapter, created.session_id))["model"] == "beta"
        finally:
            release.set()
            change.cancel()
            await asyncio.gather(change, return_exceptions=True)

    states = [
        update.field_meta["aster"]["runtime"]
        for session_id, update in client.updates
        if session_id == session.session_id
        and update.field_meta
        and "runtime" in update.field_meta.get("aster", {})
    ]
    assert states[-1]["model"] == "beta"
