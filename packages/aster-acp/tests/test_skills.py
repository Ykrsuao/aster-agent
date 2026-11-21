# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The skill browser reads metadata without activating or importing capabilities."""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from acp import RequestError
from aster_acp.server import CodingACPAdapter

from aster.skill import Skill, TextSkill
from aster.unifiedllm import FakeLLMClient


class DocumentedSkill(Skill):
    """Inspect the repository.

    Read-only details for the browser, with 中文 preserved.
    """

    api_key = "SECRET_FIELD_MUST_NOT_LEAK"


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTER_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("ASTER_PROJECT_DIR", str(tmp_path / "project"))
    monkeypatch.delenv("ASTER_SETTINGS", raising=False)
    monkeypatch.setattr("aster.skill_registry.entry_points", lambda **kwargs: [])
    monkeypatch.setattr("aster_acp.server.load_coding_skills_dirs", lambda root: [])


@asynccontextmanager
async def session(tmp_path):
    llm = FakeLLMClient([], strict_exhaustion=True)
    adapter = CodingACPAdapter(lambda: llm)
    adapter.on_connect(AsyncMock())
    try:
        initialized = await adapter.initialize(1)
        created = await adapter.new_session(str(tmp_path))
        runtime = await adapter._get_runtime(created.session_id)
        yield adapter, created.session_id, runtime, llm, initialized
    finally:
        await adapter.close()


async def browse(adapter, session_id, **kwargs):
    return await adapter.ext_method("aster/session/skills", {"sessionId": session_id, **kwargs})


async def test_skill_browser_lists_text_python_mcp_and_unloaded_skills_without_loading(
    tmp_path, monkeypatch
):
    directory = tmp_path / "browser-skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: browser-skill\ndescription: Review visual layouts\n---\n"
        "Compare the narrow and wide terminal layouts.\n",
        encoding="utf-8",
    )
    async with session(tmp_path) as (adapter, session_id, runtime, llm, initialized):
        assert initialized.model_dump(by_alias=True)["_meta"]["aster"]["skills"] is True
        registry = runtime.value.agent.skills
        registry.register("cmd.browser-skill", TextSkill(path=directory))
        registry.register("local.documented", DocumentedSkill())
        registry.register("mcp.docs", DocumentedSkill())

        def unexpected_load():
            raise AssertionError("Browsing must not import a skill")

        entry_point = SimpleNamespace(value="optional_package:OptionalSkill", load=unexpected_load)
        registry._discovered["optional.unloaded"] = SimpleNamespace(
            name="optional.unloaded", category="optional", entry_point=entry_point
        )
        before = (registry.discovered(), registry.loaded(), registry.activated())
        result = await browse(adapter, session_id)
        rows = {row["name"]: row for row in result["skills"]}
        assert result["total"] == len(before[0])
        assert result["truncated"] is False
        assert rows["cmd.browser-skill"] == {
            "name": "cmd.browser-skill",
            "description": "Review visual layouts",
            "active": False,
            "loaded": True,
            "kind": "text",
        }
        assert rows["local.documented"]["description"] == "Inspect the repository."
        assert rows["mcp.docs"]["kind"] == "mcp"
        assert rows["aster.shell"]["active"] is True
        assert rows["optional.unloaded"]["loaded"] is False
        assert "not loaded" in rows["optional.unloaded"]["description"].lower()
        detail = (await browse(adapter, session_id, name="optional.unloaded"))["skill"]
        assert "optional_package:OptionalSkill" in detail["text"]
        text_detail = (await browse(adapter, session_id, name="cmd.browser-skill"))["skill"]
        assert "Compare the narrow and wide terminal layouts." in text_detail["text"]
        assert text_detail["truncated"] is False
        python_detail = (await browse(adapter, session_id, name="local.documented"))["skill"]
        assert "中文" in python_detail["text"]
        assert "SECRET_FIELD_MUST_NOT_LEAK" not in json.dumps(python_detail)
        assert before == (registry.discovered(), registry.loaded(), registry.activated())
        assert llm.call_count == 0


async def test_skill_browser_remains_available_during_a_foreground_turn(tmp_path):
    async with session(tmp_path) as (adapter, session_id, runtime, llm, _):
        async with runtime.turn():
            result = await asyncio.wait_for(browse(adapter, session_id), 1)
            detail = await asyncio.wait_for(browse(adapter, session_id, name="aster.repo"), 1)
            assert result["skills"]
            assert detail["skill"]["active"] is True
            assert runtime.busy
        assert llm.call_count == 0


@pytest.mark.parametrize("name", [None, "", " ", 3, [], {}, "x" * 257])
async def test_skill_browser_rejects_invalid_detail_names(tmp_path, name):
    async with session(tmp_path) as (adapter, session_id, _, llm, _):
        with pytest.raises(RequestError) as error:
            await browse(adapter, session_id, name=name)
        assert error.value.code == -32602
        assert llm.call_count == 0


async def test_skill_browser_rejects_unknown_skills_and_closed_sessions(tmp_path):
    async with session(tmp_path) as (adapter, session_id, runtime, _, _):
        with pytest.raises(RequestError) as error:
            await browse(adapter, session_id, name="../outside")
        assert error.value.code == -32002
        await runtime.close()
        with pytest.raises(RequestError) as error:
            await browse(adapter, session_id)
        assert error.value.code == -32002


async def test_skill_browser_bounds_documentation_and_list_results(tmp_path, monkeypatch):
    async with session(tmp_path) as (adapter, session_id, runtime, llm, _):
        registry = runtime.value.agent.skills
        registry.register("local.long", Skill(content="x" * 70_000))
        detail = (await browse(adapter, session_id, name="local.long"))["skill"]
        assert len(detail["text"]) == 65_536
        assert len(detail["description"]) <= 512
        assert detail["truncated"] is True
        monkeypatch.setattr(
            registry, "discovered", lambda: [f"optional.{n:04}" for n in range(2001)]
        )
        result = await browse(adapter, session_id)
        assert len(result["skills"]) == 2000
        assert result["total"] == 2001
        assert result["truncated"] is True
        assert llm.call_count == 0
