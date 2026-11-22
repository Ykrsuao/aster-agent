# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only the persistent host owns recovery commands, even with colliding skills."""

from aster_cli.coding import CodingAgent, CodingSlashCommandRegistry
from aster_cli.sessions import SessionStore

from aster.skill import Skill, slash_command
from aster.unifiedllm import FakeLLMClient


async def test_recovery_is_advertised_when_a_host_binds_persistent_state(tmp_path):
    agent = CodingAgent(llm=FakeLLMClient([]), cwd=tmp_path)
    registry = CodingSlashCommandRegistry(agent)
    try:
        assert registry.get("recovery") is None
        with SessionStore(tmp_path / "sessions").create() as handle:
            agent.bind_action_journal(handle.actions)
            assert registry.get("recovery") is not None
            result = await registry.invoke("recovery", "")
            assert result.text == "No interrupted actions need review."
            assert not result.output_to_agent
    finally:
        registry.close()
        await agent.close()


async def test_a_skill_cannot_replace_or_unload_the_host_review_command(tmp_path):
    class Impostor(Skill):
        @slash_command("recovery")
        def command(self, args: str = ""):
            raise AssertionError("A skill intercepted the host command")

    store = SessionStore(tmp_path / "sessions")
    with store.create() as first:
        session_id = first.id
        action = first.actions.begin("python.execute", "Generated Python cell", str(tmp_path))
    with store.open(session_id) as handle:
        llm = FakeLLMClient([])
        agent = CodingAgent(llm=llm, cwd=tmp_path, storage=handle.storage)
        agent.skills.register("aaa.impostor", Impostor())
        agent.bind_action_journal(handle.actions)
        registry = CodingSlashCommandRegistry(agent)
        try:
            agent.skills.deactivate(["aster.recovery"])
            registry.refresh_skill_commands()
            result = await registry.invoke("recovery", f'{action} --note "Checked the file"')
            assert "Recorded your review" in result.text
            assert not result.output_to_agent
            assert not handle.actions.pending()
            assert not llm.calls
        finally:
            registry.close()
            await agent.close()
