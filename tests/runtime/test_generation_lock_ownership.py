# SPDX-License-Identifier: Apache-2.0
"""Generation locks belong to an agent session, including its awaited children."""

import asyncio

from aster import Agent, strategy
from aster.strategies import GenerationStrategy
from aster.unifiedllm import FakeLLMClient


class _RunBody(GenerationStrategy):
    async def execute(self, runtime, call):
        return await runtime.agent._body(call.kwargs["label"])


class _LockAgent(Agent, llm=FakeLLMClient()):
    @strategy(_RunBody())
    async def run(self, label: str) -> str: ...


async def test_another_agent_cannot_bypass_a_busy_child_lock():
    child = _LockAgent()
    parent = _LockAgent()
    started = asyncio.Event()
    release = asyncio.Event()
    nested = asyncio.Event()

    async def child_body(label):
        if label == "first":
            started.set()
            await release.wait()
        else:
            nested.set()
        return label

    async def parent_body(label):
        return await child.run(label)

    child._body = child_body
    parent._body = parent_body
    first = asyncio.create_task(child.run("first"))
    await started.wait()
    second = asyncio.create_task(parent.run("second"))
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not nested.is_set()
    finally:
        release.set()
        assert await asyncio.gather(first, second) == ["first", "second"]
        await child.aclose()
        await parent.aclose()


async def test_owned_agent_can_be_reentered_through_another_agent():
    first = _LockAgent()
    second = _LockAgent()

    async def first_body(label):
        if label == "outer":
            return await second.run("middle")
        return label

    async def second_body(label):
        return await asyncio.create_task(first.run("inner"))

    first._body = first_body
    second._body = second_body
    try:
        assert await asyncio.wait_for(first.run("outer"), 2) == "inner"
    finally:
        await first.aclose()
        await second.aclose()


async def test_detached_child_loses_ownership_when_its_session_finishes():
    agent = _LockAgent()
    start_detached = asyncio.Event()
    started = asyncio.Event()
    release = asyncio.Event()
    entered = asyncio.Event()
    detached = None

    async def later():
        await start_detached.wait()
        return await agent.run("detached")

    async def body(label):
        nonlocal detached
        if label == "spawn":
            detached = asyncio.create_task(later())
        elif label == "busy":
            started.set()
            await release.wait()
        else:
            entered.set()
        return label

    agent._body = body
    assert await agent.run("spawn") == "spawn"
    busy = asyncio.create_task(agent.run("busy"))
    await started.wait()
    start_detached.set()
    try:
        for _ in range(10):
            await asyncio.sleep(0)
        assert not entered.is_set()
    finally:
        release.set()
        assert await busy == "busy"
        assert await detached == "detached"
        await agent.aclose()
