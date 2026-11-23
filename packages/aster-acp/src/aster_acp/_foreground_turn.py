# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completion, cancellation, and checkpoint ownership for ACP foreground turns."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from aster.errors.storage import SerializationError
from aster.events import EventBase
from aster.interactive import AgentMessage
from aster_acp._steering import SteeringInbox

if TYPE_CHECKING:
    from aster_cli.sessions import SessionHandle

    from aster.agent import Agent
    from aster_acp._prompt import PromptProgress
    from aster_acp.dispatcher import InteractiveSessionDispatcher
    from aster_acp.event_bridge import ACPEventBridge

logger = logging.getLogger(__name__)


class InvalidTurnIdError(ValueError):
    """The requested turn cannot start accepting input."""


class ForegroundTurn:
    """One prompt's reply boundary, with synchronous success acknowledgement."""

    def __init__(
        self,
        agent: Agent,
        handle: SessionHandle,
        steering: SteeringInbox,
        progress: PromptProgress,
        turn_id: str,
        previous_turn_count: int,
    ) -> None:
        self.progress = progress
        self._agent = agent
        self._handle = handle
        self._steering = steering
        self._turn_id = turn_id
        self._previous_turn_count = previous_turn_count
        self._prompt_task = asyncio.current_task()
        self._host_reply_id: str | None = None
        self._completion_attempted = False
        self._can_save_current = True

    def complete(self) -> None:
        """Seal successful input and capture a host reply before the caller next awaits."""
        self._steering.finish("turn_finished")
        if not self._completion_attempted and self._has_recorded_turn():
            # Host-rendered slash replies never enter the model dispatcher.
            self._checkpoint_reply(self._host_reply_id)

    def _has_recorded_turn(self) -> bool:
        return self._handle.info.turn_count > self._previous_turn_count

    def _remember_host_reply(self, event: EventBase) -> None:
        if isinstance(event, AgentMessage):
            try:
                if asyncio.current_task() is self._prompt_task:
                    self._host_reply_id = event.id
            except RuntimeError:
                # Agent worker threads are not this prompt's caller.
                pass

    def _checkpoint_reply(self, reply_id: str | None) -> None:
        self._completion_attempted = True
        try:
            self._handle.checkpoint_completed(
                self._agent, completed_message_id=reply_id, turn_id=self._turn_id
            )
        except SerializationError:
            # The outcome is durable even when its state cannot be serialized;
            # never retry using a later reply or the current state.
            logger.warning(
                "Could not checkpoint completed conversation %s", self._handle.id, exc_info=True
            )
        except BaseException:
            # Without durable completion metadata, current would look like a
            # legacy restore point and could make a partial reply navigable.
            self._can_save_current = False
            raise

    def _finalize(self) -> None:
        if not self._completion_attempted and self._has_recorded_turn():
            self._handle.checkpoint_interrupted(turn_id=self._turn_id)
        if self._can_save_current:
            try:
                self._handle.save_current_snapshot(self._agent)
            except SerializationError:
                logger.warning(
                    "Could not save conversation state %s", self._handle.id, exc_info=True
                )


class ForegroundTurns:
    """Own turn lifetimes inside the session runtime's exclusive turn claim.

    The inbox seals a committed result before asynchronous worker teardown.
    The dispatcher separately checkpoints immediately after publishing its reply.
    These synchronous boundaries must not move into asynchronous cleanup.
    """

    def __init__(
        self,
        agent: Agent,
        handle: SessionHandle,
        dispatcher: InteractiveSessionDispatcher,
        bridge: ACPEventBridge,
    ) -> None:
        self._agent = agent
        self._handle = handle
        self._dispatcher = dispatcher
        self._bridge = bridge
        self._steering = SteeringInbox(agent, handle, bridge.publish)
        # Keep the inbox itself in scope: its interceptor checks control identity.
        dispatcher.steering = self._steering
        self._cancel_lock = asyncio.Lock()
        self._cancel_complete = asyncio.Event()
        self._cancel_complete.set()

    @asynccontextmanager
    async def prompt(
        self, turn_id: str, started_at: float, backend: str
    ) -> AsyncIterator[ForegroundTurn]:
        """Track one claimed prompt and finalize its state before transport flushing."""
        checkpoint_id = None
        try:
            checkpoint_id = self._handle.checkpoint(self._agent)
        except SerializationError:
            # Unsupported state must not prevent an otherwise valid prompt.
            logger.warning("Could not checkpoint conversation %s", self._handle.id, exc_info=True)
        self._handle.invalidate_current_snapshot(checkpoint_id=checkpoint_id)
        previous_turn_count = self._handle.info.turn_count
        async with self._bridge.prompt(started_at, backend) as progress:
            turn = ForegroundTurn(
                self._agent, self._handle, self._steering, progress, turn_id, previous_turn_count
            )
            unsubscribe = self._agent.event_manager.on("AgentMessage", turn._remember_host_reply)
            previous_completed = self._dispatcher.on_turn_completed
            self._dispatcher.on_turn_completed = turn._checkpoint_reply
            try:
                try:
                    self._steering.begin(turn_id)
                except ValueError as exc:
                    raise InvalidTurnIdError(str(exc)) from exc
                self._cancel_complete.clear()
                yield turn
            finally:
                self._dispatcher.on_turn_completed = previous_completed
                unsubscribe()
                # Successful and cancelled turns already sealed their inbox.
                # Any remaining input belongs to an interrupted prompt.
                self._steering.finish("interrupted")
                turn._finalize()

    def steer(self, turn_id: str, message_id: str, text: str) -> dict[str, Any]:
        return self._steering.accept(turn_id, message_id, text)

    async def cancel(self) -> None:
        """Seal input and finish cancellation cleanup before releasing its waiter."""
        async with self._cancel_lock:
            try:
                self._steering.finish("cancelled", "Stopped at your request.")
                if await self._dispatcher.cancel():
                    await self._bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
                    await self._bridge.flush()
            finally:
                self._cancel_complete.set()

    async def wait_cancelled(self) -> None:
        """Wait for dispatch, background jobs, and tool updates to finish cancelling."""
        await self._cancel_complete.wait()

    def close(self) -> None:
        self._steering.close()
