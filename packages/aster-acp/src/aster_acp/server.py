# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP adapter for the host-neutral ASTER coding agent."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.helpers import update_available_commands
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandInput,
    CloseSessionResponse,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    PromptCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionInfoUpdate,
    SessionListCapabilities,
    SetSessionModeResponse,
    SseMcpServer,
    UnstructuredCommandInput,
)
from acp.schema import (
    SessionInfo as ACPSessionInfo,
)
from aster_cli.coding.agent import CodingAgent
from aster_cli.coding.reply_strategy import CodingReplyStrategy
from aster_cli.coding.settings import load_coding_skills_dirs
from aster_cli.coding.slash_commands import CodingSlashCommand, CodingSlashCommandRegistry
from aster_cli.sessions import (
    InvalidSessionIdError,
    SessionCheckpointUnavailableError,
    SessionHandle,
    SessionNotFoundError,
    SessionStore,
)

from aster import workspace_trust
from aster.config import CodeActConfig
from aster.errors import GenerationError
from aster.mcp import MCPManager, MCPTool
from aster.paths import get_user_dir
from aster.runtime.sandbox.config import SandboxConfig
from aster.runtime.sandbox.errors import SandboxError
from aster.slash_dispatch import CoercionError
from aster.strategies.codeact import (
    MAX_ITERATIONS_MESSAGE,
    OUTPUT_TOKENS_EXHAUSTED_MESSAGE,
)
from aster.unifiedllm import UnifiedLLM
from aster_acp._attachments import ParsedPrompt, parse_prompt
from aster_acp._foreground_turn import ForegroundTurns, InvalidTurnIdError
from aster_acp._prompt import PromptProgress
from aster_acp._runtime import (
    SessionBusyError,
    SessionRuntime,
    SessionRuntimeClosedError,
    SessionRuntimePool,
)
from aster_acp._session_replay import record_activity, transcript_updates
from aster_acp._skills import skills_snapshot
from aster_acp._startup import mark, phase
from aster_acp._stdio import diagnostic_stdout, protocol_streams
from aster_acp.dispatcher import InteractiveSessionDispatcher
from aster_acp.event_bridge import ACPEventBridge
from aster_acp.runtime_controls import RuntimeControls

logger = logging.getLogger(__name__)

_SESSION_PAGE_SIZE = 50
_GENERATION_LIMIT_PREFIX = MAX_ITERATIONS_MESSAGE.partition("{")[0]


@dataclass(slots=True)
class _ACPSession:
    """Live resources owned by one ACP session runtime."""

    handle: SessionHandle
    agent: CodingAgent
    dispatcher: InteractiveSessionDispatcher
    bridge: ACPEventBridge
    commands: CodingSlashCommandRegistry
    foreground: ForegroundTurns = field(init=False)
    controls: RuntimeControls | None = None
    configuring: bool = False
    startup_warnings: tuple[str, ...] = ()
    trust_notice: str | None = None
    notification_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    commands_sent_on_prompt: bool = False
    unsubscribe_activity: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        self.foreground = ForegroundTurns(self.agent, self.handle, self.dispatcher, self.bridge)

    async def close(self) -> None:
        self.foreground.close()
        for task in self.notification_tasks:
            task.cancel()
        if self.notification_tasks:
            await asyncio.gather(*self.notification_tasks, return_exceptions=True)
        self.notification_tasks.clear()
        try:
            await self.bridge.close()
        finally:
            try:
                self.commands.close()
            finally:
                try:
                    await self.dispatcher.close()
                finally:
                    if self.unsubscribe_activity is not None:
                        self.unsubscribe_activity()
                    self.handle.close()


class CodingACPAdapter:
    def __init__(
        self,
        llm_factory: Callable[[], UnifiedLLM],
        *,
        model_factory: Callable[[str], UnifiedLLM] | None = None,
        model: str | None = None,
        sandbox: bool = False,
        sandbox_config: SandboxConfig | None = None,
        prepare_model: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._prepare_model = prepare_model
        self._model_preparation: asyncio.Task[None] | None = None
        self._llm_factory = llm_factory
        self._model_factory = model_factory
        self._startup_model = model
        self._pending_model: str | None = None
        self._workspace_models: dict[Path, str] = {}
        self._sandbox = sandbox
        self._sandbox_config = sandbox_config or SandboxConfig()
        self._client: Client | None = None
        self._stream_messages = False
        self._structured_progress = False
        self._sessions: SessionRuntimePool[_ACPSession] = SessionRuntimePool()

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        mark("initialize")
        del protocol_version, client_capabilities, client_info
        options = kwargs.get("aster")
        self._stream_messages = (
            isinstance(options, dict) and options.get("messageStreaming") is True
        )
        self._structured_progress = isinstance(options, dict) and (
            options.get("structuredProgress") is True or options.get("runtimeControls") is True
        )
        try:
            package_version = version("aster-acp")
        except PackageNotFoundError:
            package_version = "0.0.0"
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            field_meta={
                "aster": {
                    "conversationRewind": True,
                    "sessionTree": True,
                    "sessionTreeSearch": True,
                    "userSteering": True,
                    "runtimeControls": True,
                    "structuredProgress": True,
                    "skills": True,
                    "contextInspection": True,
                    "modelSelection": True,
                    "attachments": True,
                }
            },
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True, embedded_context=True),
                # McpCapabilities defaults to all-false, and a client that
                # honours the handshake then filters its HTTP/SSE servers out of
                # session/new — so the agent receives no MCP servers at all,
                # however the user configured them. _create_mcp_tools connects
                # both transports, so say so. `acp` stays off: it is unstable in
                # the spec and not implemented here.
                mcp_capabilities=McpCapabilities(http=True, sse=True),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
            auth_methods=[],
            agent_info=Implementation(
                name="aster-acp",
                title="Aster",
                version=package_version,
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        options = kwargs.get("aster")
        provisional = isinstance(options, dict) and options.get("prepare") is True
        root = self._validate_workspace(cwd, additional_directories)
        with phase("session_model"):
            llm, model = await self._new_llm(root)
        try:
            with phase("session_storage"):
                handle = self._store(root).create(
                    model=llm.model,
                    agent="CodingAgent",
                    working_directory=str(root),
                    origin="acp",
                    provisional=provisional,
                    check_same_thread=False,
                )
        except BaseException:
            await llm.aclose()
            raise
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            with phase("session_runtime"):
                runtime = await self._create_runtime(
                    handle, root, mcp_servers, llm=llm, model=model
                )
            self._remember_prepared_model(root, model)
        except BaseException:
            if runtime is not None:
                with suppress(KeyError):
                    await self._sessions.remove(handle.id, include_unavailable=True)
            else:
                handle.close()
            self._store(root).delete(handle.id)
            raise
        self._defer_bootstrap_updates(runtime.value)
        mark("session_ready")
        controls = runtime.value.controls
        return NewSessionResponse(
            session_id=handle.id, modes=controls.mode_state() if controls is not None else None
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        del kwargs
        root = self._validate_workspace(cwd, additional_directories)
        try:
            with phase("session_storage"):
                handle = self._store(root).open(session_id, check_same_thread=False)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise RequestError.resource_not_found(session_id) from None
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            with phase("session_model"):
                llm, model = await self._new_llm(root)
            with phase("session_runtime"):
                runtime = await self._create_runtime(
                    handle, root, mcp_servers, llm=llm, model=model, available=False
                )
            # After the replay, never during it: replay writes straight to the
            # client while the bridge pump drains bootstrap updates, so every
            # await here would otherwise let a commands update or an MCP warning
            # land in the middle of the restored conversation.
            with phase("session_replay"):
                await self._replay_session(handle)
            await self._sessions.publish(session_id)
            self._remember_prepared_model(root, model)
            self._defer_bootstrap_updates(runtime.value)
            mark("session_ready")
        except BaseException:
            if runtime is not None:
                with suppress(KeyError):
                    await self._sessions.remove(session_id, include_unavailable=True)
            else:
                handle.close()
            raise
        controls = runtime.value.controls
        return LoadSessionResponse(modes=controls.mode_state() if controls is not None else None)

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        options = kwargs.get("aster")
        include_history = isinstance(options, dict) and options.get("includeInputHistory") is True
        root = self._validate_workspace(cwd or str(Path.cwd()), None)
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError:
            raise RequestError.invalid_params(
                {"cursor": cursor, "reason": "Invalid cursor"}
            ) from None
        if offset < 0:
            raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})

        found = self._store(root).list(limit=offset + _SESSION_PAGE_SIZE + 1)
        page = found[offset : offset + _SESSION_PAGE_SIZE]
        sessions = [
            ACPSessionInfo(
                session_id=info.id,
                # This store belongs to the requested workspace. Historical
                # start metadata may name its location before a move or copy.
                cwd=str(root),
                title=info.title,
                updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
            )
            for info in page
        ]
        next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
        response = ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)
        if include_history and offset == 0:
            from aster_acp._input_history import recent_inputs

            history = await asyncio.to_thread(recent_inputs, self._store(root))
            response.field_meta = {"aster": {"inputHistory": history}}
        return response

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        del kwargs
        try:
            await self._sessions.remove(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None
        return CloseSessionResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """ASTER extensions; the ACP router removes the leading `_`."""
        if method == "aster/models/list":
            from aster.unifiedllm.connect.configuration import saved_models

            return {"models": await asyncio.to_thread(saved_models)}
        if method == "aster/models/select":
            from aster.unifiedllm.connect.configuration import saved_models

            model = params.get("model")
            if not isinstance(model, str) or model not in {item["id"] for item in saved_models()}:
                raise RequestError.invalid_params({"reason": "Choose a configured model."})
            self._llm_factory = lambda: self._make_model(model)
            self._startup_model = model
            self._pending_model = model
            return {"model": model}
        if method not in {
            "aster/session/checkpoints",
            "aster/session/rewind",
            "aster/session/tree",
            "aster/session/navigate",
            "aster/session/steer",
            "aster/session/runtime",
            "aster/session/configure",
            "aster/session/tasks",
            "aster/session/stop_task",
            "aster/session/context",
            "aster/session/skills",
        }:
            raise RequestError.method_not_found(f"_{method}")
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise RequestError.invalid_params({"reason": "sessionId must be a nonempty string"})
        if method == "aster/session/skills" and "name" in params:
            name = params["name"]
            if not isinstance(name, str) or not name.strip() or len(name) > 256:
                raise RequestError.invalid_params(
                    {"reason": "name must be a nonempty skill name of at most 256 characters"}
                )
        index = params.get("userMessageIndex")
        if method == "aster/session/rewind" and (type(index) is not int or index < 0):
            raise RequestError.invalid_params(
                {"reason": "userMessageIndex must be a non-negative integer"}
            )
        if method == "aster/session/tree" and "query" in params:
            query = params["query"]
            if not isinstance(query, str) or len(query) > 4096:
                raise RequestError.invalid_params(
                    {"reason": "query must be a string of at most 4096 characters"}
                )
        if method == "aster/session/context":
            for name in ("requestId", "category"):
                value = params.get(name)
                if not isinstance(value, str) or not value or len(value) > 128:
                    raise RequestError.invalid_params(
                        {"reason": f"{name} must be a nonempty string of at most 128 characters"}
                    )
            key = params.get("key")
            if key is not None and (not isinstance(key, str) or not key or len(key) > 120):
                raise RequestError.invalid_params(
                    {"reason": "key must be null or a nonempty string of at most 120 characters"}
                )
        if method == "aster/session/navigate":
            node_id = params.get("nodeId")
            if not isinstance(node_id, str) or not node_id or len(node_id) > 256:
                raise RequestError.invalid_params(
                    {"reason": "nodeId must be a nonempty string of at most 256 characters"}
                )
            expected = params.get("expectedCurrentNodeId")
            if "expectedCurrentNodeId" not in params or (
                expected is not None
                and (not isinstance(expected, str) or not expected or len(expected) > 256)
            ):
                raise RequestError.invalid_params(
                    {"reason": "expectedCurrentNodeId must be null or a nonempty node ID"}
                )
        runtime = await self._get_runtime(session_id)
        try:
            if method == "aster/session/configure" and params.keys() == {
                "sessionId",
                "permissionMode",
            }:
                async with runtime.access() as session:
                    if session.configuring:
                        raise RequestError.invalid_request(
                            {"reason": "Wait for the pending session settings change to finish."}
                        )
                    try:
                        return self._require_controls(session).set_permission_mode(
                            params["permissionMode"]
                        )
                    except ValueError as exc:
                        raise RequestError.invalid_params({"reason": str(exc)}) from exc
            if method in {
                "aster/session/runtime",
                "aster/session/tasks",
                "aster/session/stop_task",
                "aster/session/context",
                "aster/session/skills",
            }:
                async with runtime.access() as session:
                    if method == "aster/session/skills":
                        try:
                            return skills_snapshot(session.agent.skills, params.get("name"))
                        except KeyError:
                            raise RequestError.resource_not_found(params["name"]) from None
                    if method == "aster/session/runtime":
                        return self._require_controls(session).status()
                    if method == "aster/session/tasks":
                        return session.bridge.task_snapshot()
                    if method == "aster/session/context":
                        try:
                            return session.bridge.inspect_context(
                                params["requestId"], params["category"], params.get("key")
                            )
                        except ValueError as exc:
                            raise RequestError.invalid_params({"reason": str(exc)}) from exc
                    task_id = params.get("taskId")
                    if not isinstance(task_id, str) or not task_id or len(task_id) > 256:
                        raise RequestError.invalid_params(
                            {"reason": "taskId must be a nonempty string"}
                        )
                # Cancellation targets the captured job, not the foreground lock.
                try:
                    return await session.bridge.stop_task(task_id)
                except ValueError as exc:
                    raise RequestError.invalid_params({"reason": str(exc)}) from exc
            if method == "aster/session/steer":
                for key in ("turnId", "messageId", "text"):
                    value = params.get(key)
                    limit = 20_000 if key == "text" else 256
                    if not isinstance(value, str) or not value.strip() or len(value) > limit:
                        raise RequestError.invalid_params(
                            {
                                "reason": f"{key} must be a nonempty string of at most {limit} characters"
                            }
                        )
                async with runtime.access() as session:
                    try:
                        return session.foreground.steer(
                            params["turnId"], params["messageId"], params["text"]
                        )
                    except ValueError as exc:
                        raise RequestError.invalid_request({"reason": str(exc)}) from exc
            async with runtime.turn() as session:
                if method == "aster/session/configure":
                    if "permissionMode" in params and params.keys() - {
                        "sessionId",
                        "permissionMode",
                        "model",
                        "mode",
                        "approval",
                        "executionBackend",
                        "reasoningLevel",
                    }:
                        raise RequestError.invalid_params(
                            {"reason": "Unknown fields in permission configuration."}
                        )
                    changes = {}
                    for key in ("model", "mode", "approval", "permissionMode", "executionBackend"):
                        if key not in params:
                            continue
                        value = params[key]
                        if not isinstance(value, str) or not value or len(value) > 256:
                            raise RequestError.invalid_params(
                                {"reason": f"{key} must be a nonempty string"}
                            )
                        field_name = {
                            "permissionMode": "permission_mode",
                            "executionBackend": "execution_backend",
                        }.get(key, key)
                        changes[field_name] = value
                    if "reasoningLevel" in params:
                        level = params["reasoningLevel"]
                        if level is not None and (
                            not isinstance(level, str) or not level.strip() or len(level) > 256
                        ):
                            raise RequestError.invalid_params(
                                {"reason": "reasoningLevel must be null or a nonempty level name"}
                            )
                        changes["reasoning_level"] = level
                    session.configuring = True
                    try:
                        result = await self._require_controls(session).configure(**changes)
                    except ValueError as exc:
                        raise RequestError.invalid_params({"reason": str(exc)}) from exc
                    finally:
                        session.configuring = False
                    session.bridge.refresh_usage()
                    return result
                if method == "aster/session/checkpoints":
                    return {
                        "checkpoints": [
                            {"userMessageIndex": index, "prompt": prompt}
                            for index, prompt in session.handle.checkpoints()
                        ]
                    }
                if method in {"aster/session/tree", "aster/session/navigate"}:
                    try:
                        if method == "aster/session/tree":
                            return session.handle.session_tree(query=params.get("query"))
                        if session.agent.queue_manager.running_handles():
                            raise RequestError.invalid_request(
                                {
                                    "reason": "Stop or finish background jobs before switching branches."
                                }
                            )
                        target, prompt = session.handle.navigate_tree(
                            params["nodeId"], params["expectedCurrentNodeId"]
                        )
                    except SessionCheckpointUnavailableError as exc:
                        raise RequestError.invalid_params({"reason": str(exc)}) from exc
                    return {"sessionId": target, "prompt": prompt}
                try:
                    fork_id, prompt = session.handle.rewind(index)
                except SessionCheckpointUnavailableError as exc:
                    raise RequestError.invalid_params({"reason": str(exc)}) from exc
                return {"sessionId": fork_id, "prompt": prompt}
        except SessionBusyError:
            raise RequestError.invalid_request(
                {"sessionId": session_id, "reason": "Finish or cancel the current prompt first"}
            ) from None
        except SessionRuntimeClosedError:
            raise RequestError.resource_not_found(session_id) from None

    @staticmethod
    def _require_controls(session: _ACPSession) -> RuntimeControls:
        if session.controls is None:
            raise RequestError.internal_error({"reason": "Session has no execution controls"})
        return session.controls

    async def set_session_mode(
        self, session_id: str, mode_id: str, **kwargs: Any
    ) -> SetSessionModeResponse:
        del kwargs
        await self.ext_method("aster/session/configure", {"sessionId": session_id, "mode": mode_id})
        return SetSessionModeResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        received_at = perf_counter()
        options = kwargs.get("aster")
        turn_id = options.get("turnId", str(uuid4())) if isinstance(options, dict) else str(uuid4())
        if not isinstance(turn_id, str) or not turn_id.strip() or len(turn_id) > 256:
            raise RequestError.invalid_params({"reason": "turnId must be a nonempty string"})
        runtime = await self._get_runtime(session_id)
        try:
            async with runtime.turn() as session:
                # Preserve attached and replayed images; the provider decides image support.
                media_inputs = session.agent.llm.media_inputs
                if "image" not in media_inputs:
                    media_inputs = (*media_inputs, "image")
                try:
                    parsed = parse_prompt(prompt, media_inputs=media_inputs)
                except ValueError as exc:
                    raise RequestError.invalid_params({"reason": str(exc)}) from exc
                session.agent.llm.media_inputs = media_inputs
                backend = session.dispatcher._strategy.config.execution_backend
                async with session.foreground.prompt(turn_id, received_at, backend) as turn:
                    response = await self._run_prompt(session, parsed, turn.progress, turn_id)
                    if response.stop_reason == "end_turn":
                        turn.complete()
                    return response
        except InvalidTurnIdError as exc:
            raise RequestError.invalid_params({"reason": str(exc)}) from exc
        except SessionBusyError:
            raise RequestError.invalid_request(
                {"sessionId": session_id, "reason": "A prompt is already running"}
            ) from None
        except SessionRuntimeClosedError:
            # The session was closed between _get_runtime and the turn claim.
            # It is gone as far as the client is concerned, so say so rather
            # than letting this escape as an opaque internal error.
            raise RequestError.resource_not_found(session_id) from None

    async def _run_prompt(
        self, session: _ACPSession, parsed: ParsedPrompt, progress: PromptProgress, turn_id: str
    ) -> PromptResponse:
        text = parsed.text
        session.handle.record_user_message(text, turn_id=turn_id)
        if parsed.context is not None:
            session.agent.event_manager.add(parsed.context)
        try:
            if not session.commands_sent_on_prompt:
                session.bridge.publish(_available_commands_update(session.commands.commands()))
                session.commands_sent_on_prompt = True
            info = session.handle.info
            if info.turn_count == 1 and not info.title and not info.title_is_user_set:
                title_text = parsed.user_text if parsed.user_text.strip() else text
                title = " ".join(title_text.split())[:80]
                session.handle.set_title(title)
                session.bridge.publish(
                    SessionInfoUpdate(session_update="session_info_update", title=title)
                )
            slash = (
                None if parsed.has_attachments else self._slash_invocation(session.commands, text)
            )
            if slash is None:
                result = await session.dispatcher.submit(text)
            else:
                name, raw_args = slash
                try:
                    submission = await session.dispatcher.invoke_slash(
                        session.commands, name, raw_args
                    )
                except CoercionError as exc:
                    message = f"/{name}: {exc.message}"
                    if exc.hint:
                        message += f"\n\nUsage: `/{name} {exc.hint}`"
                    progress.stop("command_error")
                    session.agent.message(message)
                    return PromptResponse(stop_reason="end_turn")
                except (GenerationError, SandboxError):
                    # Preserve generation limits and sandbox failures for the outer mapping.
                    raise
                except Exception as exc:
                    # Workspace/installed command failures still need a durable
                    # reply to the already-recorded user message.
                    logger.warning(
                        "Slash command /%s failed in session %s",
                        name,
                        session.bridge.session_id,
                        exc_info=True,
                    )
                    progress.stop("command_error")
                    session.agent.message(f"/{name} failed: {exc}")
                    return PromptResponse(stop_reason="end_turn")
                if submission is None:
                    result = None
                else:
                    slash_result, result = submission
                    if not slash_result.output_to_agent:
                        message = str(slash_result)
                        if message:
                            session.agent.message(message)
                        return PromptResponse(stop_reason="end_turn")
        except GenerationError as exc:
            # An interrupted strategy may have announced a tool without an output.
            await session.bridge.fail_open_tools("Did not finish.", title="Unfinished")
            message = str(exc)
            if message == OUTPUT_TOKENS_EXHAUSTED_MESSAGE:
                progress.stop("max_tokens")
                return PromptResponse(stop_reason="max_tokens")
            if message.startswith(_GENERATION_LIMIT_PREFIX) and (
                "max_iterations=" in message or "max_retries=" in message
            ):
                progress.stop("max_turn_requests")
                return PromptResponse(stop_reason="max_turn_requests")
            raise RequestError(-32603, message, {"details": message}) from exc
        except SandboxError as exc:
            message = str(exc)
            await session.bridge.fail_open_tools(message, title="Sandbox failed")
            raise RequestError(-32603, message, {"details": message}) from exc
        finally:
            progress.mark("dispatch_end")
        if result is None:
            await session.foreground.wait_cancelled()
            # Keep the cancellation in durable history, but do not count this
            # host-authored note as the model's first answer.
            progress.stop("cancelled")
            session.agent.message("Stopped at your request.")
            return PromptResponse(stop_reason="cancelled")
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        runtime = await self._get_runtime(session_id)
        await runtime.value.foreground.cancel()

    def _make_model(self, model: str) -> UnifiedLLM:
        if self._model_factory is not None:
            return self._model_factory(model)
        from aster.unifiedllm import get_llm_client

        return get_llm_client(model)

    async def _new_llm(self, root: Path) -> tuple[UnifiedLLM, str | None]:
        if self._prepare_model is not None:
            if self._model_preparation is None:
                self._model_preparation = asyncio.create_task(self._prepare_model())
            # Cancelling one session must not cancel another's shared preparation.
            await asyncio.shield(self._model_preparation)
        model = self._workspace_models.get(root)
        if model is not None and self._pending_model is None:
            return self._make_model(model), model
        return self._llm_factory(), self._startup_model

    def _remember_model(self, root: Path, model: str) -> None:
        from aster_cli._tui_model import save_tui_model

        try:
            save_tui_model(root, model)
        except OSError:
            raise ValueError(
                "Could not save the default model. Check permissions for Aster's settings directory."
            ) from None
        # New and loaded sessions in this workspace follow the last confirmed
        # choice. Other workspaces retain their own startup selection.
        self._workspace_models[root] = model

    def _remember_prepared_model(self, root: Path, model: str | None) -> None:
        if model is not None and self._pending_model == model:
            try:
                self._remember_model(root, model)
            except ValueError as exc:
                raise RequestError.invalid_params({"reason": str(exc)}) from exc
            self._pending_model = None

    async def _create_runtime(
        self,
        handle: SessionHandle,
        root: Path,
        mcp_servers: list[Any] | None,
        *,
        llm: UnifiedLLM | None = None,
        model: str | None = None,
        available: bool = True,
    ) -> SessionRuntime[_ACPSession]:
        if llm is None:
            with phase("session_model"):
                llm, model = await self._new_llm(root)
        if self._client is None:
            await llm.aclose()
            raise RequestError.internal_error({"reason": "ACP client is not connected"})
        agent: CodingAgent | None = None
        commands: CodingSlashCommandRegistry | None = None
        value: _ACPSession | None = None
        try:
            with phase("session_model_metadata"):
                await llm.refresh_model_info()
            with phase("session_mcp"):
                mcp, mcp_warnings = await self._create_mcp_tools(mcp_servers)
            with phase("session_agent"):
                agent = CodingAgent(
                    llm=llm,
                    cwd=root,
                    storage=handle.storage,
                    # Same anchoring as the session store above: one ACP process
                    # serves many workspaces, so project-local paths follow the
                    # session's workspace rather than the process.
                    libs_dir=root / ".aster" / "libs",
                    skills_dirs=load_coding_skills_dirs(root),
                )
                # Forks start at the selected pre-message state; ordinary loads
                # use their last completed state, never a pre-last-turn checkpoint.
                handle.storage.restore_current_snapshot(agent)
                # Saved conversations can come from another workspace or host.
                agent.refresh_host_context()
            registration_warnings: list[str] = []
            for name, tool in mcp.items():
                registry_name = f"mcp.{name}"
                try:
                    agent.skills.register(registry_name, tool)
                    agent.skills.activate([registry_name])
                except ValueError as exc:
                    # A server name can collide with a core agent attribute
                    # (`shell`, `repo`) or a reserved one (`runtime`). Skipping
                    # it keeps the session usable — the same contract the
                    # connect path already offers for an unreachable server —
                    # instead of failing session/new with an opaque error.
                    registration_warnings.append(f"MCP server {name!r} was not registered: {exc}")
            coding_strategy = CodingReplyStrategy(
                config=CodeActConfig(
                    cell_timeout=1800.0,
                    execution_backend="sandbox" if self._sandbox else "inprocess",
                    sandbox=self._sandbox_config.model_copy(update={"workspace": str(root)}),
                )
            )
            dispatcher = InteractiveSessionDispatcher(agent, strategy=coding_strategy)
            bridge = ACPEventBridge(
                agent,
                self._client,
                handle.id,
                stream_messages=self._stream_messages,
                structured_progress=self._structured_progress,
            )
            # Every client gets execution approval. It uses the standard ACP
            # permission request, so it does not depend on an ASTER capability.
            controls = RuntimeControls(
                agent,
                self._client,
                handle.id,
                publish=bridge.publish_best_effort,
                execution_backend="sandbox" if self._sandbox else "native",
                change_execution_backend=dispatcher.set_execution_backend,
                model=model,
                model_factory=self._make_model,
                remember_model=lambda selected: self._remember_model(root, selected),
                waiting=bridge.set_wait_reason,
                sandbox_roots=[
                    (rule.path, rule.access == "read_write") for rule in self._sandbox_config.allow
                ],
            )
            agent._execution_control = controls
            dispatcher.on_result = bridge.observe_result
            commands = CodingSlashCommandRegistry(agent)
            value = _ACPSession(
                handle,
                agent,
                dispatcher,
                bridge,
                commands,
                controls=controls,
                startup_warnings=(*mcp_warnings, *registration_warnings),
                trust_notice=workspace_trust.restricted_notice(root),
                unsubscribe_activity=record_activity(agent.event_manager, handle),
            )
            commands.set_on_change(
                lambda available: bridge.publish(_available_commands_update(available)),
            )
            try:
                runtime = await self._sessions.add(handle.id, value, available=available)
                return runtime
            except ValueError:
                raise RequestError.invalid_request(
                    {"sessionId": handle.id, "reason": "Session is already loaded"}
                ) from None
        except BaseException:
            if value is not None:
                await value.close()
            elif agent is not None:
                if commands is not None:
                    commands.close()
                await agent.close()
            else:
                await llm.aclose()
            raise

    async def _create_mcp_tools(
        self,
        mcp_servers: list[Any] | None,
    ) -> tuple[dict[str, MCPTool], tuple[str, ...]]:
        servers = list(mcp_servers or [])
        supported_types = (McpServerStdio, HttpMcpServer, SseMcpServer)
        tools: dict[str, MCPTool] = {}
        warnings: list[str] = []
        seen_names: set[str] = set()
        for server in servers:
            name = getattr(server, "name", "<unnamed>")
            if not isinstance(server, supported_types):
                warnings.append(
                    f"MCP server {name!r} was not loaded: unsupported ACP server type "
                    f"{type(server).__name__}."
                )
                continue
            if name in seen_names:
                warnings.append(
                    f"MCP server {name!r} was not loaded: another server has the same name."
                )
                continue
            seen_names.add(name)
            try:
                if isinstance(server, McpServerStdio):
                    env = {item.name: item.value for item in server.env}
                    tools[name] = await MCPManager.create_stdio_server(
                        name,
                        command=server.command,
                        args=server.args,
                        env=env,
                    )
                elif isinstance(server, (HttpMcpServer, SseMcpServer)):
                    headers = {item.name: item.value for item in server.headers}
                    tools[name] = await MCPManager.create_url_server(
                        name,
                        server.url,
                        headers=headers,
                        transport=(
                            "streamable-http" if isinstance(server, HttpMcpServer) else "sse"
                        ),
                    )
            except Exception as exc:
                warnings.append(f"MCP server {name!r} was not loaded: {exc}")
        return tools, tuple(warnings)

    async def _get_runtime(self, session_id: str) -> SessionRuntime[_ACPSession]:
        try:
            return await self._sessions.get(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None

    @staticmethod
    def _defer_bootstrap_updates(session: _ACPSession) -> None:
        """Publish bootstrap updates after the session response can reach clients such as Zed."""

        async def _publish() -> None:
            await asyncio.sleep(0)
            session.bridge.publish_best_effort(
                _available_commands_update(session.commands.commands())
            )
            if session.controls is not None:
                session.controls.publish()
            session.bridge.refresh_tasks()
            session.bridge.refresh_usage()
            if session.trust_notice:
                session.bridge.publish_best_effort(
                    update_agent_message(text_block(session.trust_notice))
                )
            if session.startup_warnings:
                details = "\n".join(f"- {warning}" for warning in session.startup_warnings)
                session.bridge.publish_best_effort(
                    update_agent_message(
                        text_block(
                            "Aster started without one or more MCP servers. The session is still "
                            f"usable.\n\n{details}"
                        )
                    )
                )
            await session.bridge.flush()

        task = asyncio.create_task(_publish(), name="aster-acp-bootstrap")
        session.notification_tasks.add(task)

        def _finished(done: asyncio.Task[None]) -> None:
            session.notification_tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(_finished)

    async def _replay_session(self, handle: SessionHandle) -> None:
        if self._client is None:
            return
        updates = await asyncio.to_thread(lambda: transcript_updates(handle.transcript()))
        for update in updates:
            await self._client.session_update(handle.id, update)

    @staticmethod
    def _validate_workspace(cwd: str, additional_directories: list[str] | None) -> Path:
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "Additional directories are not supported"}
            )
        root = Path(cwd).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise RequestError.invalid_params(
                {"cwd": cwd, "reason": "cwd must be an existing absolute directory"}
            )
        return root.resolve()

    @staticmethod
    def _store(root: Path) -> SessionStore:
        if workspace_trust.is_trusted(root):
            return SessionStore(root / ".aster" / "sessions")
        # Loading a session evaluates its saved context expressions, so session
        # files shipped in an untrusted folder are never opened. Conversations
        # held there are kept in the user directory instead.
        folder = hashlib.sha256(os.path.normcase(str(root)).encode()).hexdigest()[:16]
        return SessionStore(get_user_dir("untrusted-sessions", folder))

    @staticmethod
    def _prompt_text(prompt: list[Any]) -> str:
        parts: list[str] = []
        for block in prompt:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
            elif block_type == "resource_link":
                parts.append(f"Resource {block.name}: {block.uri}")
            else:
                raise RequestError.invalid_params(
                    {"reason": f"Unsupported prompt content type: {block_type!r}"}
                )
        text = "\n\n".join(parts)
        if not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})
        return text

    @staticmethod
    def _slash_invocation(
        commands: CodingSlashCommandRegistry,
        text: str,
    ) -> tuple[str, str] | None:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        command_text = stripped[1:]
        parts = command_text.split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        if not name or commands.get(name) is None:
            return None
        return name, parts[1] if len(parts) == 2 else ""

    async def close(self) -> None:
        try:
            await self._sessions.close()
        finally:
            if self._model_preparation is not None:
                self._model_preparation.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await self._model_preparation


async def serve(
    llm_factory: Callable[[], UnifiedLLM],
    *,
    model_factory: Callable[[str], UnifiedLLM] | None = None,
    model: str | None = None,
    sandbox: bool = False,
    prepare_model: Callable[[], Awaitable[None]] | None = None,
) -> None:
    reader, writer = await protocol_streams()
    with diagnostic_stdout():
        adapter = CodingACPAdapter(
            llm_factory,
            model_factory=model_factory,
            model=model,
            sandbox=sandbox,
            prepare_model=prepare_model,
        )
        try:
            # Bind protocol streams before redirecting ordinary Python output.
            # Provider initialization, model calls and cleanup can all print.
            # session/close is unstable in the router and must match the
            # capability advertised by initialize(). session/list is stable.
            await run_agent(
                cast(Agent, adapter),
                input_stream=writer,
                output_stream=reader,
                use_unstable_protocol=True,
            )
        finally:
            with suppress(Exception):
                await adapter.close()


def _available_commands_update(commands: tuple[CodingSlashCommand, ...]):
    available: list[AvailableCommand] = []
    for command in commands:
        input_spec = (
            AvailableCommandInput(
                UnstructuredCommandInput(hint=command.argument_hint),
            )
            if command.argument_hint
            else None
        )
        available.append(
            AvailableCommand(
                name=command.name,
                description=command.description,
                input=input_spec,
            )
        )
    return update_available_commands(available)
