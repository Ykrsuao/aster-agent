# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live-session model selection and an approval gate for generated execution.

The gate is deliberately at the Python execution seam. A shell-only check
cannot control CodeAct, whose generated Python can open files and processes
directly. Planning dispatches bounded host file requests and disables generated
execution; the host Python interpreter is not a read-only sandbox.
"""

from __future__ import annotations

import asyncio
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from acp import text_block, tool_content
from acp.interfaces import Client
from acp.schema import (
    CurrentModeUpdate,
    PermissionOption,
    SessionMode,
    SessionModeState,
    ToolCallUpdate,
)

from aster.interactive import apply_model_limits
from aster_acp.approval_preview import approval_preview

if TYPE_CHECKING:
    from aster_cli.coding.agent import CodingAgent

    from aster.unifiedllm import UnifiedLLM

_MAX_APPROVAL_CODE = 128 * 1024
# Explicit bidirectional controls reorder how the surrounding text is displayed.
_BIDI_CONTROLS = frozenset("\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def _has_unreviewable_characters(code: str) -> bool:
    """True when the cell would not be displayed exactly as Python reads it.

    A terminal client strips control characters before showing a cell, and an
    escape sequence can swallow the text that follows it. Python still executes
    that text, and reads a lone carriage return as a line break, so a reviewer
    could approve code they were never shown.
    """
    visible = code.replace("\r\n", "\n")
    return any(
        character in _BIDI_CONTROLS
        or (unicodedata.category(character) == "Cc" and character not in "\n\t")
        for character in visible
    )


_REASONING_UNCHANGED = object()
_PERMISSION_MODES = {
    "auto": ("code", "session"),
    "manual": ("code", "ask"),
    "accept_edits": ("code", "accept_edits"),
    "plan": ("plan", "ask"),
}
_MODES = (
    SessionMode(
        id="code",
        name="Code",
        description="Use tools and Python with the selected approval policy.",
    ),
    SessionMode(
        id="plan",
        name="Plan (read-only)",
        description="Inspect workspace files and plan; generated code and file changes are disabled.",
    ),
)


def configured_models() -> list[str]:
    """List chat aliases without exposing their routes, credentials or settings."""
    from aster.unifiedllm.registry import MODELS, ensure_loaded

    ensure_loaded()
    return sorted(
        name for name, config in MODELS.items() if config.get("client_type") != "decision"
    )


def _load_model(name: str) -> UnifiedLLM:
    from aster.unifiedllm import get_llm_client

    return get_llm_client(name)


def _reasoning_status(
    selection: str | None,
    default: str | None,
    levels: Sequence[str] | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    effort = selection or default
    if effort is None:
        reasoning = config.get("reasoning")
        effort = config.get("reasoning_effort") or (
            reasoning.get("effort") if isinstance(reasoning, Mapping) else None
        )
    return {
        "reasoningLevel": effort if isinstance(effort, str) else None,
        "reasoningSelection": selection,
        "reasoningDefault": default,
        "reasoningLevels": list(levels) if levels is not None else None,
    }


def _configured_reasoning(name: str) -> dict[str, Any]:
    """Preview declared choices without constructing a client or probing its route."""
    from aster.unifiedllm.reasoning import ReasoningConfig
    from aster.unifiedllm.registry import MODELS, ensure_loaded

    ensure_loaded()
    config = MODELS.get(name, {})
    try:
        declaration = ReasoningConfig(
            levels=config.get("reasoning_levels"), default=config.get("reasoning_default")
        )
        selection = config.get("reasoning_level")
        if selection is not None:
            declaration.settings(selection)
    except (ValueError, TypeError):
        # An invalid alias must not break inspection of an otherwise usable session.
        return _reasoning_status(None, None, None, {})
    levels = None if declaration.levels is None else tuple(declaration.levels)
    return _reasoning_status(selection, declaration.default, levels, config)


class RuntimeControls:
    """Live runtime settings; execution consent is never restored.

    Hosts serialize model and execution-backend changes with foreground turns.
    Permission-only changes are synchronous and apply at the next authorization.
    Attach this object as ``agent._execution_control``; CodingReplyStrategy
    consults it before any model-authored execution, native or sandbox.
    """

    def __init__(
        self,
        agent: CodingAgent,
        client: Client,
        session_id: str,
        *,
        publish: Callable[[Any], None],
        execution_backend: str = "native",
        change_execution_backend: Callable[[str], Awaitable[None]] | None = None,
        model: str | None = None,
        model_factory: Callable[[str], UnifiedLLM] | None = None,
        model_names: Callable[[], Sequence[str]] = configured_models,
        remember_model: Callable[[str], None] | None = None,
        waiting: Callable[[str], None] | None = None,
        sandbox_roots: Sequence[tuple[str, bool]] = (),
    ) -> None:
        self.agent = agent
        self.client = client
        self.session_id = session_id
        self.execution_backend = execution_backend
        self._change_execution_backend = change_execution_backend
        self._publish = publish
        self._model_factory = model_factory or _load_model
        self._model_names = model_names
        self._remember_model = remember_model
        self._waiting = waiting
        # Directories the operator granted to the sandbox, as (path, writable).
        self._sandbox_roots = tuple(sandbox_roots)
        self._model_id = model or agent.llm.model
        self._model_id_known = model is not None
        self.mode = "code"
        self.approval = "ask"
        self._pending = 0
        self._approval_lock = asyncio.Lock()
        from aster_cli.tools.workspace_files import WorkspaceFiles

        self._plan_files = WorkspaceFiles(agent.cwd)
        self._permission_revision = 0

    def instructions(self) -> str:
        if self._recovery_pending():
            return (
                "Actions from an interrupted session still need human review. "
                "Generated execution is paused. Inspect their effects using one literal "
                "await self.files.read/list/search(...) request per cell, then ask the user "
                "to review them with /recovery. Do not repeat an earlier action."
            )
        if self.mode == "plan":
            return (
                "The user selected Plan mode. Inspect the workspace using one direct literal "
                "request per execute_python cell: await self.files.list(path='.'), "
                "await self.files.search('literal text', path='.', glob='*.py'), or "
                "await self.files.read('path', start_line=1, end_line=200). The host handles "
                "these requests without executing Python; results are shown directly. Do not "
                "assign or print them, compute arguments, combine calls, or use other tools. "
                "Only files in this workspace are accessible; links are refused and results "
                "are bounded. A truncated or skipped search is incomplete evidence. Arbitrary "
                "Python, shell commands, external tools and file changes remain disabled. "
                "Present the plan with the separate return_result JSON tool, grounded in "
                "the supplied context and files actually inspected."
            )
        if self.approval == "session":
            return "The user has allowed generated execution for this live session."
        if self.approval == "accept_edits":
            return (
                "The user selected Accept edits for this live session. A single direct "
                "await self.shell.read(path), await self.shell.write_file(path, content), or "
                "await self.shell.replace(path, old_text, new_text) with literal arguments "
                "is handled as a permitted host file operation. Its result is shown directly; "
                "keep the call in its own cell without assignments or printing. The read "
                "operation may include a literal lines=(start, end). Shell commands, Python "
                "file I/O, computed arguments, and cells containing any other code still "
                "require execution approval. A rejected cell did not run; do not retry it "
                "or bypass the user's choice."
            )
        return (
            "Generated Python requires approval before execution. Keep each cell focused and "
            "reviewable. A rejected cell did not run; do not retry it or bypass the user's choice."
        )

    def mode_state(self) -> SessionModeState:
        return SessionModeState(current_mode_id=self.mode, available_modes=list(_MODES))

    def status(self) -> dict[str, Any]:
        names = list(dict.fromkeys([self._model_id, *self._model_names()]))
        config = getattr(self.agent.llm, "config", {})
        reasoning = _reasoning_status(
            getattr(self.agent.llm, "reasoning_level", None),
            getattr(self.agent.llm, "reasoning_default", None),
            getattr(self.agent.llm, "reasoning_levels", None),
            config if isinstance(config, Mapping) else {},
        )
        return {
            "model": self._model_id,
            "modelName": self.agent.llm.model,
            "contextWindow": getattr(self.agent.llm, "context_window", None),
            **reasoning,
            "models": [
                {
                    "id": name,
                    "name": name,
                    **(reasoning if name == self._model_id else _configured_reasoning(name)),
                }
                for name in names
            ],
            "mode": self.mode,
            "modes": [mode.model_dump(by_alias=True, exclude_none=True) for mode in _MODES],
            "approval": self.approval,
            "permissionMode": (
                "plan"
                if self.mode == "plan"
                else {"ask": "manual", "session": "auto", "accept_edits": "accept_edits"}[
                    self.approval
                ]
            ),
            "executionBackend": self.execution_backend,
            **(
                {"executionBackends": ["native", "sandbox"]}
                if self._change_execution_backend
                else {}
            ),
            "executionEnabled": self.mode != "plan",
            "approvalPending": self._pending > 0,
        }

    def publish(self) -> None:
        self._publish(
            CurrentModeUpdate(
                session_update="current_mode_update",
                current_mode_id=self.mode,
                field_meta={"aster": {"runtime": self.status()}},
            )
        )

    def set_permission_mode(self, permission_mode: str) -> dict[str, Any]:
        """Change future cell authorization without interrupting accepted work."""
        if not isinstance(permission_mode, str) or permission_mode not in _PERMISSION_MODES:
            raise ValueError("Permission mode must be 'auto', 'manual', 'accept_edits', or 'plan'.")
        if self._pending:
            raise ValueError("Resolve or cancel the pending approval before changing settings.")
        self.mode, self.approval = _PERMISSION_MODES[permission_mode]
        self._permission_revision += 1
        self.publish()
        return self.status()

    async def configure(
        self,
        *,
        model: str | None = None,
        mode: str | None = None,
        approval: str | None = None,
        permission_mode: str | None = None,
        reasoning_level: Any = _REASONING_UNCHANGED,
        execution_backend: str | None = None,
    ) -> dict[str, Any]:
        """Validate changes; only permission-only updates may overlap prompts."""
        if (
            permission_mode is not None
            and all(value is None for value in (model, mode, approval, execution_backend))
            and reasoning_level is _REASONING_UNCHANGED
        ):
            return self.set_permission_mode(permission_mode)
        if execution_backend is not None:
            if execution_backend not in {"native", "sandbox"}:
                raise ValueError("Execution backend must be 'native' or 'sandbox'.")
            if self._change_execution_backend is None:
                raise ValueError("This host does not support changing the execution backend.")
            if (
                any(value is not None for value in (model, mode, approval, permission_mode))
                or reasoning_level is not _REASONING_UNCHANGED
            ):
                raise ValueError("Change the execution backend separately from other settings.")
        if permission_mode is not None:
            if permission_mode not in _PERMISSION_MODES:
                raise ValueError(
                    "Permission mode must be 'auto', 'manual', 'accept_edits', or 'plan'."
                )
            if mode is not None or approval is not None:
                raise ValueError("Choose permissionMode without also setting mode or approval.")
            mode, approval = _PERMISSION_MODES[permission_mode]
        if mode is not None and mode not in {"code", "plan"}:
            raise ValueError("Mode must be 'code' or 'plan'.")
        if approval is not None and approval not in {"ask", "session", "accept_edits"}:
            raise ValueError("Approval must be 'ask', 'session', or 'accept_edits'.")
        if (mode or self.mode) == "plan" and approval in {"session", "accept_edits"}:
            raise ValueError("Switch to Code mode before allowing execution for the session.")
        if model is not None and model not in {self._model_id, *self._model_names()}:
            raise ValueError("Choose one of the configured models shown in this session.")
        if self._pending:
            raise ValueError("Resolve or cancel the pending approval before changing settings.")
        if self.agent.queue_manager.running_handles():
            raise ValueError("Stop or finish background jobs before changing runtime settings.")

        if execution_backend is not None:
            if execution_backend != self.execution_backend:
                await self._change_execution_backend(execution_backend)
                self.execution_backend = execution_backend
            self.publish()
            return self.status()

        remember_model = self._remember_model
        if model == self._model_id and not self._model_id_known:
            # An embedding host may supply only a client factory. Its wire ID
            # is for display and may not reproduce the client's alias/route.
            remember_model = None
        previous = None
        if model is not None and model != self._model_id:
            previous = self.agent.llm
            candidate = self._model_factory(model)
            try:
                await candidate.refresh_model_info()
                if reasoning_level is not _REASONING_UNCHANGED:
                    candidate.set_reasoning_level(reasoning_level)
                self.agent.set_llm(candidate)
                apply_model_limits(self.agent)
                if remember_model is not None:
                    # Keep persistence inside the rollback boundary and without
                    # an await: cancellation must not save an uncommitted choice.
                    remember_model(model)
            except BaseException:
                self.agent.set_llm(previous)
                apply_model_limits(self.agent)
                await candidate.aclose()
                raise
            self._model_id = model
            self._model_id_known = True
        elif reasoning_level is not _REASONING_UNCHANGED:
            llm = self.agent.llm
            previous_level = llm.reasoning_level
            # Validation happens before assignment, including unknown and
            # unsupported routes. Do not silently accept an ignored setting.
            llm.set_reasoning_level(reasoning_level)
            try:
                apply_model_limits(self.agent)
                if model is not None and remember_model is not None:
                    remember_model(model)
            except BaseException:
                llm.set_reasoning_level(previous_level)
                apply_model_limits(self.agent)
                raise
        elif model is not None and remember_model is not None:
            # Explicitly choosing the current model also makes it the default.
            remember_model(model)
        if mode is not None:
            self.mode = mode
            if mode == "plan":
                self.approval = "ask"
        if approval is not None:
            self.approval = approval
        if mode is not None or approval is not None:
            self._permission_revision += 1
        # Publish all effective settings before asynchronous cleanup. If that
        # await is cancelled, the UI still sees the configuration in use.
        self.publish()
        if previous is not None:
            # Cleanup failure cannot undo a model already committed above.
            try:
                await previous.aclose()
            except Exception:
                pass
        return self.status()

    def file_edit_executor(
        self,
        code: str,
        *,
        timeout: float | None = None,
        host_path: Callable[[str], str] | None = None,
    ) -> Any:
        """Prepare a host file request without granting execution of its Python spelling."""
        if self.mode == "plan" or self._recovery_pending():
            from aster_acp._plan_files import plan_file_executor

            revision = self._permission_revision
            return plan_file_executor(
                code,
                files=self._plan_files,
                allowed=lambda: (
                    (self.mode == "plan" or self._recovery_pending())
                    and self._permission_revision == revision
                ),
                timeout=timeout,
                host_path=host_path,
            )
        if self.mode != "code" or self.approval != "accept_edits":
            return None
        from aster_acp._file_edits import file_edit_executor

        shell = self.agent._base_shell
        return file_edit_executor(
            code,
            cwd=shell.cwd,
            event_manager=self.agent.event_manager,
            max_file_bytes=shell.max_file_bytes,
            timeout=timeout,
            host_path=host_path,
            workspace=getattr(self.agent, "cwd", None),
            extra_roots=self._sandbox_roots if self.execution_backend == "sandbox" else (),
            journal=getattr(self.agent, "_action_journal", None),
        )

    def _recovery_pending(self) -> bool:
        journal = getattr(self.agent, "_action_journal", None)
        return journal is not None and bool(journal.pending())

    async def authorize(self, code: str, *, tool_call_id: str | None = None) -> str | None:
        """Return a refusal reason or None; cancellation always propagates.

        Permission applies to the complete cell shown in the request. A session
        grant is scoped to this RuntimeControls instance and disappears when
        the session is closed/reconnected. Unknown client choices fail closed.
        """
        if self._recovery_pending():
            return self.instructions()
        if self.mode == "plan":
            return (
                "Plan mode disables generated execution. This code was not run. "
                "Use one literal self.files.read/list/search request to inspect the workspace, "
                "or the separate return_result JSON tool to present your plan."
            )
        if self.approval == "session":
            return None
        if len(code.encode("utf-8")) > _MAX_APPROVAL_CODE:
            return "This cell is too large to review. Split it into smaller, focused cells."
        if _has_unreviewable_characters(code):
            return (
                "This cell contains control characters that cannot be shown for review. "
                "The code was not run. Write such characters as escapes inside string "
                "literals, for example '\\x1b', and separate lines with plain newlines."
            )
        async with self._approval_lock:
            if self._recovery_pending():
                return self.instructions()
            if self.mode == "plan":
                return (
                    "Plan mode disables generated execution. This code was not run. "
                    "Use the separate return_result JSON tool to present your plan."
                )
            # A concurrent cell can have obtained a live-session grant while
            # this one waited for the preceding dialog.
            if self.approval == "session":
                return None
            self._pending += 1
            self.publish()
            if self._waiting:
                self._waiting("Waiting for execution approval")
            try:
                preview = approval_preview(code)
                response = await self.client.request_permission(
                    self.session_id,
                    ToolCallUpdate(
                        tool_call_id=tool_call_id or f"approval-{uuid4().hex}",
                        title=f"{preview['summary']}?",
                        kind="other",
                        status="pending",
                        content=[tool_content(text_block(code))],
                        raw_input={"code": code},
                        field_meta={
                            "aster": {
                                "approvalScope": "python_execution",
                                "executionBackend": self.execution_backend,
                                "approvalPreview": preview,
                            }
                        },
                    ),
                    [
                        PermissionOption(
                            option_id="allow_once", name="Allow once", kind="allow_once"
                        ),
                        PermissionOption(
                            option_id="allow_session",
                            name="Allow all code for this session",
                            kind="allow_always",
                        ),
                        PermissionOption(option_id="reject", name="Reject", kind="reject_once"),
                    ],
                )
                outcome = response.outcome
                if self._recovery_pending():
                    return self.instructions()
                option = getattr(outcome, "option_id", None)
                if getattr(outcome, "outcome", None) == "selected":
                    if option == "allow_once":
                        return None
                    if option == "allow_session":
                        self.approval = "session"
                        return None
                return "The user rejected this execution. The code was not run."
            except Exception:
                return "Execution approval could not be obtained. The code was not run."
            finally:
                self._pending -= 1
                self.publish()
                if self._waiting:
                    self._waiting("Processing response")
